#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import DEFAULT_BETA, compute_reward, verify_binary
from src.eval_framework import magnitude_pair_bucket, pilot_bar_summary, sign_pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated binaries and compute reward metrics")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint identifier/path (metadata only)")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="JSONL predictions (defaults to outputs/predictions/<split>.jsonl)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output report JSON path")
    parser.add_argument("--summary-out", type=Path, default=None, help="Output compact learning summary JSON")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return rows


def _materialize_binary(record: dict[str, Any], tmp_dir: Path, idx: int) -> Path | None:
    parse_ok = record.get("parse_ok")
    if parse_ok is False:
        return None

    if "binary_path" in record and record["binary_path"]:
        return Path(str(record["binary_path"]))

    if "binary_b64" in record and record["binary_b64"]:
        out_path = tmp_dir / f"pred_{idx:06d}"
        out_path.write_bytes(base64.b64decode(record["binary_b64"]))
        out_path.chmod(0o755)
        return out_path

    return None


def _expected_stdout(record: dict[str, Any]) -> str:
    if "expected_stdout" in record and record["expected_stdout"] is not None:
        return str(record["expected_stdout"])
    if "a" in record and "b" in record and record["a"] is not None and record["b"] is not None:
        return f"{int(record['a']) + int(record['b'])}\n"
    raise ValueError("Prediction record must include expected_stdout or operands a/b.")


def _empty_verification(expected_stdout: str, reason: str) -> dict[str, Any]:
    return {
        "binary_path": None,
        "binary_sha256": None,
        "byte_length": 0,
        "expected_stdout": expected_stdout,
        "static": {
            "file_ok": False,
            "file_output": reason,
            "otool_ok": False,
            "otool_output": reason,
            "is_macho_arm64": False,
        },
        "execution": {
            "executed": False,
            "timed_out": False,
            "returncode": None,
            "stdout": "",
            "stderr": reason,
        },
        "metrics": {
            "macho_valid": 0,
            "exec_ok": 0,
            "stdout_correct": 0,
            "timeout_or_crash": 1,
        },
        "pass": False,
    }


def _init_slice_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "parse_success": 0,
        "exec_success": 0,
        "answer_correct": 0,
        "patch_exact": 0,
    }


def _update_slice(bucket: dict[str, Any], *, parse_ok: int, exec_ok: int, answer_ok: int, patch_exact: int) -> None:
    bucket["count"] += 1
    bucket["parse_success"] += parse_ok
    bucket["exec_success"] += exec_ok
    bucket["answer_correct"] += answer_ok
    bucket["patch_exact"] += patch_exact


def _finalize_slices(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    final: dict[str, dict[str, Any]] = {}
    for key, bucket in raw.items():
        count = max(1, int(bucket["count"]))
        final[key] = {
            "count": int(bucket["count"]),
            "parse_success_rate": float(bucket["parse_success"]) / count,
            "exec_success_rate": float(bucket["exec_success"]) / count,
            "answer_accuracy": float(bucket["answer_correct"]) / count,
            "patch_exact_match_rate": float(bucket["patch_exact"]) / count,
        }
    return final


def _coerce_parse_ok(record: dict[str, Any]) -> int:
    if "parse_ok" in record:
        return int(bool(record.get("parse_ok")))
    if "a_pred" in record and "b_pred" in record and record.get("a_pred") is not None and record.get("b_pred") is not None:
        return 1
    if record.get("mode") == "deterministic_baseline":
        return 1
    if record.get("binary_b64") or record.get("binary_path"):
        return 1
    return 0


def _load_latest_training_losses(ckpt: Path) -> dict[str, float | None]:
    if not ckpt.exists() or not ckpt.is_dir():
        return {"train_loss": None, "val_loss": None}

    train_log = ckpt / "train_log.jsonl"
    eval_log = ckpt / "eval_log.jsonl"

    train_loss: float | None = None
    val_loss: float | None = None

    if train_log.exists():
        try:
            lines = [line for line in train_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                train_loss = float(json.loads(lines[-1]).get("loss"))
        except Exception:
            train_loss = None

    if eval_log.exists():
        try:
            lines = [line for line in eval_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            parsed_rows = [json.loads(line) for line in lines]
            full_rows = [row for row in parsed_rows if row.get("eval_scope") == "full"]
            if full_rows:
                val_loss = float(full_rows[-1].get("val_loss"))
            elif parsed_rows:
                val_loss = float(parsed_rows[-1].get("val_loss"))
        except Exception:
            val_loss = None

    return {"train_loss": train_loss, "val_loss": val_loss}


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions or Path(f"outputs/predictions/{args.split}.jsonl")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    predictions = _load_jsonl(predictions_path)

    details: list[dict[str, Any]] = []
    reward_values: list[float] = []
    sum_valid = 0
    sum_exec = 0
    sum_correct = 0
    sum_exact_match = 0
    sum_parse_ok = 0
    sum_patch_exact = 0
    operand_mae_sum = 0.0
    operand_mae_count = 0

    template_slices: dict[str, dict[str, Any]] = {}
    magnitude_slices: dict[str, dict[str, Any]] = {}
    sign_slices: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="exe_coder_eval_") as tmp:
        tmp_dir = Path(tmp)
        for idx, record in enumerate(predictions):
            expected = _expected_stdout(record)
            parse_ok = _coerce_parse_ok(record)

            binary_path = _materialize_binary(record, tmp_dir, idx)
            if binary_path is None:
                verification = _empty_verification(expected, reason="missing_or_unparseable_binary")
            else:
                verification = verify_binary(binary_path, expected, timeout_seconds=args.timeout)

            kl_to_sft = float(record.get("kl_to_sft", 0.0))
            reward = compute_reward(verification, kl_to_sft=kl_to_sft, beta=args.beta)

            predicted_sha = verification.get("binary_sha256")
            target_sha = record.get("binary_sha256")
            exact_byte_match = int(predicted_sha is not None and target_sha is not None and target_sha == predicted_sha)

            a_true = int(record["a"]) if record.get("a") is not None else None
            b_true = int(record["b"]) if record.get("b") is not None else None
            a_pred = record.get("a_pred")
            b_pred = record.get("b_pred")
            if a_pred is not None:
                a_pred = int(a_pred)
            if b_pred is not None:
                b_pred = int(b_pred)

            patch_exact = int(
                parse_ok == 1
                and a_true is not None
                and b_true is not None
                and a_pred is not None
                and b_pred is not None
                and a_true == a_pred
                and b_true == b_pred
            )

            if parse_ok == 1 and a_true is not None and b_true is not None and a_pred is not None and b_pred is not None:
                operand_mae_sum += (abs(a_pred - a_true) + abs(b_pred - b_true)) / 2.0
                operand_mae_count += 1

            exec_ok = int(verification["metrics"]["exec_ok"])
            answer_ok = int(verification["metrics"]["stdout_correct"])

            sum_valid += verification["metrics"]["macho_valid"]
            sum_exec += exec_ok
            sum_correct += answer_ok
            sum_exact_match += exact_byte_match
            sum_parse_ok += parse_ok
            sum_patch_exact += patch_exact
            reward_values.append(float(reward["reward"]))

            template_key = str(record.get("prompt_template") or "unknown")
            magnitude_key = magnitude_pair_bucket(
                str(record.get("a_magnitude_bucket") or "unknown"),
                str(record.get("b_magnitude_bucket") or "unknown"),
            )
            sign_key = sign_pattern(a_true, b_true)

            template_slices.setdefault(template_key, _init_slice_bucket())
            magnitude_slices.setdefault(magnitude_key, _init_slice_bucket())
            sign_slices.setdefault(sign_key, _init_slice_bucket())

            _update_slice(template_slices[template_key], parse_ok=parse_ok, exec_ok=exec_ok, answer_ok=answer_ok, patch_exact=patch_exact)
            _update_slice(magnitude_slices[magnitude_key], parse_ok=parse_ok, exec_ok=exec_ok, answer_ok=answer_ok, patch_exact=patch_exact)
            _update_slice(sign_slices[sign_key], parse_ok=parse_ok, exec_ok=exec_ok, answer_ok=answer_ok, patch_exact=patch_exact)

            details.append(
                {
                    "id": record.get("id", f"sample_{idx:06d}"),
                    "prompt": record.get("prompt"),
                    "pred_patch_text": record.get("pred_patch_text"),
                    "parse_ok": bool(parse_ok),
                    "a_pred": a_pred,
                    "b_pred": b_pred,
                    "verification": verification,
                    "reward": reward,
                    "exact_byte_match": exact_byte_match,
                    "patch_exact_match": patch_exact,
                }
            )

    total = len(details)
    if total == 0:
        raise ValueError("No prediction records were loaded.")

    metrics = {
        "parse_success_rate": sum_parse_ok / total,
        "patch_exact_match_rate": sum_patch_exact / total,
        "operand_mae": (operand_mae_sum / operand_mae_count) if operand_mae_count > 0 else None,
        "macho_valid_rate": sum_valid / total,
        "exec_success_rate": sum_exec / total,
        "answer_accuracy": sum_correct / total,
        "exact_byte_match_rate": sum_exact_match / total,
        "avg_reward": sum(reward_values) / total,
    }
    metrics.update(_load_latest_training_losses(args.ckpt))

    slices = {
        "template": _finalize_slices(template_slices),
        "magnitude_pair": _finalize_slices(magnitude_slices),
        "sign_pattern": _finalize_slices(sign_slices),
    }

    learning_summary = pilot_bar_summary(metrics, slices)

    report = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "predictions": str(predictions_path),
        "count": total,
        "metrics": metrics,
        "slices": slices,
        "learning_summary": learning_summary,
        "reward": {
            "beta": args.beta,
            "min": min(reward_values),
            "max": max(reward_values),
        },
        "details": details,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_out = args.summary_out or args.out.with_name(f"{args.out.stem}_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "split": args.split,
                "count": total,
                "metrics": metrics,
                "learning_summary": learning_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"out": str(args.out), "summary_out": str(summary_out), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

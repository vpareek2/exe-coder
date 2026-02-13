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


def _materialize_binary(record: dict[str, Any], tmp_dir: Path, idx: int) -> Path:
    if "binary_path" in record:
        return Path(record["binary_path"])
    if "binary_b64" not in record:
        raise ValueError("Prediction record must contain either 'binary_path' or 'binary_b64'.")

    out_path = tmp_dir / f"pred_{idx:06d}"
    out_path.write_bytes(base64.b64decode(record["binary_b64"]))
    out_path.chmod(0o755)
    return out_path


def _expected_stdout(record: dict[str, Any]) -> str:
    if "expected_stdout" in record:
        return str(record["expected_stdout"])
    if "a" in record and "b" in record:
        return f"{int(record['a']) + int(record['b'])}\n"
    raise ValueError("Prediction record must include expected_stdout or operands a/b.")


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

    with tempfile.TemporaryDirectory(prefix="exe_coder_eval_") as tmp:
        tmp_dir = Path(tmp)
        for idx, record in enumerate(predictions):
            binary_path = _materialize_binary(record, tmp_dir, idx)
            expected = _expected_stdout(record)
            verification = verify_binary(binary_path, expected, timeout_seconds=args.timeout)

            kl_to_sft = float(record.get("kl_to_sft", 0.0))
            reward = compute_reward(verification, kl_to_sft=kl_to_sft, beta=args.beta)

            predicted_sha = verification["binary_sha256"]
            target_sha = record.get("binary_sha256")
            exact_byte_match = int(target_sha is not None and target_sha == predicted_sha)

            sum_valid += verification["metrics"]["macho_valid"]
            sum_exec += verification["metrics"]["exec_ok"]
            sum_correct += verification["metrics"]["stdout_correct"]
            sum_exact_match += exact_byte_match
            reward_values.append(float(reward["reward"]))

            details.append(
                {
                    "id": record.get("id", f"sample_{idx:06d}"),
                    "prompt": record.get("prompt"),
                    "verification": verification,
                    "reward": reward,
                    "exact_byte_match": exact_byte_match,
                }
            )

    total = len(details)
    if total == 0:
        raise ValueError("No prediction records were loaded.")

    report = {
        "ckpt": str(args.ckpt),
        "split": args.split,
        "predictions": str(predictions_path),
        "count": total,
        "metrics": {
            "macho_valid_rate": sum_valid / total,
            "exec_success_rate": sum_exec / total,
            "answer_accuracy": sum_correct / total,
            "exact_byte_match_rate": sum_exact_match / total,
            "avg_reward": sum(reward_values) / total,
        },
        "reward": {
            "beta": args.beta,
            "min": min(reward_values),
            "max": max(reward_values),
        },
        "details": details,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(args.out), "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

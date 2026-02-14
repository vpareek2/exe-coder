#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import compile_addition_binary, encode_file_base64
from src.eval_framework import INT64_MAX, INT64_MIN, parse_patch_text_with_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prediction binaries")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL prompts")
    parser.add_argument("--out", type=Path, required=True, help="Output prediction JSONL")
    parser.add_argument(
        "--mode",
        choices=["deterministic_baseline", "sft_patch_model"],
        default="deterministic_baseline",
        help="Inference mode",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Path to .safetensors/.npz weights (required for sft_patch_model)")
    parser.add_argument("--max-gen-tokens", type=int, default=0, help="Override max generated patch tokens; 0 uses checkpoint default")
    parser.add_argument("--decode-min-int", type=int, default=None, help="Lower decode bound for operands (defaults to checkpoint or int64 min)")
    parser.add_argument("--decode-max-int", type=int, default=None, help="Upper decode bound for operands (defaults to checkpoint or int64 max)")
    parser.add_argument("--binaries-dir", type=Path, default=Path("outputs/predictions/bin"))
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


def _expected_stdout(row: dict[str, Any]) -> str:
    if "expected_stdout" in row:
        return str(row["expected_stdout"])
    if "a" in row and "b" in row:
        return f"{int(row['a']) + int(row['b'])}\n"
    raise ValueError("Input row must include expected_stdout or operands a/b")


def _copy_metadata(row: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "prompt_template",
        "a_magnitude_bucket",
        "b_magnitude_bucket",
        "target_format",
        "target_arch",
        "target_os",
        "toolchain",
        "compile_flags",
    ]
    return {k: row.get(k) for k in keep_keys if k in row}


def _short_error(exc: Exception, *, max_len: int = 240) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _resolve_weights_path(path: Path | None) -> Path:
    if path is None:
        raise ValueError("--weights is required for mode=sft_patch_model")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    if path.suffix not in {".safetensors", ".npz"}:
        raise ValueError(f"Unsupported weights extension for {path}; expected .safetensors or .npz")
    return path


def _load_model_cfg_from_weights(weights_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    metrics_path = weights_path.parent / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found next to weights: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_cfg = payload.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Invalid model config in {metrics_path}")
    train_cfg = payload.get("train") if isinstance(payload.get("train"), dict) else {}
    decode_cfg_raw = payload.get("decode_bounds")
    decode_cfg = decode_cfg_raw if isinstance(decode_cfg_raw, dict) else {}
    decode_min = int(decode_cfg.get("min_int", INT64_MIN))
    decode_max = int(decode_cfg.get("max_int", INT64_MAX))
    return model_cfg, train_cfg, {"min_int": decode_min, "max_int": decode_max}


def run_deterministic_baseline(args: argparse.Namespace, inputs: list[dict[str, Any]]) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.binaries_dir.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as outfile:
        for idx, row in enumerate(inputs):
            if "a" not in row or "b" not in row:
                raise ValueError("deterministic_baseline mode requires integer operands 'a' and 'b' in input JSONL")

            sample_id = row.get("id", f"sample_{idx:06d}")
            a = int(row["a"])
            b = int(row["b"])
            expected_stdout = _expected_stdout(row)

            bin_path = args.binaries_dir / sample_id
            compile_result = compile_addition_binary(a, b, bin_path)

            pred = {
                "id": sample_id,
                "prompt": row.get("prompt"),
                "a": a,
                "b": b,
                "expected_stdout": expected_stdout,
                "pred_patch_text": f"A={a};B={b}\n",
                "parse_ok": True,
                "a_pred": a,
                "b_pred": b,
                "parse_error": None,
                "binary_b64": encode_file_base64(bin_path),
                "binary_sha256": compile_result.binary_sha256,
                "toolchain": compile_result.toolchain,
                "compile_flags": compile_result.compile_flags,
                "mode": args.mode,
                **_copy_metadata(row),
            }
            outfile.write(json.dumps(pred, ensure_ascii=True) + "\n")


def run_sft_patch_model(args: argparse.Namespace, inputs: list[dict[str, Any]]) -> None:
    import mlx.core as mx
    import mlx.nn as nn

    from train import ByteTokenizer, greedy_generate_patch, make_model

    weights_path = _resolve_weights_path(args.weights)
    model_cfg, train_cfg, decode_cfg = _load_model_cfg_from_weights(weights_path)

    tokenizer = ByteTokenizer()
    model = make_model(tokenizer.vocab_size, model_cfg, mx, nn)
    model.load_weights(str(weights_path), strict=False)
    model.eval()

    max_gen_tokens = int(args.max_gen_tokens) if args.max_gen_tokens > 0 else int(train_cfg.get("max_gen_tokens", 32))
    decode_min_int = int(args.decode_min_int) if args.decode_min_int is not None else int(decode_cfg["min_int"])
    decode_max_int = int(args.decode_max_int) if args.decode_max_int is not None else int(decode_cfg["max_int"])
    if decode_min_int > decode_max_int:
        raise ValueError("--decode-min-int must be <= --decode-max-int")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.binaries_dir.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as outfile:
        for idx, row in enumerate(inputs):
            sample_id = row.get("id", f"sample_{idx:06d}")
            prompt = str(row.get("prompt", ""))

            expected_stdout = _expected_stdout(row)
            a_true = int(row["a"]) if "a" in row else None
            b_true = int(row["b"]) if "b" in row else None

            pred_patch_text = greedy_generate_patch(
                model,
                prompt,
                tokenizer,
                max_seq_len=int(model_cfg["seq_len"]),
                max_gen_tokens=max_gen_tokens,
                constrained_decode=True,
                decode_min_int=decode_min_int,
                decode_max_int=decode_max_int,
                mx=mx,
            )
            parsed = parse_patch_text_with_bounds(
                pred_patch_text,
                min_int=decode_min_int,
                max_int=decode_max_int,
            )

            pred: dict[str, Any] = {
                "id": sample_id,
                "prompt": prompt,
                "a": a_true,
                "b": b_true,
                "expected_stdout": expected_stdout,
                "pred_patch_text": pred_patch_text,
                "mode": args.mode,
                **_copy_metadata(row),
            }

            if parsed is None:
                pred.update(
                    {
                        "parse_ok": False,
                        "a_pred": None,
                        "b_pred": None,
                        "parse_error": "unparseable_patch",
                        "binary_b64": None,
                        "binary_sha256": None,
                    }
                )
            else:
                a_pred, b_pred = parsed
                bin_path = args.binaries_dir / sample_id
                try:
                    compile_result = compile_addition_binary(a_pred, b_pred, bin_path)
                    pred.update(
                        {
                            "parse_ok": True,
                            "a_pred": a_pred,
                            "b_pred": b_pred,
                            "parse_error": None,
                            "binary_b64": encode_file_base64(bin_path),
                            "binary_sha256": compile_result.binary_sha256,
                            "toolchain": compile_result.toolchain,
                            "compile_flags": compile_result.compile_flags,
                        }
                    )
                except Exception as exc:
                    pred.update(
                        {
                            "parse_ok": False,
                            "a_pred": a_pred,
                            "b_pred": b_pred,
                            "parse_error": f"compile_failed: {_short_error(exc)}",
                            "binary_b64": None,
                            "binary_sha256": None,
                        }
                    )

            outfile.write(json.dumps(pred, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    inputs = _load_jsonl(args.input)

    if args.mode == "deterministic_baseline":
        run_deterministic_baseline(args, inputs)
    elif args.mode == "sft_patch_model":
        run_sft_patch_model(args, inputs)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(json.dumps({"mode": args.mode, "input": str(args.input), "out": str(args.out), "count": len(inputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

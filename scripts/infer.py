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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prediction binaries")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL prompts with at least id/prompt/a/b")
    parser.add_argument("--out", type=Path, required=True, help="Output prediction JSONL")
    parser.add_argument(
        "--mode",
        choices=["deterministic_baseline"],
        default="deterministic_baseline",
        help="Inference mode",
    )
    parser.add_argument("--binaries-dir", type=Path, default=Path("outputs/predictions/bin"))
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    inputs = _load_jsonl(args.input)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.binaries_dir.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as outfile:
        for idx, row in enumerate(inputs):
            if "a" not in row or "b" not in row:
                raise ValueError("deterministic_baseline mode requires integer operands 'a' and 'b' in input JSONL")

            sample_id = row.get("id", f"sample_{idx:06d}")
            a = int(row["a"])
            b = int(row["b"])
            expected_stdout = f"{a + b}\n"

            bin_path = args.binaries_dir / sample_id
            compile_result = compile_addition_binary(a, b, bin_path)

            pred = {
                "id": sample_id,
                "prompt": row.get("prompt"),
                "a": a,
                "b": b,
                "expected_stdout": expected_stdout,
                "binary_b64": encode_file_base64(bin_path),
                "binary_sha256": compile_result.binary_sha256,
                "toolchain": compile_result.toolchain,
                "compile_flags": compile_result.compile_flags,
                "mode": args.mode,
            }
            outfile.write(json.dumps(pred, ensure_ascii=True) + "\n")

    print(json.dumps({"mode": args.mode, "input": str(args.input), "out": str(args.out), "count": len(inputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

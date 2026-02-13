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

from src.binary.tokenization import base64_to_hex_tokens, format_training_sequence, tokens_to_target_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PROMPT -> TARGET_BIN_HEX SFT dataset")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with binary_b64")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL for training")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap; 0 means all")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.input)

    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as outfile:
        for row in rows:
            if "binary_b64" not in row:
                raise ValueError(f"Record missing binary_b64: {row.get('id', '<no-id>')}")

            tokens = base64_to_hex_tokens(row["binary_b64"])
            target_bin_hex = tokens_to_target_string(tokens)
            sequence_text = format_training_sequence(str(row.get("prompt", "")), tokens)

            output_row = {
                "id": row.get("id"),
                "prompt": row.get("prompt"),
                "a": row.get("a"),
                "b": row.get("b"),
                "expected_stdout": row.get("expected_stdout"),
                "target_bin_hex": target_bin_hex,
                "sequence_text": sequence_text,
                "target_token_count": len(tokens),
            }
            outfile.write(json.dumps(output_row, ensure_ascii=True) + "\n")

    print(json.dumps({"input": str(args.input), "out": str(args.out), "count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

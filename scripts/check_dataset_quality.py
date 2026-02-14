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

from src.eval_framework import magnitude_pair_bucket, sign_pattern

REQUIRED_SIGNS = {"++", "+-", "-+", "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset slice coverage for Track A staged runs")
    parser.add_argument("--train", type=Path, required=True, help="Train JSONL path")
    parser.add_argument("--val", type=Path, required=True, help="Validation JSONL path")
    parser.add_argument("--test", type=Path, required=True, help="Test JSONL path")
    parser.add_argument(
        "--require-heldout-val-test",
        action="store_true",
        help="Require heldout magnitude examples to exist in both val and test splits.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
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


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sign_counts: dict[str, int] = {}
    magnitude_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}
    heldout_operand_examples = 0

    for row in rows:
        a = int(row["a"])
        b = int(row["b"])
        sign_key = sign_pattern(a, b)
        sign_counts[sign_key] = sign_counts.get(sign_key, 0) + 1

        pair_key = magnitude_pair_bucket(
            str(row.get("a_magnitude_bucket") or "unknown"),
            str(row.get("b_magnitude_bucket") or "unknown"),
        )
        magnitude_counts[pair_key] = magnitude_counts.get(pair_key, 0) + 1

        template_key = str(row.get("prompt_template") or "unknown")
        template_counts[template_key] = template_counts.get(template_key, 0) + 1

        if "heldout_magnitude" in {
            str(row.get("a_magnitude_bucket") or ""),
            str(row.get("b_magnitude_bucket") or ""),
        }:
            heldout_operand_examples += 1

    return {
        "count": len(rows),
        "sign_counts": sign_counts,
        "magnitude_pair_counts": magnitude_counts,
        "template_count": len(template_counts),
        "heldout_operand_examples": heldout_operand_examples,
    }


def validate_summary(
    *,
    summary: dict[str, dict[str, Any]],
    require_heldout_val_test: bool,
) -> list[str]:
    errors: list[str] = []

    for split_name, split in summary.items():
        if split["count"] <= 0:
            errors.append(f"{split_name}: split is empty")
        if split["template_count"] <= 0:
            errors.append(f"{split_name}: no prompt templates were recorded")
        if not split["magnitude_pair_counts"]:
            errors.append(f"{split_name}: no magnitude-pair slices were recorded")

    all_signs = set()
    for split in summary.values():
        all_signs.update(split["sign_counts"].keys())
    missing_signs = REQUIRED_SIGNS - all_signs
    if missing_signs:
        errors.append(f"global: missing sign patterns {sorted(missing_signs)}")

    if require_heldout_val_test:
        for split_name in ("val", "test"):
            if summary[split_name]["heldout_operand_examples"] <= 0:
                errors.append(f"{split_name}: expected heldout magnitude examples but found none")

    return errors


def main() -> int:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    test_rows = load_jsonl(args.test)

    summary = {
        "train": summarize_split(train_rows),
        "val": summarize_split(val_rows),
        "test": summarize_split(test_rows),
    }

    errors = validate_summary(
        summary=summary,
        require_heldout_val_test=args.require_heldout_val_test,
    )

    output = {
        "train": str(args.train),
        "val": str(args.val),
        "test": str(args.test),
        "summary": summary,
        "pass": len(errors) == 0,
        "errors": errors,
    }
    print(json.dumps(output, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

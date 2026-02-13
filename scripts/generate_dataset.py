#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import compile_addition_binary, encode_file_base64

TRAIN_TEMPLATES = [
    "Add {a} and {b}",
    "What is {a} plus {b}?",
    "Compute the sum of {a} and {b}",
    "Please add {a} to {b}",
    "Calculate {a} + {b}",
    "Return the total of {a} and {b}",
    "Find the sum: {a} and {b}",
    "Give me the result of adding {a} and {b}",
    "Perform integer addition on {a} and {b}",
    "Sum these numbers: {a}, {b}",
]

VAL_HOLDOUT_TEMPLATES = [
    "If you combine {a} with {b}, what do you get?",
    "Output only the addition result for {a} and {b}",
    "Evaluate the arithmetic total of {a} and {b}",
    "Compute this integer pair sum: {a}, {b}",
    "Produce the sum where operands are {a} and {b}",
]

TEST_HOLDOUT_TEMPLATES = [
    "Take {a} together with {b} and report the sum",
    "Add the operands {a} and {b} and print the answer",
    "Provide only the integer sum for {a} and {b}",
    "Given integers {a} and {b}, compute their total",
    "Determine the addition output for {a} plus {b}",
]

SPLIT_SEED_OFFSET = {
    "train": 0,
    "val": 100_000,
    "test": 200_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic NL->Mach-O dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--n", type=int, required=True, help="Number of examples")
    parser.add_argument("--out", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-int", type=int, default=-1000)
    parser.add_argument("--max-int", type=int, default=1000)
    parser.add_argument("--holdout-threshold", type=int, default=500)
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.5,
        help="Fraction of val/test samples using held-out magnitude bucket",
    )
    parser.add_argument(
        "--binaries-dir",
        type=Path,
        default=Path("outputs/bin"),
        help="Directory where compiled binaries are materialized",
    )
    return parser.parse_args()


def _template_pool(split: str) -> list[str]:
    if split == "train":
        return TRAIN_TEMPLATES
    if split == "val":
        return VAL_HOLDOUT_TEMPLATES
    return TEST_HOLDOUT_TEMPLATES


def _sample_in_range(rng: random.Random, low: int, high: int) -> int:
    if low > high:
        raise ValueError(f"Invalid integer range: [{low}, {high}]")
    return rng.randint(low, high)


def _sample_magnitude_bucket(
    rng: random.Random,
    *,
    split: str,
    min_int: int,
    max_int: int,
    threshold: int,
    holdout_ratio: float,
) -> tuple[int, str]:
    core_low = max(min_int, -threshold)
    core_high = min(max_int, threshold)

    holdout_candidates = [x for x in range(min_int, max_int + 1) if abs(x) > threshold]
    should_holdout = split in {"val", "test"} and holdout_candidates and (rng.random() < holdout_ratio)

    if should_holdout:
        return rng.choice(holdout_candidates), "heldout_magnitude"
    return _sample_in_range(rng, core_low, core_high), "core_magnitude"


def main() -> int:
    args = parse_args()

    if args.min_int > args.max_int:
        raise ValueError("--min-int must be <= --max-int")
    if args.n <= 0:
        raise ValueError("--n must be positive")

    rng = random.Random(args.seed + SPLIT_SEED_OFFSET[args.split])
    template_pool = _template_pool(args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    split_bin_dir = args.binaries_dir / args.split
    split_bin_dir.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as outfile:
        for idx in range(args.n):
            a, a_bucket = _sample_magnitude_bucket(
                rng,
                split=args.split,
                min_int=args.min_int,
                max_int=args.max_int,
                threshold=args.holdout_threshold,
                holdout_ratio=args.holdout_ratio,
            )
            b, b_bucket = _sample_magnitude_bucket(
                rng,
                split=args.split,
                min_int=args.min_int,
                max_int=args.max_int,
                threshold=args.holdout_threshold,
                holdout_ratio=args.holdout_ratio,
            )

            template = rng.choice(template_pool)
            prompt = template.format(a=a, b=b)
            expected_stdout = f"{a + b}\n"

            sample_id = f"{args.split}_{idx:06d}"
            binary_path = split_bin_dir / sample_id
            compile_result = compile_addition_binary(a, b, binary_path)

            record = {
                "id": sample_id,
                "split": args.split,
                "prompt": prompt,
                "a": a,
                "b": b,
                "target_format": "macho64",
                "target_arch": "arm64",
                "target_os": "macos",
                "binary_b64": encode_file_base64(binary_path),
                "binary_sha256": compile_result.binary_sha256,
                "byte_length": compile_result.byte_length,
                "expected_stdout": expected_stdout,
                "toolchain": compile_result.toolchain,
                "compile_flags": compile_result.compile_flags,
                "prompt_template": template,
                "a_magnitude_bucket": a_bucket,
                "b_magnitude_bucket": b_bucket,
            }

            outfile.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "split": args.split,
                "count": args.n,
                "out": str(args.out),
                "binaries_dir": str(split_bin_dir),
                "seed": args.seed,
                "holdout_threshold": args.holdout_threshold,
                "holdout_ratio": args.holdout_ratio,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

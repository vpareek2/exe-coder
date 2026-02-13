#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_dataset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root dataset generation runner")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Generate train/val/test splits")
    mode.add_argument("--split", choices=["train", "val", "test"], help="Generate one split")

    parser.add_argument("--n", type=int, help="Examples for single split mode")
    parser.add_argument("--out", type=Path, help="Output JSONL path for single split mode")

    parser.add_argument("--train-n", type=int, default=10000)
    parser.add_argument("--val-n", type=int, default=1000)
    parser.add_argument("--test-n", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-int", type=int, default=-1000)
    parser.add_argument("--max-int", type=int, default=1000)
    parser.add_argument("--holdout-threshold", type=int, default=500)
    parser.add_argument("--holdout-ratio", type=float, default=0.5)
    parser.add_argument("--binaries-dir", type=Path, default=Path("outputs/bin"))
    return parser.parse_args()


def run_one(
    *,
    split: str,
    n: int,
    out: Path,
    seed: int,
    min_int: int,
    max_int: int,
    holdout_threshold: int,
    holdout_ratio: float,
    binaries_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        str(GENERATE_SCRIPT),
        "--split",
        split,
        "--n",
        str(n),
        "--out",
        str(out),
        "--seed",
        str(seed),
        "--min-int",
        str(min_int),
        "--max-int",
        str(max_int),
        "--holdout-threshold",
        str(holdout_threshold),
        "--holdout-ratio",
        str(holdout_ratio),
        "--binaries-dir",
        str(binaries_dir),
    ]

    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    args = parse_args()

    if args.min_int > args.max_int:
        raise ValueError("--min-int must be <= --max-int")
    if not 0.0 <= args.holdout_ratio <= 1.0:
        raise ValueError("--holdout-ratio must be in [0, 1]")

    if args.all:
        if args.n is not None:
            raise ValueError("--n is only valid with --split")
        if args.out is not None:
            raise ValueError("--out is only valid with --split")
        if args.train_n <= 0 or args.val_n <= 0 or args.test_n <= 0:
            raise ValueError("--train-n, --val-n, and --test-n must be positive")

        split_specs = [
            ("train", args.train_n, args.out_dir / "train.jsonl"),
            ("val", args.val_n, args.out_dir / "val.jsonl"),
            ("test", args.test_n, args.out_dir / "test.jsonl"),
        ]
    else:
        if args.n is None:
            raise ValueError("--n is required with --split")
        if args.n <= 0:
            raise ValueError("--n must be positive")
        out = args.out if args.out is not None else args.out_dir / f"{args.split}.jsonl"
        split_specs = [(args.split, args.n, out)]

    outputs: dict[str, str] = {}
    for split, n, out in split_specs:
        run_one(
            split=split,
            n=n,
            out=out,
            seed=args.seed,
            min_int=args.min_int,
            max_int=args.max_int,
            holdout_threshold=args.holdout_threshold,
            holdout_ratio=args.holdout_ratio,
            binaries_dir=args.binaries_dir,
        )
        outputs[split] = str(out)

    print(json.dumps({"generated": outputs, "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

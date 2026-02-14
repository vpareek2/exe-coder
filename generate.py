#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_dataset.py"

PROFILE_DEFAULTS = {
    "stage0": {
        "train_n": 512,
        "val_n": 128,
        "test_n": 128,
        "min_int": -200,
        "max_int": 200,
        "holdout_threshold": 200,
        "holdout_ratio": 0.0,
        "template_set": "train_only",
    },
    "stage1": {
        "train_n": 10_000,
        "val_n": 1_000,
        "test_n": 1_000,
        "min_int": -1_000,
        "max_int": 1_000,
        "holdout_threshold": 500,
        "holdout_ratio": 0.5,
        "template_set": "split_default",
    },
    "stage1a": {
        "train_n": 10_000,
        "val_n": 1_000,
        "test_n": 1_000,
        "min_int": -1_000,
        "max_int": 1_000,
        "holdout_threshold": 500,
        "holdout_ratio": 0.0,
        "template_set": "split_default",
    },
    "stage1b": {
        "train_n": 10_000,
        "val_n": 1_000,
        "test_n": 1_000,
        "min_int": -1_000,
        "max_int": 1_000,
        "holdout_threshold": 500,
        "holdout_ratio": 0.5,
        "template_set": "train_only",
    },
    "stage2": {
        "train_n": 50_000,
        "val_n": 5_000,
        "test_n": 5_000,
        "min_int": -5_000,
        "max_int": 5_000,
        "holdout_threshold": 1_000,
        "holdout_ratio": 0.5,
        "template_set": "split_default",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root dataset generation runner")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Generate train/val/test splits")
    mode.add_argument("--split", choices=["train", "val", "test"], help="Generate one split")

    parser.add_argument("--profile", choices=["stage0", "stage1", "stage1a", "stage1b", "stage2"], default=None)
    parser.add_argument("--n", type=int, help="Examples for single split mode")
    parser.add_argument("--out", type=Path, help="Output JSONL path for single split mode")

    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--val-n", type=int, default=None)
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-int", type=int, default=None)
    parser.add_argument("--max-int", type=int, default=None)
    parser.add_argument("--holdout-threshold", type=int, default=None)
    parser.add_argument("--holdout-ratio", type=float, default=None)
    parser.add_argument("--template-set", choices=["split_default", "train_only"], default=None)
    parser.add_argument("--binaries-dir", type=Path, default=Path("outputs/bin"))
    return parser.parse_args()


def run_one(
    *,
    split: str,
    template_set: str,
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
        "--template-set",
        template_set,
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


def resolve_value(cli_value: int | float | str | None, profile_name: str | None, key: str, fallback: int | float | str) -> int | float | str:
    if cli_value is not None:
        return cli_value
    if profile_name:
        return PROFILE_DEFAULTS[profile_name][key]
    return fallback


def main() -> int:
    args = parse_args()

    min_int = int(resolve_value(args.min_int, args.profile, "min_int", -1000))
    max_int = int(resolve_value(args.max_int, args.profile, "max_int", 1000))
    holdout_threshold = int(resolve_value(args.holdout_threshold, args.profile, "holdout_threshold", 500))
    holdout_ratio = float(resolve_value(args.holdout_ratio, args.profile, "holdout_ratio", 0.5))
    template_set = str(resolve_value(args.template_set, args.profile, "template_set", "split_default"))

    train_n = int(resolve_value(args.train_n, args.profile, "train_n", 10000))
    val_n = int(resolve_value(args.val_n, args.profile, "val_n", 1000))
    test_n = int(resolve_value(args.test_n, args.profile, "test_n", 1000))

    if min_int > max_int:
        raise ValueError("--min-int must be <= --max-int")
    if not 0.0 <= holdout_ratio <= 1.0:
        raise ValueError("--holdout-ratio must be in [0, 1]")

    if args.all:
        if args.n is not None:
            raise ValueError("--n is only valid with --split")
        if args.out is not None:
            raise ValueError("--out is only valid with --split")
        if train_n <= 0 or val_n <= 0 or test_n <= 0:
            raise ValueError("--train-n, --val-n, and --test-n must be positive")

        split_specs = [
            ("train", train_n, args.out_dir / "train.jsonl"),
            ("val", val_n, args.out_dir / "val.jsonl"),
            ("test", test_n, args.out_dir / "test.jsonl"),
        ]
    else:
        n = args.n
        if n is None and args.profile:
            n = int(PROFILE_DEFAULTS[args.profile][f"{args.split}_n"])
        if n is None:
            raise ValueError("--n is required with --split (or use --profile)")
        if n <= 0:
            raise ValueError("--n must be positive")
        out = args.out if args.out is not None else args.out_dir / f"{args.split}.jsonl"
        split_specs = [(args.split, n, out)]

    outputs: dict[str, str] = {}
    for split, n, out in split_specs:
        run_one(
            split=split,
            template_set=template_set,
            n=n,
            out=out,
            seed=args.seed,
            min_int=min_int,
            max_int=max_int,
            holdout_threshold=holdout_threshold,
            holdout_ratio=holdout_ratio,
            binaries_dir=args.binaries_dir,
        )
        outputs[split] = str(out)

    print(
        json.dumps(
            {
                "generated": outputs,
                "seed": args.seed,
                "profile": args.profile,
                "template_set": template_set,
                "min_int": min_int,
                "max_int": max_int,
                "holdout_threshold": holdout_threshold,
                "holdout_ratio": holdout_ratio,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

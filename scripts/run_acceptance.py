#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bootstrap acceptance checks")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-n", type=int, default=16)
    parser.add_argument("--val-n", type=int, default=8)
    parser.add_argument("--test-n", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    uv = ["uv", "run"]

    run([*uv, "scripts/run_phase_a_checks.py"], cwd=root)
    run(
        [
            *uv,
            "generate.py",
            "--all",
            "--train-n",
            str(args.train_n),
            "--val-n",
            str(args.val_n),
            "--test-n",
            str(args.test_n),
            "--out-dir",
            "data/processed",
            "--binaries-dir",
            "outputs/bin",
        ],
        cwd=root,
    )
    run([*uv, "scripts/prepare_sft_data.py", "--input", "data/processed/train.jsonl", "--out", "data/processed/train_sft.jsonl"], cwd=root)
    run([*uv, "train.py", "--config", "configs/sft_scratch.toml"], cwd=root)
    run([*uv, "train.py", "--config", "configs/sft_warmstart.toml"], cwd=root)
    run([*uv, "train.py", "--config", "configs/sft_warmstart_textmix.toml"], cwd=root)
    run([*uv, "train.py", "--config", "configs/rl_mainline.toml"], cwd=root)
    run([*uv, "scripts/infer.py", "--input", "data/processed/test.jsonl", "--out", "outputs/predictions/test.jsonl"], cwd=root)
    run([*uv, "scripts/evaluate.py", "--ckpt", "outputs/checkpoints/rl_mainline", "--split", "test", "--out", "outputs/reports/test.json"], cwd=root)

    print("Acceptance run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    parser.add_argument("--skip-phase-a", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    uv = ["uv", "run"]
    stage = "stage0"
    data_dir = Path("data/processed") / stage
    ckpt_dir = Path("outputs/checkpoints/sft_stage0_overfit")
    pred_path = Path("outputs/predictions") / f"{stage}_val_model_smoke.jsonl"
    report_path = Path("outputs/reports") / f"{stage}_val_model_smoke_eval.json"
    summary_path = Path("outputs/reports") / f"{stage}_val_model_smoke_eval_summary.json"

    if not args.skip_phase_a:
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
            str(data_dir),
            "--binaries-dir",
            f"outputs/bin/{stage}",
            "--template-set",
            "train_only",
            "--min-int",
            "-200",
            "--max-int",
            "200",
            "--holdout-threshold",
            "200",
            "--holdout-ratio",
            "0.0",
        ],
        cwd=root,
    )
    run([*uv, "scripts/check_dataset_quality.py", "--train", str(data_dir / "train.jsonl"), "--val", str(data_dir / "val.jsonl"), "--test", str(data_dir / "test.jsonl")], cwd=root)
    run([*uv, "scripts/prepare_sft_data.py", "--input", str(data_dir / "train.jsonl"), "--out", str(data_dir / "train_sft.jsonl")], cwd=root)
    run([*uv, "train.py", "--config", "configs/sft_stage0_overfit.toml"], cwd=root)
    run(
        [
            *uv,
            "scripts/infer.py",
            "--mode",
            "sft_patch_model",
            "--weights",
            str(ckpt_dir / "best_weights.safetensors"),
            "--input",
            str(data_dir / "val.jsonl"),
            "--out",
            str(pred_path),
            "--binaries-dir",
            f"outputs/predictions/bin/{stage}_val_smoke",
        ],
        cwd=root,
    )
    run(
        [
            *uv,
            "scripts/evaluate.py",
            "--ckpt",
            str(ckpt_dir),
            "--split",
            "val",
            "--predictions",
            str(pred_path),
            "--out",
            str(report_path),
            "--summary-out",
            str(summary_path),
        ],
        cwd=root,
    )

    print("Acceptance run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

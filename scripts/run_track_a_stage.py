#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


STAGE_CONFIGS = {
    "stage0": "configs/sft_stage0_overfit.toml",
    "stage1": "configs/sft_stage1_pilot.toml",
    "stage1a": "configs/sft_stage1a_template_shift.toml",
    "stage1b": "configs/sft_stage1b_magnitude_shift.toml",
    "stage2": "configs/sft_stage2_scale.toml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Track-A staged workflow: generate -> train -> infer -> evaluate")
    parser.add_argument("--stage", choices=["stage0", "stage1", "stage1a", "stage1b", "stage2"], required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def read_output_dir(config_path: Path) -> Path:
    with config_path.open("rb") as infile:
        payload = tomllib.load(infile)
    run_cfg = payload.get("run", {})
    if not isinstance(run_cfg, dict) or "output_dir" not in run_cfg:
        raise ValueError(f"Missing [run].output_dir in {config_path}")
    output_dir = Path(str(run_cfg["output_dir"]))
    return output_dir


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    config_rel = STAGE_CONFIGS[args.stage]
    config_path = root / config_rel
    checkpoint_dir = read_output_dir(config_path)

    data_dir = Path("data/processed") / args.stage
    binaries_dir = Path("outputs/bin") / args.stage
    pred_out = Path("outputs/predictions") / f"{args.stage}_val_model.jsonl"
    report_out = Path("outputs/reports") / f"{args.stage}_val_model_eval.json"
    summary_out = Path("outputs/reports") / f"{args.stage}_val_model_eval_summary.json"
    pred_bin_dir = Path("outputs/predictions/bin") / f"{args.stage}_val"

    uv = ["uv", "run"]

    if not args.skip_generate:
        run(
            [
                *uv,
                "generate.py",
                "--all",
                "--profile",
                args.stage,
                "--seed",
                str(args.seed),
                "--out-dir",
                str(data_dir),
                "--binaries-dir",
                str(binaries_dir),
            ],
            cwd=root,
        )
        run(
            [
                *uv,
                "scripts/prepare_sft_data.py",
                "--input",
                str(data_dir / "train.jsonl"),
                "--out",
                str(data_dir / "train_sft.jsonl"),
            ],
            cwd=root,
        )
        quality_cmd = [
            *uv,
            "scripts/check_dataset_quality.py",
            "--train",
            str(data_dir / "train.jsonl"),
            "--val",
            str(data_dir / "val.jsonl"),
            "--test",
            str(data_dir / "test.jsonl"),
        ]
        if args.stage in {"stage1", "stage1b", "stage2"}:
            quality_cmd.append("--require-heldout-val-test")
        run(quality_cmd, cwd=root)

    if not args.skip_train:
        run([*uv, "train.py", "--config", config_rel], cwd=root)

    if not args.skip_eval:
        weights = checkpoint_dir / "best_weights.safetensors"
        run(
            [
                *uv,
                "scripts/infer.py",
                "--mode",
                "sft_patch_model",
                "--weights",
                str(weights),
                "--input",
                str(data_dir / "val.jsonl"),
                "--out",
                str(pred_out),
                "--binaries-dir",
                str(pred_bin_dir),
            ],
            cwd=root,
        )
        run(
            [
                *uv,
                "scripts/evaluate.py",
                "--ckpt",
                str(checkpoint_dir),
                "--split",
                "val",
                "--predictions",
                str(pred_out),
                "--out",
                str(report_out),
                "--summary-out",
                str(summary_out),
            ],
            cwd=root,
        )

        summary_payload = json.loads((root / summary_out).read_text(encoding="utf-8"))
        gates = summary_payload.get("milestone_gates", {})
        gate_status = {
            "g0_loop_sanity": bool(gates.get("g0_loop_sanity", {}).get("pass")),
            "g1_pilot_success": bool(gates.get("g1_pilot_success", {}).get("pass")),
            "g2_rl_ready": bool(gates.get("g2_rl_ready", {}).get("pass")),
        }
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "config": config_rel,
                    "checkpoint_dir": str(checkpoint_dir),
                    "summary_out": str(summary_out),
                    "gate_status": gate_status,
                },
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

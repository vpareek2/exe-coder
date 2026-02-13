#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent


class ConfigError(RuntimeError):
    """Raised when a training config is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python-MLX training config runner")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config file")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as infile:
        return tomllib.load(infile)


def require_section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    section = cfg.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing or invalid section [{name}]")
    return section


def require_value(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required key '{key}'")
    return section[key]


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                count += 1
    return count


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def detect_mlx() -> dict[str, Any]:
    details: dict[str, Any] = {
        "available": False,
        "module": None,
        "version": None,
        "error": None,
    }

    cmd = [
        sys.executable,
        "-c",
        (
            "import mlx, mlx.core, mlx.nn, mlx.optimizers; "
            "print(getattr(mlx, '__version__', 'unknown'))"
        ),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode == 0:
        details["available"] = True
        details["module"] = "mlx"
        details["version"] = proc.stdout.strip() or "unknown"
    else:  # pragma: no cover
        stderr = (proc.stderr or "").strip()
        details["error"] = stderr or f"mlx import subprocess exited with code {proc.returncode}"

    return details


def validate_phase_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = require_section(cfg, "run")
    data = require_section(cfg, "data")
    model = require_section(cfg, "model")
    optimizer = require_section(cfg, "optimizer")
    init = require_section(cfg, "init")
    rl = require_section(cfg, "rl")

    phase = require_value(run, "phase")
    if phase not in {"sft", "rl"}:
        raise ConfigError("[run].phase must be 'sft' or 'rl'")

    _ = require_value(run, "name")
    _ = require_value(run, "seed")
    _ = require_value(run, "output_dir")

    _ = require_value(data, "train_jsonl")
    _ = require_value(data, "val_jsonl")
    _ = require_value(data, "tokenization_mode")
    _ = require_value(data, "track")

    _ = require_value(model, "layers")
    _ = require_value(model, "hidden_size")
    _ = require_value(model, "heads")
    _ = require_value(model, "seq_len")
    _ = require_value(model, "dropout")

    _ = require_value(optimizer, "scheme")
    _ = require_value(optimizer, "lr_adamw")
    _ = require_value(optimizer, "lr_muon")
    _ = require_value(optimizer, "weight_decay")

    _ = require_value(init, "mode")
    _ = require_value(init, "warmstart_ckpt")
    _ = require_value(init, "textmix_ratio")

    _ = require_value(rl, "beta")
    _ = require_value(rl, "rollout_samples")
    _ = require_value(rl, "mode")

    return run, data, model, optimizer, init, rl


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_sft(
    config_path: Path,
    run: dict[str, Any],
    data: dict[str, Any],
    model: dict[str, Any],
    optimizer: dict[str, Any],
    init: dict[str, Any],
    rl: dict[str, Any],
) -> int:
    output_dir = resolve_path(str(run["output_dir"]))
    train_jsonl = resolve_path(str(data["train_jsonl"]))
    val_jsonl = resolve_path(str(data["val_jsonl"]))

    mlx = detect_mlx()

    train_rows = count_jsonl_rows(train_jsonl)
    val_rows = count_jsonl_rows(val_jsonl)

    checkpoint = {
        "phase": "sft",
        "created_at": iso_now(),
        "config": str(config_path),
        "run_name": run["name"],
        "status": "python_mlx_scaffold_checkpoint",
        "track": data["track"],
        "init_mode": init["mode"],
        "mlx": mlx,
    }

    metrics = {
        "phase": "sft",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "tokenization_mode": data["tokenization_mode"],
        "track": data["track"],
        "model": model,
        "optimizer": optimizer,
        "init": init,
        "rl": rl,
        "notes": "Scaffold run only; replace with full MLX transformer SFT loop.",
    }

    write_json(output_dir / "checkpoint.json", checkpoint)
    write_json(output_dir / "metrics.json", metrics)

    print(f"[train.py:sft] config={config_path}")
    print(f"[train.py:sft] train_rows={train_rows} val_rows={val_rows}")
    print(f"[train.py:sft] checkpoint={output_dir / 'checkpoint.json'}")
    print(f"[train.py:sft] metrics={output_dir / 'metrics.json'}")
    return 0


def run_rl(
    config_path: Path,
    run: dict[str, Any],
    data: dict[str, Any],
    model: dict[str, Any],
    optimizer: dict[str, Any],
    init: dict[str, Any],
    rl: dict[str, Any],
) -> int:
    output_dir = resolve_path(str(run["output_dir"]))
    train_jsonl = resolve_path(str(data["train_jsonl"]))
    val_jsonl = resolve_path(str(data["val_jsonl"]))

    rl_mode = str(rl["mode"])
    init_ckpt_raw = str(rl.get("init_ckpt", "")).strip()

    if rl_mode == "mainline":
        if not init_ckpt_raw:
            raise ConfigError("[rl].init_ckpt is required for rl mode 'mainline'")
        init_ckpt = resolve_path(init_ckpt_raw)
        if not init_ckpt.exists():
            raise FileNotFoundError(f"Mainline RL init checkpoint not found: {init_ckpt}")
    elif rl_mode == "scratch_exploratory":
        init_ckpt = resolve_path(init_ckpt_raw) if init_ckpt_raw else None
    else:
        raise ConfigError("[rl].mode must be 'mainline' or 'scratch_exploratory'")

    mlx = detect_mlx()

    train_rows = count_jsonl_rows(train_jsonl)
    val_rows = count_jsonl_rows(val_jsonl)

    checkpoint = {
        "phase": "rl",
        "created_at": iso_now(),
        "config": str(config_path),
        "run_name": run["name"],
        "status": "python_mlx_scaffold_checkpoint",
        "track": data["track"],
        "rl_mode": rl_mode,
        "init_ckpt": str(init_ckpt) if init_ckpt else None,
        "mlx": mlx,
    }

    metrics = {
        "phase": "rl",
        "train_rows": train_rows,
        "val_rows": val_rows,
        "tokenization_mode": data["tokenization_mode"],
        "track": data["track"],
        "model": model,
        "optimizer": optimizer,
        "init": init,
        "rl": rl,
        "reward_formula": "1*valid + 1*exec + 2*correct - 1*timeout_or_crash - beta*kl",
        "notes": "Scaffold run only; replace with full MLX policy optimization loop.",
    }

    write_json(output_dir / "checkpoint.json", checkpoint)
    write_json(output_dir / "metrics.json", metrics)

    print(f"[train.py:rl] config={config_path}")
    print(f"[train.py:rl] mode={rl_mode} train_rows={train_rows} val_rows={val_rows}")
    print(f"[train.py:rl] checkpoint={output_dir / 'checkpoint.json'}")
    print(f"[train.py:rl] metrics={output_dir / 'metrics.json'}")
    return 0


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()

    cfg = load_toml(config_path)
    run, data, model, optimizer, init, rl = validate_phase_config(cfg)

    phase = str(run["phase"])
    if phase == "sft":
        return run_sft(config_path, run, data, model, optimizer, init, rl)
    if phase == "rl":
        return run_rl(config_path, run, data, model, optimizer, init, rl)

    raise ConfigError(f"Unsupported phase: {phase}")


if __name__ == "__main__":
    raise SystemExit(main())

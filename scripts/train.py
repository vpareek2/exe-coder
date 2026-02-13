#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for root training runner")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(repo_root / "train.py"), "--config", str(args.config)]
    proc = subprocess.run(cmd, check=False, cwd=repo_root)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import DEFAULT_BETA, compute_reward, decode_escaped_text, verify_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify binary format + execution correctness")
    parser.add_argument("--bin", type=Path, required=True, help="Binary to execute")
    parser.add_argument(
        "--expected-stdout",
        required=True,
        help="Expected stdout string (supports escaped sequences like \\n)",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Execution timeout seconds")
    parser.add_argument("--kl-to-sft", type=float, default=0.0, help="KL term for reward calculation")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA, help="KL multiplier")
    parser.add_argument("--json-out", type=Path, help="Optional output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_stdout = decode_escaped_text(args.expected_stdout)

    verification = verify_binary(args.bin, expected_stdout, timeout_seconds=args.timeout)
    reward = compute_reward(verification, kl_to_sft=args.kl_to_sft, beta=args.beta)

    payload = {"verification": verification, **reward}

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0 if verification["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

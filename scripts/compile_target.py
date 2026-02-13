#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import compile_addition_binary, encode_file_base64, verify_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile deterministic Mach-O addition binary")
    parser.add_argument("--a", type=int, required=True, help="First integer operand")
    parser.add_argument("--b", type=int, required=True, help="Second integer operand")
    parser.add_argument("--out", type=Path, required=True, help="Output binary path")
    parser.add_argument("--metadata-out", type=Path, help="Optional JSON metadata path")
    parser.add_argument(
        "--include-binary-b64",
        action="store_true",
        help="Include full base64 binary blob in output payload",
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-compile verification")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_stdout = f"{args.a + args.b}\n"

    result = compile_addition_binary(args.a, args.b, args.out)
    payload: dict[str, object] = {
        "a": args.a,
        "b": args.b,
        "target_format": "macho64",
        "target_arch": "arm64",
        "target_os": "macos",
        "binary_path": result.binary_path,
        "binary_sha256": result.binary_sha256,
        "byte_length": result.byte_length,
        "expected_stdout": expected_stdout,
        "toolchain": result.toolchain,
        "compile_flags": result.compile_flags,
        "compile_command": result.compile_command,
    }
    if args.include_binary_b64:
        payload["binary_b64"] = encode_file_base64(result.binary_path)

    if not args.skip_verify:
        verification = verify_binary(result.binary_path, expected_stdout)
        payload["verification"] = verification
        if not verification["pass"]:
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return 1

    if args.metadata_out:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

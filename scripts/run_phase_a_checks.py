#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binary.pipeline import compile_addition_binary, verify_binary

GOLDEN_CASES: list[tuple[int, int]] = [
    (0, 0),
    (1, 2),
    (-1, 1),
    (-7, 5),
    (-12, -9),
    (999, 1),
    (17, 25),
    (42, -42),
    (123, 456),
    (-100, -200),
    (500, 500),
    (-500, 500),
    (32767, 1),
    (-32768, -1),
    (111, -222),
    (-333, 444),
    (1024, 2048),
    (-1024, 2048),
    (314, 159),
    (-271, 828),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase A deterministic compile + golden correctness checks")
    parser.add_argument("--a", type=int, default=17, help="First operand for determinism test")
    parser.add_argument("--b", type=int, default=25, help="Second operand for determinism test")
    parser.add_argument("--determinism-runs", type=int, default=20, help="Number of compile runs")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/phase_a"),
        help="Directory for intermediate binaries",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("outputs/reports/phase_a_step1_report.json"),
        help="JSON report output path",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="Execution timeout seconds")
    return parser.parse_args()


def run_determinism_check(a: int, b: int, runs: int, work_dir: Path) -> dict[str, object]:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "determinism_target"
    expected_stdout = f"{a + b}\n"

    hashes: list[str] = []
    for _ in range(runs):
        result = compile_addition_binary(a, b, out_path)
        hashes.append(result.binary_sha256)

    unique_hashes = sorted(set(hashes))
    verification = verify_binary(out_path, expected_stdout)

    return {
        "a": a,
        "b": b,
        "runs": runs,
        "hashes": hashes,
        "unique_hashes": unique_hashes,
        "determinism_pass": len(unique_hashes) == 1,
        "verification": verification,
    }


def run_golden_checks(work_dir: Path, timeout: float) -> dict[str, object]:
    golden_dir = work_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for index, (a, b) in enumerate(GOLDEN_CASES):
        binary_path = golden_dir / f"golden_{index:02d}"
        compile_result = compile_addition_binary(a, b, binary_path)
        verification = verify_binary(binary_path, f"{a + b}\n", timeout_seconds=timeout)
        entries.append(
            {
                "index": index,
                "a": a,
                "b": b,
                "compile": {
                    "binary_path": compile_result.binary_path,
                    "binary_sha256": compile_result.binary_sha256,
                    "byte_length": compile_result.byte_length,
                    "toolchain": compile_result.toolchain,
                    "compile_flags": compile_result.compile_flags,
                },
                "verification": verification,
            }
        )

    pass_count = sum(1 for row in entries if row["verification"]["pass"])
    return {
        "total": len(entries),
        "pass_count": pass_count,
        "all_pass": pass_count == len(entries),
        "cases": entries,
    }


def main() -> int:
    args = parse_args()

    determinism = run_determinism_check(args.a, args.b, args.determinism_runs, args.work_dir)
    golden = run_golden_checks(args.work_dir, timeout=args.timeout)

    step_1_pass = bool(
        determinism["determinism_pass"]
        and determinism["verification"]["pass"]
        and golden["all_pass"]
    )

    report = {
        "step": "phase_a_step_1",
        "runtime_contract": "stdout-only; executable must print '<sum>\\n'",
        "determinism": determinism,
        "golden_suite": golden,
        "step_1_pass": step_1_pass,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "step_1_pass": step_1_pass,
        "determinism_pass": determinism["determinism_pass"],
        "golden_all_pass": golden["all_pass"],
        "report": str(args.report_out),
    }, indent=2))

    return 0 if step_1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

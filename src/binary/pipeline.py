from __future__ import annotations

import base64
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TARGET = "arm64-apple-macos11"
DEFAULT_COMPILE_FLAGS = (
    "-O2",
    "-std=c11",
    "-target",
    DEFAULT_TARGET,
)
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_BETA = 0.01
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class CompileResult:
    binary_path: str
    binary_sha256: str
    byte_length: int
    toolchain: str
    compile_flags: str
    compile_command: str


def decode_escaped_text(text: str) -> str:
    """Decodes simple escape sequences like '\\n' for CLI-friendly usage."""
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return text


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_file_base64(path: Path | str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _run_checked(command: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {joined}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc


def _toolchain_descriptor(clang_bin: str) -> str:
    first = _run_checked([clang_bin, "--version"]).stdout.strip().splitlines()[0]
    return f"{first} | clang_bin={clang_bin}"


def _render_addition_source(a: int, b: int) -> str:
    return (
        "#include <stdio.h>\n\n"
        "int main(void) {\n"
        f"  long long a = {a}LL;\n"
        f"  long long b = {b}LL;\n"
        "  printf(\"%lld\\n\", a + b);\n"
        "  return 0;\n"
        "}\n"
    )


def _validate_int64_operand(name: str, value: int) -> None:
    if not (INT64_MIN <= int(value) <= INT64_MAX):
        raise ValueError(
            f"{name}={value} is outside signed 64-bit range "
            f"[{INT64_MIN}, {INT64_MAX}]"
        )


def compile_addition_binary(
    a: int,
    b: int,
    out_path: Path | str,
    *,
    clang_bin: str = "clang",
    compile_flags: Sequence[str] = DEFAULT_COMPILE_FLAGS,
) -> CompileResult:
    """Compiles a deterministic Mach-O binary that prints a+b and newline.

    Determinism strategy:
    - compile in a temp directory with fixed input/output filenames
    - use stable linker flags to reduce metadata variance
    - copy resulting bytes to requested output path
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _validate_int64_operand("a", a)
    _validate_int64_operand("b", b)

    with tempfile.TemporaryDirectory(prefix="exe_coder_compile_") as tmp_dir:
        temp_root = Path(tmp_dir)
        source_file = temp_root / "main.c"
        built_file = temp_root / "program"

        source_file.write_text(_render_addition_source(a, b), encoding="utf-8")

        command = [
            clang_bin,
            *compile_flags,
            str(source_file),
            "-o",
            str(built_file),
        ]
        _run_checked(command)

        shutil.copy2(built_file, out_file)
        out_file.chmod(out_file.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return CompileResult(
        binary_path=str(out_file),
        binary_sha256=sha256_file(out_file),
        byte_length=out_file.stat().st_size,
        toolchain=_toolchain_descriptor(clang_bin),
        compile_flags=" ".join(compile_flags),
        compile_command=" ".join(command),
    )


def _run_optional(command: Sequence[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if not executable:
        return False, f"{command[0]} not found"
    proc = subprocess.run(command, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, proc.stdout.strip()


def _macho_static_check(binary_path: Path | str) -> dict[str, Any]:
    path = str(binary_path)

    file_ok, file_out = _run_optional(["file", "-b", path])
    otool_ok, otool_out = _run_optional(["otool", "-hv", path])

    looks_macho = file_ok and ("Mach-O" in file_out)
    looks_arm64 = ("arm64" in file_out.lower()) or ("ARM64" in otool_out)

    return {
        "file_ok": file_ok,
        "file_output": file_out,
        "otool_ok": otool_ok,
        "otool_output": otool_out,
        "is_macho_arm64": bool(looks_macho and looks_arm64),
    }


def execute_binary(binary_path: Path | str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [str(binary_path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return {
            "executed": True,
            "timed_out": False,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "executed": False,
            "timed_out": True,
            "returncode": None,
            "stdout": exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }


def verify_binary(
    binary_path: Path | str,
    expected_stdout: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    static = _macho_static_check(binary_path)
    execution = execute_binary(binary_path, timeout_seconds=timeout_seconds)

    exec_ok = execution["executed"] and not execution["timed_out"] and execution["returncode"] == 0
    stdout_correct = execution["stdout"] == expected_stdout

    return {
        "binary_path": str(binary_path),
        "binary_sha256": sha256_file(binary_path),
        "byte_length": Path(binary_path).stat().st_size,
        "expected_stdout": expected_stdout,
        "static": static,
        "execution": execution,
        "metrics": {
            "macho_valid": int(static["is_macho_arm64"]),
            "exec_ok": int(exec_ok),
            "stdout_correct": int(stdout_correct),
            "timeout_or_crash": int(execution["timed_out"] or (execution["returncode"] not in (0, None))),
        },
        "pass": bool(static["is_macho_arm64"] and exec_ok and stdout_correct),
    }


def compute_reward(
    verify_result: dict[str, Any],
    *,
    kl_to_sft: float = 0.0,
    beta: float = DEFAULT_BETA,
) -> dict[str, Any]:
    metrics = verify_result["metrics"]
    reward = (
        1.0 * metrics["macho_valid"]
        + 1.0 * metrics["exec_ok"]
        + 2.0 * metrics["stdout_correct"]
        - 1.0 * metrics["timeout_or_crash"]
        - beta * kl_to_sft
    )

    return {
        "reward": reward,
        "reward_components": {
            "is_valid_macho": float(metrics["macho_valid"]),
            "exec_ok": float(metrics["exec_ok"]),
            "stdout_correct": float(metrics["stdout_correct"]),
            "timeout_or_crash": float(metrics["timeout_or_crash"]),
            "kl_to_sft": float(kl_to_sft),
            "beta": float(beta),
        },
    }


def ensure_parent(path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def as_executable(path: Path | str) -> None:
    current_mode = Path(path).stat().st_mode
    os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

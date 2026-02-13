from __future__ import annotations

import base64
from pathlib import Path


def bytes_to_hex_tokens(blob: bytes) -> list[str]:
    return [f"{byte:02x}" for byte in blob]


def hex_tokens_to_bytes(tokens: list[str]) -> bytes:
    return bytes(int(token, 16) for token in tokens)


def base64_to_hex_tokens(binary_b64: str) -> list[str]:
    return bytes_to_hex_tokens(base64.b64decode(binary_b64))


def file_to_hex_tokens(path: Path | str) -> list[str]:
    return bytes_to_hex_tokens(Path(path).read_bytes())


def tokens_to_target_string(tokens: list[str]) -> str:
    return " ".join(tokens)


def format_training_sequence(prompt: str, target_hex_tokens: list[str]) -> str:
    return (
        f"PROMPT: {prompt}\n"
        f"TARGET_BIN_HEX: {tokens_to_target_string(target_hex_tokens)}"
    )

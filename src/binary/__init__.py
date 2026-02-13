"""Binary compile/verify helpers for exe-coder."""

from .pipeline import (
    DEFAULT_BETA,
    DEFAULT_COMPILE_FLAGS,
    DEFAULT_TARGET,
    compile_addition_binary,
    compute_reward,
    decode_escaped_text,
    encode_file_base64,
    execute_binary,
    sha256_file,
    verify_binary,
)
from .tokenization import (
    base64_to_hex_tokens,
    bytes_to_hex_tokens,
    file_to_hex_tokens,
    format_training_sequence,
    hex_tokens_to_bytes,
    tokens_to_target_string,
)

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_COMPILE_FLAGS",
    "DEFAULT_TARGET",
    "compile_addition_binary",
    "compute_reward",
    "decode_escaped_text",
    "encode_file_base64",
    "execute_binary",
    "base64_to_hex_tokens",
    "bytes_to_hex_tokens",
    "file_to_hex_tokens",
    "format_training_sequence",
    "hex_tokens_to_bytes",
    "tokens_to_target_string",
    "sha256_file",
    "verify_binary",
]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval_framework import is_better_checkpoint, parse_patch_text

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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return rows


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


def validate_phase_config(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = require_section(cfg, "run")
    data = require_section(cfg, "data")
    model = require_section(cfg, "model")
    optimizer = require_section(cfg, "optimizer")
    init = require_section(cfg, "init")
    rl = require_section(cfg, "rl")
    train = cfg.get("train", {})
    if not isinstance(train, dict):
        raise ConfigError("Section [train] must be a table if provided")

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
    if "lr_adamw" not in optimizer and "learning_rate" not in optimizer:
        raise ConfigError("[optimizer] requires lr_adamw (or learning_rate)")
    _ = require_value(optimizer, "weight_decay")

    _ = require_value(init, "mode")
    _ = require_value(init, "warmstart_ckpt")
    _ = require_value(init, "textmix_ratio")

    _ = require_value(rl, "beta")
    _ = require_value(rl, "rollout_samples")
    _ = require_value(rl, "mode")

    return run, data, model, optimizer, init, rl, train


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as outfile:
        outfile.write(json.dumps(payload, ensure_ascii=True) + "\n")


class ByteTokenizer:
    def __init__(self) -> None:
        self.bos_id = 256
        self.eos_id = 257
        self.pad_id = 258
        self.vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        byte_values = [t for t in tokens if 0 <= t <= 255]
        return bytes(byte_values).decode("utf-8", errors="replace")


@dataclass
class ExamplePack:
    input_ids: list[int]
    labels: list[int]
    loss_mask: list[float]


@dataclass
class SFTSettings:
    target_mode: str
    batch_size: int
    epochs: int
    max_steps: int
    log_interval: int
    eval_interval: int
    save_interval: int
    eval_max_samples: int
    max_gen_tokens: int
    grad_clip_norm: float
    warmup_steps: int
    min_lr_ratio: float
    label_smoothing: float
    full_eval_every: int


def parse_sft_settings(data: dict[str, Any], train: dict[str, Any]) -> SFTSettings:
    target_mode = str(data.get("target_mode", "patch_text"))
    return SFTSettings(
        target_mode=target_mode,
        batch_size=int(train.get("batch_size", 16)),
        epochs=int(train.get("epochs", 20)),
        max_steps=int(train.get("max_steps", 200)),
        log_interval=max(1, int(train.get("log_interval", 2))),
        eval_interval=max(1, int(train.get("eval_interval", 20))),
        save_interval=max(1, int(train.get("save_interval", 20))),
        eval_max_samples=max(1, int(train.get("eval_max_samples", 64))),
        max_gen_tokens=max(8, int(train.get("max_gen_tokens", 32))),
        grad_clip_norm=float(train.get("grad_clip_norm", 1.0)),
        warmup_steps=max(0, int(train.get("warmup_steps", 10))),
        min_lr_ratio=float(train.get("min_lr_ratio", 0.1)),
        label_smoothing=float(train.get("label_smoothing", 0.0)),
        full_eval_every=max(1, int(train.get("full_eval_every", 5))),
    )


def deterministic_subset(rows: list[dict[str, Any]], max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples >= len(rows):
        return list(rows)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    keep = set(idx[:max_samples])
    return [row for i, row in enumerate(rows) if i in keep]


def ensure_patch_target_mode(track: str, settings: SFTSettings) -> None:
    if track == "template_delta_operand_patch" and settings.target_mode != "patch_text":
        raise ConfigError("template_delta_operand_patch requires data.target_mode='patch_text'")
    if settings.target_mode != "patch_text":
        raise ConfigError(
            "Only data.target_mode='patch_text' is implemented in the real SFT loop. "
            "Use Track A first."
        )


def format_prompt_prefix(prompt: str) -> str:
    return f"PROMPT: {prompt}\nTARGET_PATCH:"


def format_patch_target(a: int, b: int) -> str:
    return f"A={a};B={b}\n"


def extract_prompt(record: dict[str, Any]) -> str:
    prompt = record.get("prompt")
    if prompt is None:
        raise ValueError("Record missing 'prompt'")
    return str(prompt)


def extract_operands(record: dict[str, Any]) -> tuple[int, int]:
    if "a" not in record or "b" not in record:
        raise ValueError("Record missing operands 'a'/'b'")
    return int(record["a"]), int(record["b"])


def build_example_pack(record: dict[str, Any], tokenizer: ByteTokenizer, max_seq_len: int) -> ExamplePack:
    prompt = extract_prompt(record)
    a, b = extract_operands(record)

    prompt_ids = tokenizer.encode(format_prompt_prefix(prompt))
    target_ids = tokenizer.encode(format_patch_target(a, b))

    # Include BOS/EOS so generation has explicit boundaries.
    tokens = [tokenizer.bos_id] + prompt_ids + target_ids + [tokenizer.eos_id]

    # We need len(tokens) <= max_seq_len + 1 because input/label are shifted.
    max_full_len = max_seq_len + 1
    if len(tokens) > max_full_len:
        overflow = len(tokens) - max_full_len
        if overflow >= len(prompt_ids):
            # Keep suffix of prompt if we are forced to trim heavily.
            prompt_ids = prompt_ids[-1:]
        else:
            prompt_ids = prompt_ids[overflow:]
        tokens = [tokenizer.bos_id] + prompt_ids + target_ids + [tokenizer.eos_id]

    target_start = 1 + len(prompt_ids)
    token_loss_mask = [0.0] * target_start + [1.0] * (len(tokens) - target_start)

    input_ids = tokens[:-1]
    labels = tokens[1:]
    loss_mask = token_loss_mask[1:]

    return ExamplePack(input_ids=input_ids, labels=labels, loss_mask=loss_mask)


def collate_batch(
    records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    max_seq_len: int,
    *,
    mx: Any,
) -> tuple[Any, Any, Any, int]:
    packed = [build_example_pack(record, tokenizer, max_seq_len) for record in records]
    max_len = max(len(item.input_ids) for item in packed)

    batch_inputs: list[list[int]] = []
    batch_labels: list[list[int]] = []
    batch_mask: list[list[float]] = []
    target_tokens = 0

    for item in packed:
        pad = max_len - len(item.input_ids)
        batch_inputs.append(item.input_ids + [tokenizer.pad_id] * pad)
        batch_labels.append(item.labels + [0] * pad)
        batch_mask.append(item.loss_mask + [0.0] * pad)
        target_tokens += int(sum(item.loss_mask))

    return (
        mx.array(batch_inputs, dtype=mx.int32),
        mx.array(batch_labels, dtype=mx.int32),
        mx.array(batch_mask, dtype=mx.float32),
        target_tokens,
    )


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= 1:
        return base_lr

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    decay_steps = max(1, total_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    progress = max(0.0, min(1.0, progress))

    min_lr = base_lr * min_lr_ratio
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def resolve_warmstart_weights(path_like: str) -> Path | None:
    if not path_like:
        return None

    raw_path = resolve_path(path_like)
    if not raw_path.exists():
        return None

    if raw_path.is_dir():
        for name in ("best_weights.safetensors", "latest_weights.safetensors", "model.safetensors", "weights.safetensors"):
            candidate = raw_path / name
            if candidate.exists():
                return candidate
        return None

    if raw_path.suffix in {".safetensors", ".npz"}:
        return raw_path

    if raw_path.suffix == ".json":
        try:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        for key in ("best_weights", "latest_weights", "weights_path"):
            val = payload.get(key)
            if isinstance(val, str) and val:
                candidate = resolve_path(val)
                if candidate.exists() and candidate.suffix in {".safetensors", ".npz"}:
                    return candidate
    return None


def make_model(vocab_size: int, model_cfg: dict[str, Any], mx: Any, nn: Any) -> Any:
    layers = int(model_cfg["layers"])
    hidden_size = int(model_cfg["hidden_size"])
    heads = int(model_cfg["heads"])
    seq_len = int(model_cfg["seq_len"])
    dropout = float(model_cfg["dropout"])

    if hidden_size % heads != 0:
        raise ConfigError("[model].hidden_size must be divisible by [model].heads")

    class DecoderBlock(nn.Module):
        def __init__(self, dims: int, num_heads: int, p_drop: float):
            super().__init__()
            self.ln1 = nn.LayerNorm(dims)
            self.ln2 = nn.LayerNorm(dims)
            self.attn = nn.MultiHeadAttention(dims, num_heads, bias=True)
            self.fc1 = nn.Linear(dims, dims * 4, bias=True)
            self.fc2 = nn.Linear(dims * 4, dims, bias=True)
            self.drop1 = nn.Dropout(p_drop)
            self.drop2 = nn.Dropout(p_drop)

        def __call__(self, x: Any, mask: Any) -> Any:
            h = self.ln1(x)
            h = self.attn(h, h, h, mask)
            h = self.drop1(h)
            x = x + h

            h = self.ln2(x)
            h = self.fc1(h)
            h = nn.gelu(h)
            h = self.fc2(h)
            h = self.drop2(h)
            return x + h

    class CausalPatchTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.max_seq_len = seq_len
            self.tok_emb = nn.Embedding(vocab_size, hidden_size)
            self.pos_emb = nn.Embedding(seq_len, hidden_size)
            self.layers = [DecoderBlock(hidden_size, heads, dropout) for _ in range(layers)]
            self.ln_f = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        def __call__(self, input_ids: Any) -> Any:
            batch, time = input_ids.shape
            del batch
            if time > self.max_seq_len:
                raise ValueError(f"Input length {time} exceeds model.seq_len={self.max_seq_len}")

            pos = mx.arange(time, dtype=mx.int32)
            x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(time, x.dtype)
            for layer in self.layers:
                x = layer(x, mask)
            x = self.ln_f(x)
            return self.lm_head(x)

    return CausalPatchTransformer()


def greedy_generate_patch(
    model: Any,
    prompt: str,
    tokenizer: ByteTokenizer,
    *,
    max_seq_len: int,
    max_gen_tokens: int,
    mx: Any,
) -> str:
    prefix = format_prompt_prefix(prompt)
    prefix_ids = tokenizer.encode(prefix)
    context = [tokenizer.bos_id] + prefix_ids

    generated: list[int] = []
    for _ in range(max_gen_tokens):
        if len(context) >= max_seq_len:
            break

        logits = model(mx.array([context], dtype=mx.int32))
        next_id = int(mx.argmax(logits[0, -1]).item())

        if next_id == tokenizer.eos_id:
            break
        if next_id == tokenizer.pad_id:
            break

        generated.append(next_id)
        context.append(next_id)

    return tokenizer.decode(generated)


def evaluate_model(
    model: Any,
    eval_rows: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    settings: SFTSettings,
    model_cfg: dict[str, Any],
    *,
    label_smoothing: float,
    beta: float,
    mx: Any,
    nn: Any,
) -> dict[str, Any]:
    from src.binary.pipeline import compile_addition_binary, compute_reward, verify_binary

    if not eval_rows:
        raise ValueError("Evaluation rows are empty")

    model.eval()

    # Teacher-forced validation loss.
    val_losses: list[float] = []
    val_tokens = 0
    for start in range(0, len(eval_rows), settings.batch_size):
        batch_rows = eval_rows[start : start + settings.batch_size]
        input_ids, labels, loss_mask, target_tokens = collate_batch(
            batch_rows,
            tokenizer,
            int(model_cfg["seq_len"]),
            mx=mx,
        )
        logits = model(input_ids)
        loss_per_token = nn.losses.cross_entropy(
            logits,
            labels,
            label_smoothing=label_smoothing,
            reduction="none",
        )
        denom = mx.maximum(loss_mask.sum(), mx.array(1.0, dtype=loss_per_token.dtype))
        loss = (loss_per_token * loss_mask).sum() / denom
        val_losses.append(float(loss.item()))
        val_tokens += target_tokens

    sum_valid = 0
    sum_exec = 0
    sum_correct = 0
    sum_exact_patch = 0
    sum_parse_ok = 0
    operand_mae_sum = 0.0
    operand_mae_count = 0
    rewards: list[float] = []

    with tempfile.TemporaryDirectory(prefix="exe_coder_val_bins_") as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(eval_rows):
            prompt = extract_prompt(row)
            expected_a, expected_b = extract_operands(row)
            expected_stdout = row.get("expected_stdout", f"{expected_a + expected_b}\n")

            pred_text = greedy_generate_patch(
                model,
                prompt,
                tokenizer,
                max_seq_len=int(model_cfg["seq_len"]),
                max_gen_tokens=settings.max_gen_tokens,
                mx=mx,
            )
            parsed = parse_patch_text(pred_text)

            if parsed is None:
                rewards.append(-1.0)
                continue

            sum_parse_ok += 1
            pred_a, pred_b = parsed
            sum_exact_patch += int(pred_a == expected_a and pred_b == expected_b)
            operand_mae_sum += (abs(pred_a - expected_a) + abs(pred_b - expected_b)) / 2.0
            operand_mae_count += 1

            pred_bin = tmp_dir / f"pred_{idx:06d}"
            compile_addition_binary(pred_a, pred_b, pred_bin)
            verification = verify_binary(pred_bin, str(expected_stdout))
            reward = compute_reward(verification, beta=beta)

            sum_valid += verification["metrics"]["macho_valid"]
            sum_exec += verification["metrics"]["exec_ok"]
            sum_correct += verification["metrics"]["stdout_correct"]
            rewards.append(float(reward["reward"]))

    total = len(eval_rows)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    return {
        "count": total,
        "val_loss": sum(val_losses) / max(1, len(val_losses)),
        "val_target_tokens": val_tokens,
        "parse_success_rate": sum_parse_ok / total,
        "macho_valid_rate": sum_valid / total,
        "exec_success_rate": sum_exec / total,
        "answer_accuracy": sum_correct / total,
        "patch_exact_match_rate": sum_exact_patch / total,
        "operand_mae": (operand_mae_sum / operand_mae_count) if operand_mae_count > 0 else None,
        "avg_reward": avg_reward,
    }


def run_sft(
    config_path: Path,
    run: dict[str, Any],
    data: dict[str, Any],
    model_cfg: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    init: dict[str, Any],
    rl: dict[str, Any],
    train: dict[str, Any],
) -> int:
    output_dir = resolve_path(str(run["output_dir"]))
    train_jsonl = resolve_path(str(data["train_jsonl"]))
    val_jsonl = resolve_path(str(data["val_jsonl"]))

    mlx = detect_mlx()
    if not mlx["available"]:
        raise RuntimeError(f"MLX is unavailable: {mlx['error']}")

    # Delayed MLX imports keep startup graceful in environments missing MLX.
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from src.binary.pipeline import DEFAULT_BETA

    seed = int(run["seed"])
    random.seed(seed)
    mx.random.seed(seed)

    settings = parse_sft_settings(data, train)
    ensure_patch_target_mode(str(data["track"]), settings)

    train_rows = load_jsonl(train_jsonl)
    val_rows = load_jsonl(val_jsonl)
    if not train_rows:
        raise ValueError(f"No training rows found in {train_jsonl}")
    if not val_rows:
        raise ValueError(f"No validation rows found in {val_jsonl}")
    test_rows: list[dict[str, Any]] = []
    test_jsonl_path: Path | None = None
    test_jsonl_raw = str(data.get("test_jsonl", "")).strip()
    if test_jsonl_raw:
        test_jsonl_path = resolve_path(test_jsonl_raw)
    else:
        candidate_default_test = resolve_path("data/processed/test.jsonl")
        if candidate_default_test.exists():
            test_jsonl_path = candidate_default_test
    if test_jsonl_path is not None and test_jsonl_path.exists():
        test_rows = load_jsonl(test_jsonl_path)

    tokenizer = ByteTokenizer()
    model = make_model(tokenizer.vocab_size, model_cfg, mx, nn)

    init_mode = str(init.get("mode", "scratch"))
    warmstart_raw = str(init.get("warmstart_ckpt", "")).strip()
    warmstart_loaded = None
    if init_mode != "scratch":
        weights_path = resolve_warmstart_weights(warmstart_raw)
        if weights_path is None:
            print(
                f"[train.py:sft] warning: init.mode={init_mode} but no usable weights found at '{warmstart_raw}'. "
                "Falling back to scratch."
            )
        else:
            try:
                model.load_weights(str(weights_path), strict=False)
                warmstart_loaded = str(weights_path)
                print(f"[train.py:sft] loaded warmstart weights from {weights_path}")
            except Exception as exc:  # pragma: no cover
                print(
                    f"[train.py:sft] warning: failed to load warmstart weights from {weights_path}: {exc}. "
                    "Falling back to scratch."
                )

    base_lr = float(optimizer_cfg.get("lr_adamw", optimizer_cfg.get("learning_rate", 3e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    betas = optimizer_cfg.get("betas", [0.9, 0.95])
    eps = float(optimizer_cfg.get("eps", 1e-8))

    scheme = str(optimizer_cfg.get("scheme", "adamw"))
    if scheme != "adamw":
        print(f"[train.py:sft] warning: optimizer.scheme='{scheme}' requested. Using AdamW per project decision.")

    optimizer = optim.AdamW(
        learning_rate=base_lr,
        weight_decay=weight_decay,
        betas=(float(betas[0]), float(betas[1])),
        eps=eps,
    )

    def loss_fn(model_ref: Any, input_ids: Any, labels: Any, loss_mask: Any) -> Any:
        logits = model_ref(input_ids)
        per_token = nn.losses.cross_entropy(
            logits,
            labels,
            label_smoothing=settings.label_smoothing,
            reduction="none",
        )
        denom = mx.maximum(loss_mask.sum(), mx.array(1.0, dtype=per_token.dtype))
        return (per_token * loss_mask).sum() / denom

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    steps_per_epoch = max(1, math.ceil(len(train_rows) / settings.batch_size))
    planned_steps = settings.epochs * steps_per_epoch
    total_steps = planned_steps if settings.max_steps <= 0 else min(settings.max_steps, planned_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train_log.jsonl"
    eval_log_path = output_dir / "eval_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()
    if eval_log_path.exists():
        eval_log_path.unlink()

    best_full_eval: dict[str, float] | None = None
    best_test_eval: dict[str, Any] | None = None
    best_step = -1
    mini_eval_count = 0
    full_eval_count = 0
    mini_eval_rows = deterministic_subset(val_rows, settings.eval_max_samples, seed + 7001)

    global_step = 0
    seen_tokens = 0

    print(f"[train.py:sft] config={config_path}")
    print(
        "[train.py:sft] "
        f"train_rows={len(train_rows)} val_rows={len(val_rows)} "
        f"target_mode={settings.target_mode} batch_size={settings.batch_size} "
        f"total_steps={total_steps} full_eval_every={settings.full_eval_every}"
    )

    for epoch in range(1, settings.epochs + 1):
        if global_step >= total_steps:
            break

        indices = list(range(len(train_rows)))
        random.Random(seed + epoch).shuffle(indices)

        for start in range(0, len(indices), settings.batch_size):
            if global_step >= total_steps:
                break

            batch_indices = indices[start : start + settings.batch_size]
            batch_rows = [train_rows[idx] for idx in batch_indices]

            input_ids, labels, loss_mask, target_tokens = collate_batch(
                batch_rows,
                tokenizer,
                int(model_cfg["seq_len"]),
                mx=mx,
            )

            step_lr = cosine_lr(
                global_step,
                base_lr,
                settings.warmup_steps,
                total_steps,
                settings.min_lr_ratio,
            )
            optimizer.learning_rate = step_lr

            loss, grads = loss_and_grad_fn(model, input_ids, labels, loss_mask)
            if settings.grad_clip_norm > 0:
                grads, grad_norm = optim.clip_grad_norm(grads, settings.grad_clip_norm)
            else:
                grad_norm = mx.array(0.0)

            optimizer.update(model, grads)
            mx.eval(loss, grad_norm)

            loss_value = float(loss.item())
            grad_norm_value = float(grad_norm.item())

            global_step += 1
            seen_tokens += target_tokens

            train_row = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss_value,
                "grad_norm": grad_norm_value,
                "lr": step_lr,
                "target_tokens": target_tokens,
                "seen_target_tokens": seen_tokens,
                "timestamp": iso_now(),
            }
            append_jsonl(train_log_path, train_row)

            if global_step % settings.log_interval == 0 or global_step == 1:
                print(
                    "[train.py:sft] "
                    f"step={global_step}/{total_steps} epoch={epoch} "
                    f"loss={loss_value:.4f} grad_norm={grad_norm_value:.4f} lr={step_lr:.6f}"
                )

            should_eval = (global_step % settings.eval_interval == 0) or (global_step == total_steps)
            if should_eval:
                mini_eval_count += 1
                mini_metrics = evaluate_model(
                    model,
                    mini_eval_rows,
                    tokenizer,
                    settings,
                    model_cfg,
                    label_smoothing=settings.label_smoothing,
                    beta=float(rl.get("beta", DEFAULT_BETA)),
                    mx=mx,
                    nn=nn,
                )
                mini_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "timestamp": iso_now(),
                    "eval_scope": "mini",
                    **mini_metrics,
                }
                append_jsonl(eval_log_path, mini_payload)

                print(
                    "[train.py:sft:eval-mini] "
                    f"step={global_step} val_loss={mini_metrics['val_loss']:.4f} "
                    f"parse_success_rate={mini_metrics['parse_success_rate']:.3f} "
                    f"answer_accuracy={mini_metrics['answer_accuracy']:.3f} "
                    f"exec_success_rate={mini_metrics['exec_success_rate']:.3f}"
                )

                should_run_full = (mini_eval_count % settings.full_eval_every == 0) or (global_step == total_steps)
                if should_run_full:
                    full_eval_count += 1
                    full_metrics = evaluate_model(
                        model,
                        val_rows,
                        tokenizer,
                        settings,
                        model_cfg,
                        label_smoothing=settings.label_smoothing,
                        beta=float(rl.get("beta", DEFAULT_BETA)),
                        mx=mx,
                        nn=nn,
                    )
                    full_payload = {
                        "step": global_step,
                        "epoch": epoch,
                        "timestamp": iso_now(),
                        "eval_scope": "full",
                        **full_metrics,
                    }
                    append_jsonl(eval_log_path, full_payload)

                    print(
                        "[train.py:sft:eval-full] "
                        f"step={global_step} val_loss={full_metrics['val_loss']:.4f} "
                        f"parse_success_rate={full_metrics['parse_success_rate']:.3f} "
                        f"answer_accuracy={full_metrics['answer_accuracy']:.3f} "
                        f"exec_success_rate={full_metrics['exec_success_rate']:.3f} "
                        f"macho_valid_rate={full_metrics['macho_valid_rate']:.3f}"
                    )

                    candidate = {
                        "answer_accuracy": float(full_metrics["answer_accuracy"]),
                        "exec_success_rate": float(full_metrics["exec_success_rate"]),
                        "val_loss": float(full_metrics["val_loss"]),
                    }
                    if is_better_checkpoint(candidate, best_full_eval):
                        best_full_eval = candidate
                        best_step = global_step
                        best_weights_path = output_dir / "best_weights.safetensors"
                        mx.eval(model.parameters())
                        model.save_weights(str(best_weights_path))
                        if test_rows:
                            best_test_eval = evaluate_model(
                                model,
                                test_rows,
                                tokenizer,
                                settings,
                                model_cfg,
                                label_smoothing=settings.label_smoothing,
                                beta=float(rl.get("beta", DEFAULT_BETA)),
                                mx=mx,
                                nn=nn,
                            )
                            append_jsonl(
                                eval_log_path,
                                {
                                    "step": global_step,
                                    "epoch": epoch,
                                    "timestamp": iso_now(),
                                    "eval_scope": "test_on_best",
                                    **best_test_eval,
                                },
                            )
                            print(
                                "[train.py:sft:test-on-best] "
                                f"step={global_step} answer_accuracy={best_test_eval['answer_accuracy']:.3f} "
                                f"exec_success_rate={best_test_eval['exec_success_rate']:.3f}"
                            )

            if global_step % settings.save_interval == 0 or global_step == total_steps:
                latest_weights_path = output_dir / "latest_weights.safetensors"
                mx.eval(model.parameters())
                model.save_weights(str(latest_weights_path))

    latest_weights_path = output_dir / "latest_weights.safetensors"
    if not latest_weights_path.exists():
        mx.eval(model.parameters())
        model.save_weights(str(latest_weights_path))
    best_weights_path = output_dir / "best_weights.safetensors"
    if not best_weights_path.exists():
        mx.eval(model.parameters())
        model.save_weights(str(best_weights_path))
    if best_full_eval is None:
        best_full_eval = {
            "answer_accuracy": 0.0,
            "exec_success_rate": 0.0,
            "val_loss": float("inf"),
        }

    checkpoint = {
        "phase": "sft",
        "created_at": iso_now(),
        "config": str(config_path),
        "run_name": run["name"],
        "status": "trained",
        "track": data["track"],
        "target_mode": settings.target_mode,
        "init_mode": init_mode,
        "warmstart_loaded": warmstart_loaded,
        "mlx": mlx,
        "global_step": global_step,
        "seen_target_tokens": seen_tokens,
        "mini_eval_count": mini_eval_count,
        "full_eval_count": full_eval_count,
        "best_step": best_step,
        "best_by_answer_accuracy": {
            "step": best_step,
            "answer_accuracy": (best_full_eval or {}).get("answer_accuracy"),
            "exec_success_rate": (best_full_eval or {}).get("exec_success_rate"),
            "val_loss": (best_full_eval or {}).get("val_loss"),
        },
        "test_eval_on_best": best_test_eval,
        "test_rows": len(test_rows),
        "test_jsonl": str(test_jsonl_path) if test_jsonl_path else None,
        "latest_weights": str(latest_weights_path),
        "best_weights": str(best_weights_path),
        "train_log": str(train_log_path),
        "eval_log": str(eval_log_path),
    }

    metrics = {
        "phase": "sft",
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "tokenization_mode": data["tokenization_mode"],
        "track": data["track"],
        "target_mode": settings.target_mode,
        "model": model_cfg,
        "optimizer": {
            "scheme": "adamw",
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "betas": [float(betas[0]), float(betas[1])],
            "eps": eps,
        },
        "init": init,
        "rl": rl,
        "train": {
            "epochs": settings.epochs,
            "max_steps": settings.max_steps,
            "batch_size": settings.batch_size,
            "log_interval": settings.log_interval,
            "eval_interval": settings.eval_interval,
            "save_interval": settings.save_interval,
            "eval_max_samples": settings.eval_max_samples,
            "max_gen_tokens": settings.max_gen_tokens,
            "grad_clip_norm": settings.grad_clip_norm,
            "warmup_steps": settings.warmup_steps,
            "min_lr_ratio": settings.min_lr_ratio,
            "label_smoothing": settings.label_smoothing,
            "full_eval_every": settings.full_eval_every,
        },
        "global_step": global_step,
        "seen_target_tokens": seen_tokens,
        "mini_eval_count": mini_eval_count,
        "full_eval_count": full_eval_count,
        "best_step": best_step,
        "best_answer_accuracy": (best_full_eval or {}).get("answer_accuracy"),
        "best_exec_success_rate": (best_full_eval or {}).get("exec_success_rate"),
        "best_val_loss": (best_full_eval or {}).get("val_loss"),
        "test_eval_on_best": best_test_eval,
        "notes": "Real SFT loop with AdamW on Track-A patch targets.",
    }

    write_json(output_dir / "checkpoint.json", checkpoint)
    write_json(output_dir / "metrics.json", metrics)

    print(f"[train.py:sft] checkpoint={output_dir / 'checkpoint.json'}")
    print(f"[train.py:sft] metrics={output_dir / 'metrics.json'}")
    return 0


def run_rl(
    config_path: Path,
    run: dict[str, Any],
    data: dict[str, Any],
    model_cfg: dict[str, Any],
    optimizer: dict[str, Any],
    init: dict[str, Any],
    rl: dict[str, Any],
    train: dict[str, Any],
) -> int:
    del train
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
        "model": model_cfg,
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
    run, data, model, optimizer, init, rl, train = validate_phase_config(cfg)

    phase = str(run["phase"])
    if phase == "sft":
        return run_sft(config_path, run, data, model, optimizer, init, rl, train)
    if phase == "rl":
        return run_rl(config_path, run, data, model, optimizer, init, rl, train)

    raise ConfigError(f"Unsupported phase: {phase}")


if __name__ == "__main__":
    raise SystemExit(main())

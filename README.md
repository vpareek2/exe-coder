# exe-coder

Research project: test whether natural language can map directly to executable machine code and eventually to optimized binaries, reducing dependence on source-language/IR-centric pipelines.

## Core Question

Can a transformer learn:
1. `NL -> executable binary` (correct and runnable)
2. `NL -> optimized executable binary` (correct plus better runtime/size properties)

Current toy domain: integer addition on macOS arm64 (Apple Silicon).

---

## Experiment Design (Dual-Track)

We intentionally run two tracks under one evaluator.

### Track A: Tractable Directness (Template-Delta)

- Model predicts compact patch values (operands `a,b`) from NL.
- Deterministic reconstructor applies the patch to a fixed binary template.
- Final artifact is a full executable Mach-O binary.
- Runtime semantics remain program-like (binary computes `a+b` at runtime).

Why this exists:
- Fast iteration and stable learning signal.
- Lets us validate NL understanding + executable generation pipeline reliably.

### Track B: Strict Directness (One-Shot Full Binary)

- Model predicts full binary bytes autoregressively.
- No deterministic patch/reconstructor shortcut in inference path.
- Highest-fidelity test of the long-term thesis.

Why this exists:
- Direct evidence for NL -> machine code feasibility.
- Harder sequence modeling problem; expected to be slower and less stable early.

### Shared Protocol

Both tracks share:
- same prompt distributions
- same held-out splits
- same verifier and metrics

This keeps comparison scientifically meaningful.

---

## Scope (v1)

In scope:
- signed integer addition
- single executable output per prompt
- Mach-O arm64 on macOS
- Python-only training stack with MLX

Out of scope for now:
- multi-operation arithmetic language
- multi-file programs
- hard security sandboxing for adversarial binaries
- production deployment

---

## Python-MLX Stack

Canonical training runner:

```bash
uv run train.py --config <toml>
```

Canonical dataset runner:

```bash
uv run generate.py --all --train-n <N> --val-n <N> --test-n <N>
```

Environment policy:
- use repo-local `uv` virtualenv only (`.venv`)
- run scripts through `uv run ...` (not direct `python ...`)
- manage dependencies from `pyproject.toml` via `uv sync`

Setup:

```bash
uv venv .venv
uv sync
# Optional: install MLX training dependency group
uv sync --group train
```

Training implementation direction:
- Python MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`)
- root-level config-runner (`train.py`)
- TOML configs in `configs/`

Rationale:
- fastest research iteration on M3
- easier ablation matrix execution and debugging
- cleaner path to frequent experiment updates

---

## Initialization Ablations (Run in Parallel)

SFT runs three init arms:
1. `scratch`
2. `warmstart`
3. `warmstart_textmix`

Purpose:
- quantify value of language priors for NL -> binary mapping
- measure whether mixed text objective helps retain language grounding

Configs:
- `configs/sft_scratch.toml`
- `configs/sft_warmstart.toml`
- `configs/sft_warmstart_textmix.toml`

---

## Optimizer Plan (Day One)

Scheme: `adamw`

Why:
- Strong baseline stability for small-model laptop experiments.
- Simpler debugging while we validate architecture and training loop behavior.
- Muon is deferred until after SFT correctness is stable.

---

## RL Plan

Mainline branch:
1. train SFT baseline
2. RL fine-tune from SFT checkpoint

Exploratory branch:
1. RL-from-scratch (high risk)
2. tracked as separate experiment, not replacing mainline

Reward contract:

```text
reward = 1.0*is_valid_macho + 1.0*exec_ok + 2.0*stdout_correct - 1.0*timeout_or_crash - beta*kl_to_sft
```

RL configs:
- `configs/rl_mainline.toml`
- `configs/rl_scratch_exploratory.toml`

---

## Data and Verification Pipeline

Stable components currently in use:
- dataset runner: `generate.py`
- compile: `scripts/compile_target.py`
- verify: `scripts/verify_binary.py`
- phase-A checks: `scripts/run_phase_a_checks.py`
- dataset generation: `scripts/generate_dataset.py`
- SFT preparation: `scripts/prepare_sft_data.py`
- inference baseline: `scripts/infer.py`
- evaluation: `scripts/evaluate.py`

Core library modules:
- `src/binary/pipeline.py`
- `src/binary/tokenization.py`

Runtime contract:
- generated executable must print exactly `"<sum>\n"`.

---

## Training Config Schema (TOML)

All training configs follow this schema:

```toml
[run]
phase = "sft|rl"
name = "experiment_name"
seed = 1337
output_dir = "outputs/checkpoints/..."

[data]
train_jsonl = "data/processed/..."
val_jsonl = "data/processed/..."
test_jsonl = "data/processed/test.jsonl" # optional; used for test-on-best eval
tokenization_mode = "hex_pair"
track = "template_delta_operand_patch|strict_one_shot"
target_mode = "patch_text|full_hex"

[model]
layers = 6
hidden_size = 512
heads = 8
seq_len = 2048
dropout = 0.0

[optimizer]
scheme = "adamw"
lr_adamw = 3e-4
weight_decay = 0.01
betas = [0.9, 0.95]
eps = 1e-8

[init]
mode = "scratch|warmstart|warmstart_textmix"
warmstart_ckpt = ""
textmix_ratio = 0.0

[rl]
beta = 0.01
rollout_samples = 32
mode = "mainline|scratch_exploratory"
init_ckpt = ""

[train]
batch_size = 16
epochs = 20
max_steps = 200
log_interval = 2
eval_interval = 20
full_eval_every = 5
save_interval = 20
eval_max_samples = 64
max_gen_tokens = 32
grad_clip_norm = 1.0
warmup_steps = 10
min_lr_ratio = 0.1
label_smoothing = 0.0
```

---

## Bootstrap Commands

```bash
# 1) Determinism + golden correctness checks
uv run scripts/run_phase_a_checks.py

# 2) Build dataset splits
uv run generate.py --all --train-n 10000 --val-n 1000 --test-n 1000

# 3) Prepare SFT sequences
uv run scripts/prepare_sft_data.py --input data/processed/train.jsonl --out data/processed/train_sft.jsonl

# 4) Run SFT ablations
uv run train.py --config configs/sft_scratch.toml
uv run train.py --config configs/sft_warmstart.toml
uv run train.py --config configs/sft_warmstart_textmix.toml

# 5) Run RL branches
uv run train.py --config configs/rl_mainline.toml
uv run train.py --config configs/rl_scratch_exploratory.toml

# 6) Evaluate pipeline
uv run scripts/infer.py --input data/processed/test.jsonl --out outputs/predictions/test.jsonl
uv run scripts/evaluate.py --ckpt outputs/checkpoints/rl_mainline --split test --out outputs/reports/test.json

# 7) Evaluate learned SFT patch model
uv run scripts/infer.py --mode sft_patch_model --weights outputs/checkpoints/sft_scratch/best_weights.safetensors --input data/processed/val.jsonl --out outputs/predictions/val_model.jsonl
uv run scripts/evaluate.py --ckpt outputs/checkpoints/sft_scratch --split val --predictions outputs/predictions/val_model.jsonl --out outputs/reports/val_model_eval.json

# Optional: one-command acceptance flow
uv run scripts/run_acceptance.py
```

---

## Evaluation Metrics

Primary:
- `train_loss`
- `val_loss`
- `parse_success_rate`
- `patch_exact_match_rate`
- `operand_mae`
- `macho_valid_rate`
- `exec_success_rate`
- `answer_accuracy`

Secondary:
- `exact_byte_match_rate`
- `avg_reward`
- latency and error taxonomy

Generalization axes:
- unseen prompt templates
- held-out magnitude buckets
- operand sign patterns (`++`, `+-`, `-+`, `--`)

Slice reports:
- template slices (`prompt_template`)
- magnitude pair slices (`core/core`, `core/heldout`, `heldout/core`, `heldout/heldout`)
- sign-pattern slices

---

## Repository Layout

```text
exe-coder/
  README.md
  pyproject.toml
  generate.py
  train.py
  configs/
  scripts/
  src/
    binary/
  data/
    raw/
    processed/
  outputs/
    checkpoints/
    predictions/
    reports/
  ref/
```

---

## Migration Note

The previous C++/CMake training scaffold has been removed from the active workflow.
The official path is now pure Python MLX via:
- `uv run generate.py ...`
- `uv run train.py --config <toml>`

---

## Project Status

Python-MLX migration phase.
Deterministic compile/verify/data/eval tooling is active.
Training runner is in place with phase dispatch and experiment-matrix config scaffolding.

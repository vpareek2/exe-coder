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
uv run generate.py --all --profile stage1 --out-dir data/processed/stage1 --binaries-dir outputs/bin/stage1
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

## Current SFT Policy

Current default path is scratch-only staged training for Track A:
1. `configs/sft_stage0_overfit.toml`
2. `configs/sft_stage1_pilot.toml`
3. `configs/sft_stage1a_template_shift.toml` (template shift isolated)
4. `configs/sft_stage1b_magnitude_shift.toml` (magnitude shift isolated)
5. `configs/sft_stage2_scale.toml`

Warmstart/textmix configs remain available for later ablations:
- `configs/sft_scratch.toml`
- `configs/sft_warmstart.toml`
- `configs/sft_warmstart_textmix.toml`

Run policy:
- reach Stage 1 (`G1`) with scratch first
- run warmstart/textmix matrix after scratch baseline is stable

---

## Optimizer Plan (Day One)

Scheme: `adamw`

Why:
- Strong baseline stability for small-model laptop experiments.
- Simpler debugging while we validate architecture and training loop behavior.
- Muon is deferred until after SFT correctness is stable.

---

## Staged Track-A Regimen

Stage profiles are built into `generate.py`:
- `stage0`: overfit sanity (`train=512`, `val=128`, `test=128`, range `[-200, 200]`, train templates on all splits)
- `stage1`: pilot generalization (`train=10000`, `val=1000`, `test=1000`, range `[-1000, 1000]`, held-out val/test templates)
- `stage1a`: template shift only (`train templates for train`, held-out templates for val/test, no held-out magnitude sampling)
- `stage1b`: magnitude shift only (`train templates for all splits`, held-out magnitude sampling on val/test)
- `stage2`: scale-up (`train=50000`, `val=5000`, `test=5000`, range `[-5000, 5000]`, held-out val/test templates)

Track-A decoding is constrained to `A=<int>;B=<int>\n` with per-config operand bounds (`data.decode_min_int`, `data.decode_max_int`).

Milestone gates:
- `G0` (loop sanity): train loss decreases + `parse_success_rate >= 0.95` + `answer_accuracy >= 0.90`
- `G1` (pilot success): `answer_accuracy >= 0.50` + `exec_success_rate >= 0.70` + no catastrophic slice collapse
- `G2` (RL-ready): `answer_accuracy >= 0.75` + `parse_success_rate >= 0.95` + `heldout/heldout answer_accuracy >= 0.55`

Gate status is emitted in evaluation reports under `milestone_gates`.

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
- staged pipeline runner: `scripts/run_track_a_stage.py`
- compile: `scripts/compile_target.py`
- verify: `scripts/verify_binary.py`
- phase-A checks: `scripts/run_phase_a_checks.py`
- dataset generation: `scripts/generate_dataset.py`
- dataset quality checks: `scripts/check_dataset_quality.py`
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
decode_min_int = -1000
decode_max_int = 1000

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

# 2) Build Stage 0 dataset (train-style templates across train/val/test)
uv run generate.py --all --profile stage0 --out-dir data/processed/stage0 --binaries-dir outputs/bin/stage0

# 3) Prepare Stage 0 SFT sequences
uv run scripts/prepare_sft_data.py --input data/processed/stage0/train.jsonl --out data/processed/stage0/train_sft.jsonl

# 4) Dataset quality checks (sign coverage + slice availability)
uv run scripts/check_dataset_quality.py --train data/processed/stage0/train.jsonl --val data/processed/stage0/val.jsonl --test data/processed/stage0/test.jsonl

# 5) Run staged scratch SFT
uv run train.py --config configs/sft_stage0_overfit.toml

# 6) Evaluate learned Stage 0 model on validation
uv run scripts/infer.py --mode sft_patch_model --weights outputs/checkpoints/sft_stage0_overfit/best_weights.safetensors --input data/processed/stage0/val.jsonl --out outputs/predictions/stage0_val_model.jsonl
uv run scripts/evaluate.py --ckpt outputs/checkpoints/sft_stage0_overfit --split val --predictions outputs/predictions/stage0_val_model.jsonl --out outputs/reports/stage0_val_model_eval.json

# 7) Optional one-command stage runner (generate -> prepare -> train -> infer -> evaluate)
uv run scripts/run_track_a_stage.py --stage stage0

# 8) Move to Stage 1/2 when the previous gate passes
uv run scripts/run_track_a_stage.py --stage stage1
uv run scripts/run_track_a_stage.py --stage stage1a
uv run scripts/run_track_a_stage.py --stage stage1b
uv run scripts/run_track_a_stage.py --stage stage2

# Optional smoke acceptance run
uv run scripts/run_acceptance.py --train-n 16 --val-n 8 --test-n 8
```

---

## Evaluation Metrics

Primary:
- `train_loss`
- `val_loss`
- `train_loss_start`
- `train_loss_end`
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
- milestone gates (`g0_loop_sanity`, `g1_pilot_success`, `g2_rl_ready`)

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

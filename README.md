# exe-coder

Research project: train a transformer that maps natural language requests to a **compiled executable binary**.

Core task (v1): **basic integer addition**
- Input: natural language, e.g. `"add 17 and 25"`
- Output: an executable **Mach-O arm64 binary file** (not source code)
- Execution target: macOS on Apple Silicon (M3)
- Success condition: running the generated binary prints the correct sum (for example, `42`)

---

## Why this project

Generating a Python script is easy. Generating a valid compiled binary is materially harder.

This project tests the harder question:

> Can a transformer learn NL -> native executable bytes well enough to produce binaries that are both valid and behaviorally correct?

Why this is difficult:
- Binary formats are brittle (one wrong byte can break execution)
- Native executables encode strict structural constraints (headers, sections, alignment, load commands)
- Long byte sequences increase decoding and context complexity
- Functional correctness requires both syntactic validity and runtime semantics

---

## Research Goals

1. Build a deterministic dataset of `natural language -> compiled binary` pairs for addition.
2. Train a transformer in two phases: supervised learning first, then RL fine-tuning with executable/correctness rewards.
3. Measure validity, executability, and arithmetic correctness on held-out prompts.
4. Compare direct-binary generation with stronger baselines.
5. Test whether more examples improve true generalization rather than memorization.

Success criteria (v1):
- `>= 99%` binaries recognized as valid Mach-O arm64 by validation checks
- `>= 95%` execution success rate on held-out prompts
- `>= 90%` exact arithmetic correctness on held-out prompts
- Robustness across unseen phrasing templates

---

## Scope (v1)

In scope:
- Addition of signed integers only
- Single executable binary output per prompt
- Target format: Mach-O 64-bit, arm64
- Automated execution-based evaluation
- Apple Silicon local experimentation on M3

Out of scope (for now):
- Multi-operation arithmetic
- Multi-file projects
- Full sandbox hardening for adversarial payloads
- Production deployment

---

## System Concept

Pipeline:
1. Generate NL prompts and compile canonical target binaries with a deterministic toolchain.
2. Serialize binary bytes into model tokens.
3. Train an autoregressive transformer on `PROMPT -> BINARY` sequence pairs.
4. Decode bytes at inference and write output as an executable file.
5. Validate binary format and run behavioral tests.

Recommended training sequence:
- `PROMPT: <natural language request>`
- `TARGET_BIN_HEX: <hex tokens for executable bytes>`
- Training objective: next-token prediction over the concatenated sequence

Primary challenge:
- Learn both semantic mapping (`17 + 25`) and low-level binary structure simultaneously.

---

## Dataset Design

Example record:

```json
{
  "id": "ex_000001",
  "prompt": "Please add 17 and 25",
  "a": 17,
  "b": 25,
  "target_format": "macho64",
  "target_arch": "arm64",
  "target_os": "macos",
  "binary_b64": "<base64-bytes>",
  "binary_sha256": "<sha256>",
  "byte_length": 1840,
  "expected_stdout": "42\n"
}
```

Data generation strategy:
- Programmatically sample integer pairs from a defined range
- Use varied NL templates (`add`, `sum`, `plus`, `total`)
- Include synonyms: add/sum/plus/total
- Include signs: positive, zero, negative
- Compile with deterministic flags and fixed toolchain versions
- Record compiler metadata for reproducibility

Suggested split:
- Train: 80%
- Validation: 10%
- Test: 10%

Generalization test ideas:
- Hold out larger magnitudes in test
- Hold out certain phrasing templates from train
- Hold out negative-number-heavy examples

Determinism requirement:
- The same `(prompt, a, b)` and toolchain config should produce identical bytes (same SHA-256).
- If determinism is not achieved, exact-byte metrics become noisy.

---

## Model Plan (M3 + MLX)

Start with a small decoder-only transformer:
- Layers: 6-12
- Hidden size: 384-768
- Heads: 6-12
- Params: ~20M to ~120M

Why small first:
- Faster iteration on M3
- Easier debugging of binary tokenization/eval loops
- Lets us prove direct-binary feasibility before scaling

Implementation stack:
- [MLX](https://github.com/ml-explore/mlx) with C++ for model/training code
- Apple Silicon acceleration via MLX backends on M3

Training notes for Apple Silicon:
- Prefer smaller batch sizes with gradient accumulation
- Track thermal throttling during long runs
- Start with shorter binaries (or binary deltas) to control sequence length

Tokenization options to compare:
- Byte tokens (`0-255`)
- Hex-pair tokens (`00-ff`)
- Chunked tokens (2-4 byte groups)

---

## Training Strategy

Phase 1: Supervised training (standard next-token training)
- Train on `PROMPT -> TARGET_BIN_HEX` pairs with teacher forcing.
- Objective: strong initialization for syntax and structure of valid binaries.

Phase 2: RL fine-tuning (verifiable-domain optimization)
- Sample binaries from model policy.
- Execute verifier pipeline and compute reward from objective checks.
- Optimize for runnable and correct binaries, not just token imitation.

Reward sketch (example):
- `+1.0` if Mach-O is valid
- `+1.0` if binary executes successfully
- `+2.0` if stdout equals correct addition result
- `-1.0` for invalid binary or timeout

Why RL here:
- Domain is highly verifiable.
- Binary quality has sparse, outcome-based signals that supervised loss does not fully capture.

---

## Evaluation

Primary metrics:
- `macho_valid_rate`: file passes Mach-O format checks
- `exec_success_rate`: file launches and exits cleanly
- `answer_accuracy`: stdout matches expected sum

Secondary metrics:
- exact byte match vs reference binary
- inference latency
- error type distribution

Failure taxonomy:
- invalid binary format
- non-executable permissions or launch failure
- runtime error
- wrong arithmetic result
- prompt misinterpretation

Suggested validator stack:
- Static check: `file` / `otool` header validation
- Runtime check: execute with timeout and capture stdout/stderr
- Correctness check: compare stdout to expected result
- RL diagnostics: average reward, reward component breakdown, and pass@k correctness

---

## Experiments Roadmap

1. Deterministic binary baseline (non-ML)
   - Parse prompt, compute sum, emit binary via fixed compile pipeline
2. Tiny direct-binary transformer
   - Verify end-to-end generation of runnable binaries
3. RL phase on top of supervised checkpoint
   - Optimize executable/correctness reward and compare against SFT-only
4. Data scaling study
   - 10k, 50k, 100k+ examples as hardware allows
5. Tokenization ablation
   - Byte vs hex vs chunked tokens
6. Model size ablation
   - Measure gains relative to sequence length and decode stability

Key research question:
- Does supervised + RL materially outperform supervised-only for valid, correct binaries?

---

## Proposed Repository Layout

```text
exe-coder/
  README.md
  data/
    raw/
    processed/
  scripts/
    generate_dataset.py
    compile_target.py
    train.py
    evaluate.py
    infer.py
  src/
    binary/
    model/
    tokenization/
    training/
    evaluation/
  outputs/
    checkpoints/
    predictions/
    reports/
```

---

## Milestones

M1: Deterministic target pipeline
- Build compile pipeline that emits canonical Mach-O binaries
- Verify deterministic byte reproducibility with hashing

M2: Dataset + evaluator
- Generate first NL -> binary dataset
- Build binary validator + execution harness

M3: First direct-binary supervised model
- Train tiny model and produce first validity/correctness report

M4: RL fine-tuning phase
- Run RL on top of the supervised checkpoint using execution rewards
- Compare SFT-only vs SFT+RL across validation/test metrics

M5: Scaling and ablations
- Scale data/model and run tokenization experiments

M6: Research write-up
- Summarize method, results, and limitations

---

## Risks and Mitigations

Risk: non-deterministic compiler output corrupts supervision  
Mitigation: lock toolchain flags/version; track SHA reproducibility checks

Risk: model memorizes template-bytes instead of learning semantics  
Mitigation: hold out prompt styles and number ranges

Risk: binary sequences are too long/noisy for small models  
Mitigation: start with compact targets, evaluate chunked tokenization, test binary-delta approaches

Risk: RL becomes unstable or reward-hacked  
Mitigation: use conservative KL regularization to supervised policy, strict verifier checks, and held-out evaluation

Risk: hardware limits training scale on M3  
Mitigation: start small, profile memory, use gradient accumulation/checkpointing

---

## Immediate Next Steps

1. Implement deterministic binary compile recipe and verify stable SHA-256 outputs.
2. Implement `scripts/generate_dataset.py` for NL prompts + binary artifacts.
3. Set up minimal MLX C++ training loop for supervised `PROMPT -> BINARY` modeling.
4. Build `evaluate.py` with Mach-O validation + runtime correctness checks + reward logging.
5. Add RL fine-tuning loop and publish SFT-only vs SFT+RL comparison metrics.

---

## Project Status

Planning phase.  
This README defines a direct-binary research protocol for NL -> executable generation.

# Discover on Trainium

This port replaces Tinker with local PyTorch/Neuron LoRA training and keeps kernel verification, benchmarking, rewards, and search on the Trainium host. Selected upstream snapshots are tracked in [`../../discover-reference/`](../../discover-reference/README.md), commit `6c40e82dab9d5de7416ac873ad5cd3106084aaed`.

`vendor/PROVENANCE.json` identifies retained reference code: State, PUCTSampler, adaptive entropic advantages, GPT-OSS renderer, and TwoPhaseTokenCompleter. Tests compare the algorithm ASTs to those snapshots. `discover.py` samples parents, generates groups, evaluates NKI candidates, updates search, computes advantages, performs a reward-weighted LoRA update, and uses that checkpoint for the next rollout. Constant-reward groups are dropped as in the reference. Valid states store negative latency; scalar reward is 1000/geometric-mean microseconds. Failed candidates receive zero. Candidate diagnostics and best correct code/measurements persist across attempts and resume.

The backend uses the actual GPT-OSS-20B checkpoint, Transformers eager attention, BF16 dequantized MXFP4 weights, and rank-32 LoRA on attention q/k/v/o projections. The original EP2 path shards frozen experts across two cores. The requested TP4 path shards attention heads/projections and the intermediate channels inside every expert across four cores; embedding, LM head, router and normalization remain replicated. LoRA is created at global rank 32 before sharding: column A and row B factors are replicated and their gradients are summed once per update. Rank-local adapter hashes are gathered into a global fingerprint; checkpoint files remain per rank. Fixed-shape expert routing replaces dynamic indexing for compilation; it computes all local experts and is more expensive than sparse routing. CPU forward/backward equivalence is tested. The importance-sampling loss uses Tinker's documented sum reduction, with optional clipping and the reference policy/base KL advantage adjustment. Sampling records policy version and adapter fingerprint; stale rollouts are rejected.

This is a focused port, not the entire reference launcher: synchronous sample/evaluate/train phases, smaller group and token budgets, explicit LoRA targets, no Modal/Triton, no Tinker service, no unrelated integrations. Kernel evaluation runs after policy workers exit, releasing their cores. The task contract and benchmark methodology are in the parent README.

## Environment

The recorded hardware runs used Trainium2 `trn2.48xlarge`, one exposed device,
four cores with LNC2 and 24 GiB per core. TP4 needs all four cores free. The
optional `hardware_session.py` wrapper captures the seat's existing Qwen
server settings, pauses that service, runs a bounded test, and restores it.
It is specific to a dedicated seat with a known Qwen server PID. Other hosts
should manage their own service lifecycle and core allocation.

The isolated Python 3.12 environment pins torch 2.9.1, torch-neuronx 2.9.0.2.15.32035+de43f57c, transformers 4.57.6, peft 0.18.1, Neuron compiler 2.27.5334.0+f702b353, NKI 0.6.0+31049202112.g85070674, and islpy 2026.1. Runtime is 2.34.10. `setup_training.sh` recreates it. The SDK's distributed initializer overwrote explicitly selected cores; `apply_sdk_fix.py` applies the narrow recorded patch to this isolated environment. `launch.py` sets library paths, compiler cache, and core assignment.

Checkpoint: `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, downloaded under `../.hf-cache`. No model endpoint is needed.

## Verification status

- **Partial-model GPT-OSS-20B TP4 gate passed (2026-10-10):** original layers
  0/6/12/17/23, eight input tokens, one generated token. All four ranks passed
  forward, finite nonzero gradients, one optimizer update, updated-policy
  generation and checkpoint restore; no executed fallback ops. Qwen was
  restored and healthy. See [partial verification](PARTIAL_VERIFICATION.md)
  for per-rank evidence, retained settings and the bounded run. This does not
  validate full GPT-OSS-20B training, task RL or kernel speedup.

- Existing seed: 72 simulator and 72 on-chip correctness cases, 24 device benchmarks, geometric mean 207.8703 microseconds. This is a locally written seed, not a model improvement.
- Ten RL tests pass in the isolated Python 3.12 training environment (the seven evaluator tests use the original Python 3.13 environment): reference algorithm preservation, loss/masks, branching and persistence, routing gradients, updated sampler/checkpoint restore, two-phase completion, static cache parity, explicit reference stop handling, and two mocked search/RL iterations followed by resume. Mock orchestration is not an on-chip optimization run.
- An additional four-process CPU TP4 parity test passes: forward log probabilities, sharded and replicated gradients, two optimizer updates against the unsharded reference, global fingerprints and checkpoint restore. This is a CPU test, not a chip measurement.
- [Tiny TP4 on-chip evidence](../evidence/continuation/tiny-tp4/rank-0/result.json)
  records cached decoding, finite nonzero gradients (global norm 2.28780541),
  an optimizer update, updated-policy sampling and checkpoint restore. All
  four ranks share the updated hash; no fallback ops executed.
- Two partial-model tests check exact checkpoint loading, original indices,
  attention settings and nonadjacent selection. Their [recorded output](../evidence/continuation/partial-model-tests.txt) is retained.
- Full GPT-OSS-20B TP4 training and task-driven discovery have not passed a
  hardware gate. Earlier EP2 backward compilation failed with NCC_EOOM001
  (33.29 GiB peak against 24 GiB per core); those older local runs are not
  bundled. The superseded full TP4 attempt was cancelled during initialization;
  [its recorded status](../evidence/continuation/superseded-full.json) is retained.

## Commands

From the repository's `trainium-softmax` directory. Setup requires `uv`,
network access to the pinned package indexes, and enough local disk for weights
and caches:

```bash
bash rl/setup_training.sh
.venv-training312/bin/python rl/download_model.py
.venv-training312/bin/python -m unittest discover -s rl -v
```

For the verified partial-model procedure, see [PARTIAL_VERIFICATION.md](PARTIAL_VERIFICATION.md).
A tiny TP4 mechanics check is available when all four cores are free:

```bash
.venv-training312/bin/python rl/launch.py probe --tiny --static-experts --tensor-parallel-size 4 --cached-sampling --checkpoint-activations --sample-tokens 2 --out rl/runs/new-tiny-probe
```

The full-model and discovery entry points are implemented but remain
experimental. They are not part of the verified partial-model procedure:

```bash
MODEL=$(cat rl/model-path.txt)
.venv-training312/bin/python rl/launch.py probe --model "$MODEL" --static-experts --tensor-parallel-size 4 --checkpoint-activations --out rl/runs/new-20b-probe
.venv-training312/bin/python rl/launch.py discover --model "$MODEL" --tensor-parallel-size 4 --iterations 2 --group-size 4 --max-context 2048 --max-tokens 512 --out rl/runs/new-discovery
```

Hardware commands need access to `/dev/neuron0` outside the restricted Codex sandbox. Fresh runs require unused output directories. Resume the same discovery command with `--resume`; `--iterations` counts additional iterations. Keep the same group count when resuming. Compilation and dense expert routing can be slow; the production run remains unverified until recorded results demonstrate it.

Inspect `iteration-N.json`, `feedback-N-G-C.json`, `archive_step_N.json`, `resume.json`, worker logs/results, and `best.py`/`best.json`. A synthetic +1-advantage probe proves a gradient step; only a discovery iteration with nonconstant measured task rewards proves a task-driven RL step.

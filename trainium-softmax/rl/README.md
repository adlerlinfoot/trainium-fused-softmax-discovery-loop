# Discover on Trainium

This port replaces Tinker with local PyTorch/Neuron LoRA training and keeps kernel verification, benchmarking, rewards, and search on the Trainium host. The original repository is cloned at `/workspace/discover-reference`, commit `6c40e82dab9d5de7416ac873ad5cd3106084aaed`.

`vendor/PROVENANCE.json` identifies retained reference code: State, PUCTSampler, adaptive entropic advantages, GPT-OSS renderer, and TwoPhaseTokenCompleter. Tests compare the algorithm ASTs to the clone. `discover.py` samples parents, generates groups, evaluates NKI candidates, updates search, computes advantages, performs a reward-weighted LoRA update, and uses that checkpoint for the next rollout. Constant-reward groups are dropped as in the reference. Valid states store negative latency; scalar reward is 1000/geometric-mean microseconds. Failed candidates receive zero. Candidate diagnostics and best correct code/measurements persist across attempts and resume.

The backend uses the actual GPT-OSS-20B checkpoint, Transformers eager attention, BF16 dequantized MXFP4 weights, and rank-32 LoRA on attention q/k/v/o projections. The original EP2 path shards frozen experts across two cores. The requested TP4 path shards attention heads/projections and the intermediate channels inside every expert across four cores; embedding, LM head, router and normalization remain replicated. LoRA is created at global rank 32 before sharding: column A and row B factors are replicated and their gradients are summed once per update. Rank-local adapter hashes are gathered into a global fingerprint; checkpoint files remain per rank. Fixed-shape expert routing replaces dynamic indexing for compilation; it computes all local experts and is more expensive than sparse routing. CPU forward/backward equivalence is tested. The importance-sampling loss uses Tinker's documented sum reduction, with optional clipping and the reference policy/base KL advantage adjustment. Sampling records policy version and adapter fingerprint; stale rollouts are rejected.

This is a focused port, not the entire reference launcher: synchronous sample/evaluate/train phases, smaller group and token budgets, explicit LoRA targets, no Modal/Triton, no Tinker service, no unrelated integrations. Kernel evaluation runs after policy workers exit, releasing their cores. The task contract and benchmark methodology are in the parent README.

## Environment

Trainium2 `trn2.48xlarge`; one exposed device, four logical cores with LNC2 and 24 GiB per logical core. Qwen inference normally occupies cores 0–1. The user authorized temporarily pausing it for TP4 verification on cores 0–3; it must be restored after experiments. Legacy expert-parallel workers use cores 2–3. The full BF16 checkpoint does not fit on one core; that failed gate is preserved in `runs/gptoss20b-xla`.

The isolated Python 3.12 environment pins torch 2.9.1, torch-neuronx 2.9.0.2.15.32035+de43f57c, transformers 4.57.6, peft 0.18.1, Neuron compiler 2.27.5334.0+f702b353, NKI 0.6.0+31049202112.g85070674, and islpy 2026.1. Runtime is 2.34.10. `setup_training.sh` recreates it. The SDK's distributed initializer overwrote explicitly selected cores; `apply_sdk_fix.py` applies the narrow recorded patch to this isolated environment. `launch.py` sets library paths, compiler cache, and core assignment.

Checkpoint: `openai/gpt-oss-20b`, revision `6cee5e81ee83917806bbde320786a8fb61efebee`, downloaded under `../.hf-cache`. No model endpoint is needed.

## Verification status

- Existing seed: 72 simulator and 72 on-chip correctness cases, 24 device benchmarks, geometric mean 207.8703 microseconds. This is a locally written seed, not a model improvement.
- Ten RL tests pass in the isolated Python 3.12 training environment (the seven evaluator tests use the original Python 3.13 environment): reference algorithm preservation, loss/masks, branching and persistence, routing gradients, updated sampler/checkpoint restore, two-phase completion, static cache parity, explicit reference stop handling, and two mocked search/RL iterations followed by resume. Mock orchestration is not an on-chip optimization run.
- An additional four-process CPU TP4 parity test passes: forward log probabilities, sharded and replicated gradients, two optimizer updates against the unsharded reference, global fingerprints and checkpoint restore. This is a CPU test, not a chip measurement.
- True TP4 tiny on-chip test passes in `runs/tiny-xla-tp4`: cached decoding, activation checkpointing, finite nonzero gradients (global norm 2.28780541), an optimizer update, updated-policy generation, and checkpoint restoration. All four ranks agree on updated global hash `7aaba860e39479504098f4f13653f137e085e8f7763404ea780d1c6c77175e7a`; no executed fallback ops. This validates the four-core training mechanics with a tiny model, not full GPT-OSS-20B.
- True TP2 tiny on-chip test passes in `runs/tiny-xla-tp2b`: cached decoding, XLA reentrant activation checkpointing, synthetic +1-advantage update, global gradient norm 2.19380935, changed adapter hash, updated-policy sampling and checkpoint restore. Both ranks agree on global hash `139834547c0a40df244f0c4ec4191eb11b8f42665003f4ff321dbc6c0b7a3a55`; XLA reports no executed fallback ops. This is a small model, not GPT-OSS-20B.
- Tiny GPT-OSS architecture passes real Trainium forward/backward/optimizer/sampling/checkpoint restore on one core and two expert-parallel cores. Both parallel ranks have identical updated adapter hashes. This validates backend mechanics, not full-model support.
- Full GPT-OSS-20B expert-parallel forward graphs compiled successfully in `runs/gptoss20b-ep2`. That two-token probe was superseded to avoid an extra sequence-shape compilation; the corrected one-token gate in `runs/gptoss20b-ep2-minimal` sampled identical token 350 with log probability -4.1712145805 on both ranks, but backward compilation failed: NCC_EOOM001, 33.29 GiB peak versus 24 GiB per core (21.27 GiB I/O plus 12.01 GiB scratch). A true TP4 port passes CPU parity and is now being tested on hardware; it shards attention and each expert’s intermediate channels, preserves global rank-32 LoRA, and reduces replicated adapter gradients. Full-model training and task RL remain unverified until that gate passes. Never treat compilation progress as a successful chip run.

## Commands

From the repository's `trainium-softmax` directory, after setup. TP4 commands require all four cores to be free:

```bash
.venv-training312/bin/python -m unittest discover -s rl -v
.venv-training312/bin/python rl/launch.py probe --tiny --static-experts --tensor-parallel-size 4 --cached-sampling --checkpoint-activations --sample-tokens 2 --out rl/runs/new-tiny-probe
MODEL=$(cat rl/model-path.txt)
.venv-training312/bin/python rl/launch.py probe --model "$MODEL" --static-experts --tensor-parallel-size 4 --checkpoint-activations --out rl/runs/new-20b-probe
.venv-training312/bin/python rl/launch.py discover --model "$MODEL" --tensor-parallel-size 4 --iterations 2 --group-size 4 --max-context 2048 --max-tokens 512 --out rl/runs/new-discovery
```

Hardware commands need access to `/dev/neuron0` outside the restricted Codex sandbox. Fresh runs require unused output directories. Resume the same discovery command with `--resume`; `--iterations` counts additional iterations. Keep the same group count when resuming. Compilation and dense expert routing can be slow; the production run remains unverified until recorded results demonstrate it.

Inspect `iteration-N.json`, `feedback-N-G-C.json`, `archive_step_N.json`, `resume.json`, worker logs/results, and `best.py`/`best.json`. A synthetic +1-advantage probe proves a gradient step; only a discovery iteration with nonconstant measured task rewards proves a task-driven RL step.

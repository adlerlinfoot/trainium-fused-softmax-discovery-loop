# Discover on Trainium2

An NKI fused-softmax evaluator and a focused
[TTT-Discover](https://github.com/test-time-training/discover) port using local
GPT-OSS LoRA training on Trainium2. The search code preserves PUCT sampling,
adaptive entropic advantages and two-phase GPT-OSS completion. Sampling and
training workers exit before kernel evaluation uses the device.

## Verified results

| Verification | Recorded result |
| --- | --- |
| Locally written softmax seed | 72 simulator cases, 72 on-chip cases, 24 benchmarks; 207.8703 µs geometric mean |
| Tiny-model TP4 mechanics | Forward, gradients, optimizer update, updated-policy generation and checkpoint restore passed |
| Partial GPT-OSS-20B TP4 | Original layers 0, 6, 12, 17, 23 passed the same checks on all four ranks; no executed fallback ops |

Partial-model verification preserves original layer indices, weights and
attention settings. It uses eight input tokens, one generated token and one
optimizer update. All 98 extracted checkpoint tensor hashes match the source.
See the [partial-model report](trainium-softmax/rl/PARTIAL_VERIFICATION.md) and
[per-rank evidence](trainium-softmax/evidence/continuation/partial-tp4/summary.json).

**Full GPT-OSS-20B training, task-driven RL, and model-generated kernel speedups
remain unverified.** The seed is a correctness baseline, not a discovered
optimization. No new kernels were generated in the partial-model test.

## Setup and use

Use the [training setup and commands](trainium-softmax/rl/README.md) for the
pinned Python/Neuron environment and checkpoint download. The
[kernel contract and evaluator commands](trainium-softmax/README.md) describe
correctness and timing. Hardware tests need an accessible Trainium device;
TP4 requires four free cores. Fresh clones need local setup and model weights.

Upstream algorithm snapshots and their MIT license are retained in
[discover-reference](discover-reference/README.md); runtime vendor provenance
is in [PROVENANCE.json](trainium-softmax/rl/vendor/PROVENANCE.json).
Weights, virtual environments, compilation caches and training checkpoints
are excluded from Git. Historical session notes are [archived](docs/archive/CONTINUATION.md).

## Proposed upstream contribution

The [integration report](docs/REPORT.md), [PR draft](docs/PR_DESCRIPTION.md)
and [handoff instructions](docs/HANDOFF.md)
prepare an additive contribution to `yahavb/trainium-agent-labs`. The draft
does not constitute a published pull request.

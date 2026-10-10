# Discover on Trainium2

Current continuation: [bounded partial GPT-OSS-20B TP4 hardware verification](trainium-softmax/rl/PARTIAL_VERIFICATION.md)
using original layers 0, 6, 12, 17 and 23. Full-model and discovery experiments
are suspended for this run. Partial results must not be read as full-model
verification or a measured kernel speedup.

Port of [TTT-Discover](https://github.com/test-time-training/discover) for fused
softmax kernel search using `openai/gpt-oss-20b` and tensor parallelism across
four Trainium2 cores (TP4).

The implementation is in `trainium-softmax/`: NKI correctness checks and device
benchmarks, PUCT search, adaptive entropic advantages, GPT-OSS two-phase
completion, and local Neuron LoRA updates. Sampling and training release the
cores before generated kernels are evaluated. Failed correctness checks receive
zero reward. Only measured, correct device kernels can replace the best kernel.

This repository continues existing work found in the pod. Historical evidence
records a correct seed at **207.8703 microseconds** geometric mean across 24
benchmark cases, and tiny-model TP4 tests. Full GPT-OSS-20B TP4 training and
task-driven search are being verified; no model-generated speedup is claimed.

See `trainium-softmax/rl/README.md` for setup and commands and
`trainium-softmax/README.md` for the kernel contract. Upstream source snapshots
used by parity tests are in `discover-reference/`; provenance and the upstream
MIT license are retained. Model weights, compilation caches, and training
checkpoints are local artifacts and excluded from Git.

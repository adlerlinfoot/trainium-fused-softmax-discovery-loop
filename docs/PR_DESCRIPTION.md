# Add Discover softmax evaluation and verified partial GPT-OSS TP4 training

The labs currently provide agent exercises and shared GPT-OSS inference.
This contribution adds a standalone NKI softmax correctness/benchmark evaluator
and a focused Discover port with local Neuron LoRA training, plus reproducible
partial-model TP4 verification using real GPT-OSS-20B checkpoint layers.

The accompanying [integration report](REPORT.md)
explains how this extends the workshop's model → checker → feedback loop,
where the standalone softmax evaluator fits, and why local training requires
checkpoint access rather than a vLLM inference endpoint. It documents the
recorded Qwen3-8B failure in the AMI's vLLM-Neuron 0.21 model integration
(`Qwen3ForCausalLM.from_configs` missing), the working workshop container,
and the distinction between a common vLLM API and hardware-specific backends.
It does not claim Qwen3-8B is inherently unsupported on Trainium2.

The evaluator rejects incorrect candidates before timing, checks 72 simulator
and 72 on-chip cases, and records 24 device-traced benchmarks. Search retains
upstream PUCT sampling, adaptive entropic advantages and GPT-OSS two-phase
completion. Separate sampling/training processes release cores before kernel
evaluation. Checkpoint restore and resume preserve policy fingerprints.

## Validation

The prepared upstream checkout passes all **20 software tests** below; no
hardware experiment was repeated during handoff preparation.

- Seven evaluator tests passed.
- Ten RL tests, one four-process CPU TP4 parity test, and two partial-model
  loading/selection tests passed.
- Tiny TP4 on-chip mechanics passed on all four ranks.
- Partial GPT-OSS-20B TP4 passed using original layers **0, 6, 12, 17, 23**, an
  eight-token input and one generated token: finite nonzero gradients (global
  norm **433.7534582**), one optimizer update, updated-policy generation and
  checkpoint restoration. All ranks agree on the changed adapter fingerprint;
  no fallback ops executed. Original attention settings and 98 checkpoint
  tensor hashes are preserved. Qwen was restored and healthy after testing.
- The locally written softmax seed passed 72 simulator and 72 on-chip cases;
  its recorded geometric mean is **207.8703 µs** across 24 benchmark cases.

Compact per-rank evidence, tensor provenance, setup instructions and the
partial-model report are included. Model weights, environments, training
checkpoints and compiler caches are excluded. Upstream snapshots retain their
MIT license and source provenance.

The isolated environment also installs the evaluator's `httpx` client, so a
fresh setup can run both test suites. Whitespace attributes preserve upstream
snapshot/vendor formatting and the meaningful context spaces in SDK patches.

## Scope and limitations

This is **partial-model verification**, not proof of full GPT-OSS-20B training,
task-driven RL, or a model-generated kernel speedup. The seed is a correctness
baseline. Dense fixed-shape expert routing favors compatibility over sparse
execution efficiency. The NKI adapter and explicit-core SDK patch depend on
the pinned Neuron stack. The optional Qwen lifecycle wrapper is seat-specific.

The contribution adds `trainium-softmax/`, `discover-reference/` and
`docs/discover-trainium/REPORT.md`, merges
artifact exclusions into `.gitignore`, and adds documentation links to the
existing lab README. Existing lab projects and serving configuration are
preserved. Full-model and discovery entry points remain experimental.

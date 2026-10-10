# Trainium fused-softmax discovery loop

The on-chip NKI verifier is in `loop.py`. The Trainium training and Discover search port is in [`rl/README.md`](rl/README.md), with `rl/discover.py` as its entry point. The older `loop.py run` command is a separate inference-only baseline and does not perform RL.

## Verified environment (2026-10-10)

Outside the restricted execution sandbox, `neuron-ls -j` identifies a **trn2.48xlarge / Trainium2**, one accessible device, four physical NeuronCores, LNC=2, and 96 GiB device memory. NKI **0.6.0+31049202112.g85070674**, neuronx-cc **2.27.5334.0+f702b353**, runtime **2.34.10.0-ac18d186d**, driver **2.30.2.0**, PyTorch **2.11.0** are installed. In the sandbox, `/dev/neuron0` is hidden and `neuron-ls` fails; use the actual Trainium host for hardware mode.

At the initial verifier run, the local `http://localhost:8000/v1/models` advertised **Qwen/Qwen3-8B**, not GPT-OSS-20B. `GPTOSS_BASE_URL` is unset. The workshop's `gptoss/README.md` describes GPT-OSS-20B as a separate shared Trainium2 service, but supplies no URL. That initial run performed no GPT-OSS generation. The inference-only implementation checks `/models` and requires `gpt-oss-20b`; it never silently substitutes Qwen.

## Reference inspection

Reference repository: https://github.com/test-time-training/discover, commit `6c40e82dab9d5de7416ac873ad5cd3106084aaed` (selected snapshots are tracked in [`../discover-reference/`](../discover-reference/README.md)).

* `examples/gpu_mode/env.py::GpuModeEnv.get_question` combines task templates with `State.to_prompt`, previous implementation, runtime, and target; demands one Python code block. Supported standalone task selectors are `trimul` and `mla_decode_nvidia`.
* `GpuModeRewardEvaluator.get_reward` checks for `@triton.jit`, then sends code to Modal in leaderboard mode. Failed execution/correctness gets zero reward; accepted ranking latency in microseconds gets `score_scale / score_us` (1500 for TriMul, 5000 for MLA).
* `lib/mla-decode/eval.py::benchmark` performs mandatory correctness/compilation before CUDA-event timing, repeats launches, and synchronizes. Task configuration supplies test/benchmark shapes and ranking aggregation. This CUDA/Modal path is not used here.
* `ttt_discover/tinker_utils/dataset_builder.py::_run_verification` reads reward, correctness, raw score and message. `_create_next_state` stores generated code, signed raw score and stdout; states/experiences feed the sampling buffer and later rollout prompts. `State.to_prompt` renders prior code, value, and observation. The GPU evaluator returns empty stdout: its `msg` should not be mistaken for automatically persisted diagnostic observations.
* `examples/gpu_mode/prompt.py` contains softmax inside MLA attention and a Triton helper, but no standalone fused-softmax task specification. This project defines one explicitly below.

## Task contract

Contiguous 2D `x`, shapes `(1,32)`, `(17,128)`, `(128,512)`, `(257,1024)`; float32 or float16. Same-shape float32 additive mask: **0 keep, -1e30 masked**. Supports no mask (all zeros), random masks, and causal masks (`column <= row`). Each row must retain at least one element. Input logits are finite in `[-80,80]`. No all-masked rows, other shapes/dtypes, noncontiguous inputs, NaN/Inf, or gradients are supported.

PyTorch reference:

```python
output = torch.softmax(x.float() + mask, dim=-1).to(x.dtype)
```

Output must preserve shape/dtype, be finite, and inputs must remain unchanged. Float32 tolerance: atol `2e-5`, rtol `2e-4`; float16: atol `5e-4`, rtol `2e-3`. Fixed seed 314159; 72 correctness cases span all shapes/dtypes/mask modes with normal, extreme, and constant logits. The constant cases check uniform probabilities over retained elements. The reference and data remain in the evaluator, outside candidate code.

## Evaluation and feedback

1. Parse source and enforce one `@nki.jit fused_softmax(x, mask)` function and a restricted NKI import/call policy. Import and evaluate each candidate in a fresh subprocess, with a 900-second deadline and saved log. This policy is not a hardened security sandbox; generated code should run in the dedicated workshop container.
2. Check all 72 cases using the NKI CPU simulator before compilation. Reject wrong shapes, dtypes, nonfinite outputs, numerical mismatches, and input mutation; report first bad coordinate, expected/actual value, mismatch count, and maximum absolute error.
3. In hardware mode, compile each shape/dtype specialization into a saved NEFF, reject input/output aliases, and check all 72 cases **on chip before calling the benchmark**. Compile/runtime errors get zero reward and no latency.
4. For accepted implementations, benchmark all 24 shape/dtype/mask combinations with normal inputs, using **10 warmups and 100 timed iterations**, runtime device-side tracing (not Python wall time). Transfers and compilation are excluded. Record per-case mean/min/max/std latency; aggregate with geometric mean across cases. Speed reward is `1000 / geometric_mean_latency_us`. It uses a fixed scale so rewards are comparable across attempts. Feedback also gives latency and speedup relative to the current best.
5. Next GPT-OSS prompt includes task, best/seed source, previous attempt, and evaluator feedback. Save every request, response, candidate, feedback, compiler artifact and log. `best.py` and `best.json` update only for the lowest-latency fully correct on-chip candidate, including the seed. A new run requires a fresh output directory.

NKI 0.6 removes the older top-level `nki.baremetal` and `nki.benchmark` APIs described in [older AWS documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.baremetal.html). `compile_device` is a narrow adapter to the installed NKI compiler APIs (`compile_kernel_to_nir`, `compile_bir_to_neff`, `CompiledKernel.execute/benchmark`). These internal imports may need updating for a different SDK. The installed simulator accepts `nl.divide`, but this compiler rejects its lowering; the seed uses reciprocal and multiplication instead.

## Commands

From the repository root:

```bash
python -m unittest discover -s trainium-softmax -v
python trainium-softmax/loop.py probe
python trainium-softmax/loop.py run --mode simulate --attempts 0 --out trainium-softmax/runs/new-smoke
```

Simulation never emits a speed reward or claims device timing. For real hardware and the model loop:

```bash
export GPTOSS_BASE_URL='https://YOUR-SHARED-ENDPOINT'
NEURON_RT_NUM_CORES=1 python trainium-softmax/loop.py run \
  --mode hardware --target trn2 --attempts 3 \
  --out trainium-softmax/runs/new-discovery
```

Use an explicit base ending in `/agg/v1`, `/disagg/v1`, or `/v1` to select the endpoint; otherwise `/agg/v1` is appended. TLS verification is enabled; `--insecure` explicitly permits the workshop endpoint's documented hostname mismatch if necessary. HTTP failures, missing model, empty/truncated code answers, and invalid candidates enter feedback for the next attempt. No access keys are required by the workshop service.

`NEURON_RT_NUM_CORES=1` prevents runtime initialization from trying to allocate all cores while the local model occupies others. Do not stop existing model processes. Benchmark results should be remeasured on an idle device before comparing small improvements. Hardware mode requires host/device access, including permission to execute outside the Codex sandbox here.

## Preserved results

The [seed kernel](evidence/seed-hardware/best.py) passed all 72 simulator and
72 on-chip correctness cases and 24 device benchmarks. Its recorded geometric
mean is **207.87033785930112 microseconds**, reward **4.810691175558**.
[Per-case timing evidence](evidence/seed-hardware/best.json) is tracked;
compiler artifacts and earlier local run directories are excluded. This is a
locally written correctness baseline, not a GPT-OSS-generated improvement.

The [partial GPT-OSS-20B TP4 verification](rl/PARTIAL_VERIFICATION.md) passed
on all four ranks using original layers 0/6/12/17/23. Full-model training,
task-driven RL, and model-generated kernel speedups remain unverified.

Seven evaluator tests cover the contract, source rejection, wrong outputs
being rejected before compilation/timing, zero simulator speed reward,
feedback construction and mocked generation/evaluation control flow. Their
[recorded output](evidence/continuation/evaluator-tests.txt) is preserved.
Mocked results do not establish hardware performance. The simulator baseline
[result](evidence/seed-hardware/seed-result.json) is also tracked.

See [training setup](rl/README.md) for the pinned environment, checkpoint
preparation and verification commands. Model weights and caches are local
artifacts; a fresh clone requires setup and download.

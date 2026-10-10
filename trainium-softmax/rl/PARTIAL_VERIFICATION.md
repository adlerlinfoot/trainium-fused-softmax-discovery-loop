# Partial GPT-OSS-20B hardware verification

**Passed on all four Trainium ranks at 2026-10-10 20:28:37 UTC.**
On-chip forward and one-token generation, finite nonzero gradients (global
norm **433.7534582**), one optimizer update, generation from policy version 1,
and checkpoint restoration all completed. All ranks agree on the updated hash
`a8039e06c32eb7dfe4493753f4f9d2fdcecbe222367c1045707b144e6c130bf2`.
Executed fallback ops: none. Input length: eight tokens. Generation length:
one token before and one token after the update. Experiments stopped after
17.32 minutes of hardware testing. Qwen was restored with HTTP health 200,
test workers exited, and final evidence was captured at 20:32:17 UTC, about
25.26 minutes after the deadline clock started. All 98 extracted tensor hashes
match the original checkpoint, and all original configuration fields match.
See `../evidence/continuation/partial-tp4/summary.json` for the cleanup/budget
record and `hardware-session.txt` for the complete session log.

This experiment executes five layers from the real GPT-OSS-20B checkpoint.
It does **not** verify the full model, task-driven RL, or softmax optimization.

Checkpoint revision: `6cee5e81ee83917806bbde320786a8fb61efebee`.
Original depth: 24. Selected zero-based layers: **0, 6, 12, 17, 23**
(0%, 26.1%, 52.2%, 73.9%, 100% of the original depth range).
Selections are nonadjacent. Layers 0/6/12 retain sliding attention with window
128; layers 17/23 retain full attention. Original names, `layer_idx`, weights,
RoPE, expert count and routing settings are preserved. Embedding, final norm,
and LM head are retained. Removing intervening layers changes the model's
function, so generated text is not a full-model quality measurement.

`prepare_partial.py` copies only selected checkpoint tensors, preserving their
names and recording their hashes. `partial_model.py` keeps the original depth
configuration and uses an ordered module dictionary with original indices.
The loader rejects missing, unexpected, or mismatched weights. As in the
existing backend, checkpoint MXFP4 experts are dequantized to BF16, frozen base
weights are sharded with TP4, and global-rank-32 LoRA trains attention q/k/v/o.

The gate uses an eight-token input and one generated token. Required results:
on-chip forward, finite nonzero adapter gradients, one AdamW update with a
changed global adapter hash, generation at policy version 1, and restoration
of the saved adapter/optimizer/RNG checkpoint with its expected global hash.
Per-rank results and events are under `../evidence/continuation/partial-tp4/`;
tensor provenance is `../evidence/continuation/partial-provenance.json`.

Budget for this run started at **2026-10-10 20:07:02 UTC**. Preparation was
completed and pushed at approximately 20:11 UTC, within ten minutes. Testing
is capped at 45 minutes, experiments stop no later than **21:02:02 UTC**, and
cleanup, Qwen restoration, evidence and final GitHub push must finish by
**21:22:02 UTC**. `hardware_session.py` enforces an absolute interrupt and
restores the captured Qwen launch command/environment. During testing it
copies compact per-rank evidence and pushes GitHub checkpoints every four
minutes. Qwen restoration is checked through its HTTP health endpoint.

For a new partial extraction, from the repository root:

```bash
trainium-softmax/.venv-training312/bin/python trainium-softmax/rl/prepare_partial.py \
  --source /path/to/pinned/checkpoint --out trainium-softmax/.partial-checkpoint
trainium-softmax/.venv-training312/bin/python trainium-softmax/rl/launch.py probe \
  --model trainium-softmax/.partial-checkpoint --partial --static-experts \
  --tensor-parallel-size 4 --checkpoint-activations --sample-tokens 1 \
  --out trainium-softmax/rl/runs/new-partial-tp4
```

The hardware command needs four free cores. On this seat, use the hardware
session wrapper with the currently verified Qwen server PID, absolute stop
time, and evidence paths. Push source changes before pausing the service.
Use fresh output directories; preserve failed results without relabeling them
as successful execution.

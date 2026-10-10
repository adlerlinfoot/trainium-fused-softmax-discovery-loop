# Trainium TP4 verification continuation

## Superseding bounded partial-model request

Start: 2026-10-10 20:07:02 UTC. Preparation ends by 20:17:02 UTC.
Experiments stop by 21:02:02 UTC; hard deadline including Qwen restoration,
evidence and final push is 21:22:02 UTC. No full-model or discovery run.
Select original real-checkpoint layers at approximately 0/25/50/75/100%
depth, nonadjacent, preserving indices, weights and attention settings.
TP4, short input, one generated token; verify forward, finite nonzero
gradients, one update, updated-policy generation and checkpoint restoration.
Results must be labeled partial-model verification. Stop the superseded
full-model process and preserve its reached stage without claiming success.

20:11 UTC: preparation completed. Selected original layers [0, 6, 12, 17, 23],
retaining full-depth config and original module names/layer_idx. Extracted
98 exact tensors (4.5 GiB checkpoint) with per-tensor source hashes; original
sliding attention at 0/6/12 and full attention at 17/23. Two partial-loading
tests and ten RL tests pass. Superseded full run stopped during initialization;
its workers exited and Qwen became healthy again. The partial hardware runner
has a 2700-second testing limit, absolute minute-55 interrupt, process-tree
cleanup, service restoration, and automatic GitHub pushes every 240 seconds.

20:28:37 UTC: partial-model hardware verification passed on all four ranks.
Selected original checkpoint layers [0, 6, 12, 17, 23], short eight-token input,
one generated token before and after one update. Finite nonzero global gradient
norm 433.75345819947074; all ranks agree on updated hash
`a8039e06c32eb7dfe4493753f4f9d2fdcecbe222367c1045707b144e6c130bf2`.
Checkpoint restoration matches this hash; no executed fallback ops. Results,
events and checkpoint policy metadata are captured per rank. Testing ended
well before the 45-minute and minute-55 caps; Qwen is being restored. No full
model or discovery experiment was attempted after the scope change.

20:32:17 UTC: cleanup and evidence capture completed. Qwen health HTTP 200;
test workers/compilers exited. All 98 extracted tensors match the recorded
original checkpoint hashes and original config fields are unchanged. Hardware
testing took 17.32 minutes; elapsed budget before final push was 25.26 minutes.
Final artifacts include per-rank result/events/checkpoint policy metadata,
complete session log and `partial-tp4/summary.json`. Final GitHub checkpoint
contains this cleanup record; no further experiments are planned for this task.

Target: verify real GPT-OSS-20B TP4 forward, backward, optimizer update,
updated-policy generation and checkpoint restore; then execute measured
softmax discovery with nonconstant task rewards.

2026-10-10: cloned at `aeb63fe`. Trainium2 is accessible outside the
execution sandbox: four cores, 96 GiB total, LNC2. Qwen occupies two cores.
The cloned checkout has no training environment, model weights or prior
compilation cache. Recreate these before attempting hardware verification.

Checkpoint policy: push before risky actions and at least every five minutes
of active work. Preserve source and compact evidence, exclude weights/caches.
Do not claim full-model or task RL success from tiny tests or compilation.

Initial checks: seven evaluator tests pass. The pinned GPT-OSS checkpoint
has downloaded to the checkout-local Hugging Face cache. Pinned training
dependencies are being installed. Qwen was healthy before hardware tests.
`rl/hardware_session.py` captures its existing command and environment and
restores it after hardware subprocess exit, including failures/interruption.

19:56 UTC: isolated pinned training setup completed, including the recorded
SDK explicit-core-selection patch. CPU RL and TP4 parity suite started.

19:57 UTC: all eleven training tests pass outside the sandbox, including
four-process TP4 forward/gradient/two-update/checkpoint parity. The sandbox
run passed ten RL tests but could not create Gloo's local rendezvous.
Compact test logs are in `trainium-softmax/evidence/continuation/`.
Next gate: tiny TP4 on chip, with Qwen restored by the hardware session wrapper.

20:00 UTC: tiny TP4 on-chip gate passed on all four ranks. Global gradient
norm 2.28780541, finite nonzero gradients, optimizer parameter change,
updated-policy generation, checkpoint restore, and no executed fallback ops.
Global updated hash:
`7aaba860e39479504098f4f13653f137e085e8f7763404ea780d1c6c77175e7a`.
Per-rank event/result JSON is preserved under evidence/continuation/tiny-tp4.
This is a tiny-model compatibility result; full GPT-OSS-20B remains unverified.

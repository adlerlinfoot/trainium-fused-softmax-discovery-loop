# Full-model TP4 continuation

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

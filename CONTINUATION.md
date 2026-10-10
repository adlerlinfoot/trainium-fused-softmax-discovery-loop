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

#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
uv --cache-dir "$TASK_ROOT/.uv-cache" python install 3.12.14 --install-dir "$TASK_ROOT/.python"
uv --cache-dir "$TASK_ROOT/.uv-cache" venv --allow-existing --seed \
  --python "$TASK_ROOT/.python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" "$TASK_ROOT/.venv-training312"
"$TASK_ROOT/.venv-training312/bin/python" -m pip install \
  -r "$TASK_ROOT/rl/requirements-training.txt" --extra-index-url https://pip.repos.neuron.amazonaws.com \
  --cache-dir "$TASK_ROOT/.pip-cache"
"$TASK_ROOT/.venv-training312/bin/python" "$TASK_ROOT/rl/apply_sdk_fix.py"

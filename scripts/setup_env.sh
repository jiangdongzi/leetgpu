#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
echo "/opt/ml-stack/venv/lib/python3.10/site-packages" > .venv/lib/python3.10/site-packages/ml_stack_bridge.pth
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt

python - <<'PY'
import torch
import triton
print(f"torch={torch.__version__}")
print(f"triton={triton.__version__}")
PY

echo "Environment ready. Activate with: source .venv/bin/activate"

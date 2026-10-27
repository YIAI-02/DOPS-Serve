#!/usr/bin/env bash
set -euo pipefail
module load pytorch/24.12-py3
singularity exec "$PYTORCH_SIF" python3 - <<'PY'
import platform
import sys
import torch
print("python:", sys.version)
print("torch:", torch.__version__)
print("platform:", platform.platform())
PY

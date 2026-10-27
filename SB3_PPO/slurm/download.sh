#!/usr/bin/env bash
# 只在 data 节点下载 wheel；不安装、不运行训练。
set -euo pipefail
cd "$(dirname "$0")/.."
module load pytorch/24.12-py3
mkdir -p wheelhouse
singularity exec "$PYTORCH_SIF" env PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
    python3 -m pip download --only-binary=:all: --no-deps \
    --index-url https://download.pytorch.org/whl/cpu --dest wheelhouse torch==2.14.1
singularity exec "$PYTORCH_SIF" env PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
    python3 -m pip download --only-binary=:all: \
    --find-links wheelhouse --dest wheelhouse -r requirements.txt

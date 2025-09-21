#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
VENV=${VENV:-.venv}
NO_TEST=${NO_TEST:-0}

# 1) 创建并激活虚拟环境
$PY -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# 2) 升级 pip 并安装（可编辑 + 开发依赖）
python -m pip install -U pip wheel
pip install -e .[dev]

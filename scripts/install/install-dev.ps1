param(
  [string]$Py = "python",
  [string]$Venv = ".venv",
  [switch]$NoTest = $false
)

$ErrorActionPreference = "Stop"

# 1) 创建并激活虚拟环境
& $Py -m venv $Venv
& "$Venv/Scripts/Activate.ps1"

# 2) 升级 pip 并安装（可编辑 + 开发依赖）
python -m pip install -U pip wheel
pip install -e .[dev]

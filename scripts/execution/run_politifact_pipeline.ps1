<#
.SYNOPSIS
  Run the PolitiFact pipeline: prepare_data → train_trainer
.DESCRIPTION
  Invoke two KAN pipelines:
  1. kan.pipelines.prepare_data
  2. kan.pipelines.train_trainer

  By default, use configs/data/politifact.yaml and configs/train/politifact.yaml.
  External configs can also be provided.
#>

param(
    [string[]]$PrepareConfigs = @("configs/data/politifact.yaml"),
    [string[]]$TrainConfigs   = @("configs/train/politifact_5fold.yaml"),
    [string[]]$Overrides      = @()
)

# ---------------------------------------------------------
# Helper for colored output
# ---------------------------------------------------------
function Info($msg)  { Write-Host "[INFO]  $msg"  -ForegroundColor Cyan }
function Warn($msg)  { Write-Host "[WARN]  $msg"  -ForegroundColor Yellow }
function ErrorMsg($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# ---------------------------------------------------------
# Automatically locate project root
# ---------------------------------------------------------
$ScriptDir = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $ScriptDir       # assume scripts/execution/
Push-Location $Root

Info "KAN Project Root = $Root"
Info "Python = $(Get-Command python)"

# ---------------------------------------------------------
# 1) Data preparation: prepare_data
# ---------------------------------------------------------
Info "==== Step 1: Prepare Data ===="

$prepareCmd = @(
    "-m", "kan.pipelines.prepare_data",
    "-c"
) + $PrepareConfigs + @(
    "-o"
) + $Overrides

Info "Running prepare_data: python $($prepareCmd -join ' ')"

python $prepareCmd
if ($LASTEXITCODE -ne 0) {
    ErrorMsg "prepare_data FAILED (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Info "prepare_data SUCCESS"

# ---------------------------------------------------------
# 2) Training: train_trainer
# ---------------------------------------------------------
Info "==== Step 2: Train Model ===="

$trainCmd = @(
    "-m", "kan.pipelines.train_trainer",
    "-c"
) + $TrainConfigs + @(
    "-o"
) + $Overrides

Info "Running train_trainer: python $($trainCmd -join ' ')"

python $trainCmd
if ($LASTEXITCODE -ne 0) {
    ErrorMsg "train_trainer FAILED (exit code $LASTEXITCODE)"
    exit $LASTEXITCODE
}

Info "train_trainer SUCCESS"
Info "Full pipeline completed: PolitiFact data preparation + model training"

Pop-Location

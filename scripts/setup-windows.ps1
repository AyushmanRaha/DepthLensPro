param(
  [ValidateSet('small','compare','all')] [string]$Models = 'all',
  [switch]$SkipModels,
  [switch]$Offline,
  [switch]$Force,
  [int]$TimeoutSeconds = 900,
  [int]$Retries = 1,
  [switch]$WithoutOnnx,
  [switch]$WithOnnx,
  [string]$OnnxModels = 'midas_small'
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "[DepthLens] Windows setup: dependencies"
$doctorArgs = @('--enforce-arch')
if ($WithoutOnnx) { $doctorArgs += '--without-onnx' }
if ($WithOnnx) { $doctorArgs += @('--with-onnx','--onnx-models',$OnnxModels) }
$candidates = @(@('py','-3.12'), @('py','-3.11'), @('py','-3.10'), @('python'))
$python = $null
foreach ($candidate in $candidates) { try { & $candidate[0] @($candidate[1..($candidate.Length-1)] | Where-Object { $_ }) -c "import sys" *> $null; $python = $candidate; break } catch {} }
if (-not $python) { throw "No Python launcher available. Install Python 3.10-3.12." }
& $python[0] @($python[1..($python.Length-1)] | Where-Object { $_ }) scripts/doctor.py @doctorArgs
if ($SkipModels) { Write-Host "[DepthLens] Skipping model installation; Compare requires MiDaS_small, DPT_Hybrid, DPT_Large before use."; exit 0 }
$venvPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { $venvPython = Join-Path (Get-Location) 'venv\Scripts\python.exe' }
if (-not (Test-Path $venvPython)) { $venvPython = $python[0] }
$env:TORCH_HOME = if ($env:TORCH_HOME) { $env:TORCH_HOME } else { Join-Path (Get-Location) 'models\torch-cache' }
Write-Host "[DepthLens] Windows setup: installing PyTorch MiDaS models=$Models TORCH_HOME=$env:TORCH_HOME"
$args = @('scripts/manage_model_assets.py','install','--models',$Models,'--timeout-seconds',[string]$TimeoutSeconds,'--retries',[string]$Retries)
if ($Offline) { $args += '--offline' }
if ($Force) { $args += '--force' }
& $venvPython -u @args

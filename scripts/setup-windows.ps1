$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "DepthLens setup requires Node.js on PATH as node. Install Node.js, then re-run: npm run setup:win" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "DepthLens setup requires npm on PATH. Install Node.js/npm, then re-run: npm run setup:win" }
$candidates = @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python"))
$probeFailures = @()
foreach ($candidate in $candidates) {
  $exe = $candidate[0]; $extra = @(); if ($candidate.Length -gt 1) { $extra = $candidate[1..($candidate.Length - 1)] }
  try { & $exe @extra -c "import sys; raise SystemExit(0)" *> $null } catch { $probeFailures += "$exe $($extra -join ' '): $($_.Exception.Message)"; continue }
  Write-Host "[DepthLens] Using Python launcher: $exe $extra"
  & $exe @extra scripts/doctor.py --enforce-arch @args
  exit $LASTEXITCODE
}
throw "No Python launcher was available to run scripts/doctor.py. Install Python 3.10-3.12 from python.org. Probe failures: $($probeFailures -join '; ')"

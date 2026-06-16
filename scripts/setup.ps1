$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Write-Host "DepthLens Pro setup - installing dependencies and required model assets."
py -3 scripts/doctor.py --enforce-arch @args

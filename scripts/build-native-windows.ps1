$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
./scripts/setup-windows.ps1
Push-Location electron-app
npm run verify:resources -- --mode native --onnx optional
npm run build:win:arm64
Pop-Location

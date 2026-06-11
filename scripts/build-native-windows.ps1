$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
./scripts/setup-windows.ps1
Push-Location electron-app
npm run clean:dist
Pop-Location
Write-Host "Cleaned previous dist/ output."
Push-Location electron-app
npm run verify:resources:native
npm run build:win:arm64:raw
npm run verify:packaged:win
Pop-Location

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (!(Test-Path setup-manifest.json)) { throw "Setup manifest missing. Run .\scripts\setup.ps1 first." }
cd electron-app
npm run verify:resources:native
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq "Arm64") { npm run build:win:arm64 } else { npm run build:win:x64 }

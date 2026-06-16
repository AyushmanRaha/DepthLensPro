$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$app = Get-ChildItem -Path electron-app\dist -Recurse -Filter "DepthLens Pro.exe" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
if (!$app) { throw "No built app found. Run .\scripts\build.ps1" }
Write-Host "Launching: $($app.FullName)"
Start-Process $app.FullName

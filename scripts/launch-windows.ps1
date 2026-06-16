$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Arch = if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'arm64' } else { 'x64' }
$App = "electron-app\dist\win-$Arch-unpacked\DepthLens Pro.exe"
Write-Host "[DepthLens] Expected Windows app path: $App"
if (!(Test-Path $App)) { throw "Packaged app not found. Build first: npm run build:win:$Arch" }
Start-Process $App

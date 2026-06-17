param([ValidateSet('arm64','x64')] [string]$Arch = $(if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'arm64' } else { 'x64' }), [switch]$WithOnnx, [switch]$WithoutOnnx, [string]$OnnxModels = 'midas_small', [switch]$AutoSetup)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$OnnxVerifyMode = 'optional'; if ($WithOnnx) { $OnnxVerifyMode='require-all'; $OnnxModels='all' }; if ($WithoutOnnx) { $OnnxVerifyMode='off' }
Write-Host "[DepthLens] Step 3 Build: platform=Windows arch=$Arch onnx=$OnnxVerifyMode models=$OnnxModels resource_root=$(Get-Location)"
if ($AutoSetup) { Write-Host "[DepthLens] -AutoSetup requested; running setup explicitly."; if ($OnnxVerifyMode -eq 'require-all') { & ./scripts/setup-windows.ps1 --with-onnx --onnx-models all } else { & ./scripts/setup-windows.ps1 --without-onnx } }
Push-Location electron-app
try {
  Write-Host "[DepthLens] Verifying repo resources before packaging (no downloads)."
  node scripts/verify-resources.js --root-kind repo --mode native --torch-cache required --onnx $OnnxVerifyMode --models $OnnxModels ..
  if ($LASTEXITCODE -ne 0) { throw "Run npm run setup:win$(if ($OnnxVerifyMode -eq 'require-all') { ':onnx' }) first." }
  npm run clean:dist
  Write-Host "[DepthLens] Packaging Windows $Arch..."; npm run "build:win:$Arch`:raw"
  Write-Host "[DepthLens] Verifying packaged Windows $Arch resources..."; node scripts/verify-packaged-resources.js --platform win32 --arch $Arch --mode native --torch-cache required --onnx $OnnxVerifyMode --models $OnnxModels
  Write-Host "[DepthLens] SUCCESS Windows $Arch build. Outputs under electron-app/dist"
} finally { Pop-Location }

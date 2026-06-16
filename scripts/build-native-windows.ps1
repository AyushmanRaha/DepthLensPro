param(
  [ValidateSet('arm64','x64')] [string]$Arch = $(if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'arm64' } else { 'x64' }),
  [switch]$WithOnnx,
  [switch]$WithoutOnnx,
  [string]$OnnxModels = 'midas_small'
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$OnnxVerifyMode = 'optional'
$SetupArgs = @()
if ($WithOnnx) { $OnnxVerifyMode = 'require-all'; $OnnxModels = 'all'; $SetupArgs += @('--with-onnx','--onnx-models','all') }
if ($WithoutOnnx) { $OnnxVerifyMode = 'off'; $SetupArgs += @('--without-onnx') }
if ($Arch -ne 'arm64' -and $Arch -ne 'x64') { throw "Unsupported Windows architecture: $Arch. Supported: arm64, x64." }
Write-Host "[DepthLens] Windows native build starting for arch=$Arch onnx=$OnnxVerifyMode models=$OnnxModels"
& ./scripts/setup-windows.ps1 @SetupArgs
Push-Location electron-app
npm run clean:dist
Write-Host "[DepthLens] Verifying repo resources before packaging..."
node scripts/verify-resources.js --root-kind repo --mode native --onnx $OnnxVerifyMode --models $OnnxModels ..
Write-Host "[DepthLens] Packaging Windows $Arch..."
npm run "build:win:$Arch`:raw"
Write-Host "[DepthLens] Verifying packaged Windows $Arch resources..."
node scripts/verify-packaged-resources.js --platform win32 --arch $Arch --mode native --onnx $OnnxVerifyMode --models $OnnxModels
Pop-Location
Write-Host "[DepthLens] Windows $Arch native build complete."

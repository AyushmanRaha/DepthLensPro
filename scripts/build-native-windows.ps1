$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$OnnxVerifyMode = "optional"
$OnnxModels = "midas_small"
$SetupArgs = @($args)
for ($i = 0; $i -lt $args.Count; $i++) {
  switch ($args[$i]) {
    "--with-onnx" { $OnnxVerifyMode = "required" }
    "--without-onnx" { $OnnxVerifyMode = "off" }
    "--onnx-models" {
      if ($i + 1 -lt $args.Count) {
        $i++
        $OnnxModels = [string]$args[$i]
        if ($OnnxVerifyMode -ne "off") {
          if ($OnnxModels -eq "all") { $OnnxVerifyMode = "require-all" }
          else { $OnnxVerifyMode = "required" }
        }
      }
    }
    "--onnx-strict" { if ($OnnxVerifyMode -ne "off") { $OnnxVerifyMode = "require-all" } }
  }
}
& ./scripts/setup-windows.ps1 @SetupArgs
Push-Location electron-app
npm run clean:dist
Pop-Location
Write-Host "Cleaned previous dist/ output."
Push-Location electron-app
node scripts/verify-resources.js --root-kind repo --mode native --onnx $OnnxVerifyMode --models $OnnxModels ..
npm run build:win:arm64:raw
node scripts/verify-packaged-resources.js --platform win32 --arch arm64 --mode native --onnx $OnnxVerifyMode --models $OnnxModels
Pop-Location
if ($OnnxVerifyMode -eq "off") { Write-Host "ONNX was intentionally skipped for this Windows ARM package." }

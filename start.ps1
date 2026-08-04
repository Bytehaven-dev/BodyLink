$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvPythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"

$poseModel = Join-Path $PSScriptRoot "models\rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7.onnx"
$detectorModel = Join-Path $PSScriptRoot "models\yolox-m-humanart.onnx"
$needsInstall = -not (Test-Path -LiteralPath $venvPython) -or
    -not (Test-Path -LiteralPath $poseModel) -or
    -not (Test-Path -LiteralPath $detectorModel)

if (-not $needsInstall) {
    & $venvPython -c "import onnxruntime, openvr, rtmlib" 2>$null
    $needsInstall = $LASTEXITCODE -ne 0
}

if ($needsInstall) {
    Write-Host "[BodyLink] First run: installing dependencies..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "install.ps1")
}

Start-Process -FilePath $venvPythonw -ArgumentList "main.py" -WorkingDirectory $PSScriptRoot

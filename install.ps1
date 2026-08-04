$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "[BodyLink] Creating Python environment..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment." }
}

Write-Host "[BodyLink] Installing dependencies..." -ForegroundColor Cyan

$cpuOrtInstalled = (& $venvPython -c "import importlib.metadata as m; print('1' if any(d.metadata['Name'].lower() == 'onnxruntime' for d in m.distributions()) else '0')") -eq "1"
if ($cpuOrtInstalled) {
    Write-Host "[BodyLink] Removing the CPU-only ONNX Runtime package..." -ForegroundColor Cyan
    & $venvPython -m pip uninstall --yes onnxruntime onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) { throw "Failed to remove the CPU-only ONNX Runtime package." }
}

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install BodyLink dependencies." }
& $venvPython -m pip install --no-deps "rtmlib==0.0.16"
if ($LASTEXITCODE -ne 0) { throw "Failed to install rtmlib." }

Write-Host "[BodyLink] Downloading RTMW3D, person detector and face models..." -ForegroundColor Cyan
& $venvPython scripts\fetch_model.py
if ($LASTEXITCODE -ne 0) { throw "Failed to download or verify BodyLink models." }

Write-Host "[BodyLink] Verifying ONNX Runtime CUDA sessions..." -ForegroundColor Cyan
& $venvPython scripts\check_runtime.py
if ($LASTEXITCODE -ne 0) { throw "ONNX Runtime CUDA verification failed." }

Write-Host "[BodyLink] Installation complete. Run start.bat." -ForegroundColor Green

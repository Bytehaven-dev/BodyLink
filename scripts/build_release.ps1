param(
    [switch]$SkipTests,
    [switch]$SkipToolInstall,
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "BodyLink virtual environment not found. Run install.ps1 first."
}

$version = (Get-Content -Raw -LiteralPath (Join-Path $projectRoot "VERSION")).Trim()
$moduleVersion = (& $python -c "from bodylink import __version__; print(__version__)").Trim()
if ($LASTEXITCODE -ne 0 -or $moduleVersion -ne $version) {
    throw "VERSION ($version) does not match bodylink.__version__ ($moduleVersion)."
}

$versionInfo = Get-Content -Raw -LiteralPath (Join-Path $projectRoot "packaging\version_info.txt")
$installerScript = Get-Content -Raw -LiteralPath (Join-Path $projectRoot "packaging\BodyLink.iss")
if (-not $versionInfo.Contains("u'$version'") -or -not $installerScript.Contains("#define MyAppVersion `"$version`"")) {
    throw "Packaging metadata does not match VERSION $version."
}

if (-not $SkipTests) {
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "BodyLink tests failed." }
}

& $python scripts\fetch_model.py
if ($LASTEXITCODE -ne 0) { throw "Model verification failed." }

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "release") | Out-Null
$modelPack = Join-Path $projectRoot "release\BodyLink-Models-v$version.zip"
& $python scripts\build_model_pack.py --version $version --output $modelPack
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $modelPack)) {
    throw "Failed to build the BodyLink model pack."
}
$modelPackHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPack).Hash.ToLowerInvariant()
$modelPackSize = (Get-Item -LiteralPath $modelPack).Length

if (-not $SkipToolInstall) {
    & $python -m pip install --disable-pip-version-check -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw "Failed to install build dependencies." }
}

& $python scripts\generate_icon.py build\bodylink.ico
if ($LASTEXITCODE -ne 0) { throw "Failed to generate the BodyLink icon." }

& $python -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\bodylink.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$appExe = Join-Path $projectRoot "dist\BodyLink\BodyLink.exe"
if (-not (Test-Path -LiteralPath $appExe)) {
    throw "Frozen BodyLink executable was not created."
}

$runtimeReport = Join-Path $projectRoot "build\frozen-runtime-report.json"
Remove-Item -LiteralPath $runtimeReport -Force -ErrorAction SilentlyContinue
$previousModelDirectory = $env:BODYLINK_MODEL_DIR
$env:BODYLINK_MODEL_DIR = Join-Path $projectRoot "models"
try {
    $runtimeProcess = Start-Process -FilePath $appExe -ArgumentList @("--runtime-report", $runtimeReport) -PassThru -Wait
}
finally {
    if ($null -eq $previousModelDirectory) {
        Remove-Item Env:BODYLINK_MODEL_DIR -ErrorAction SilentlyContinue
    }
    else {
        $env:BODYLINK_MODEL_DIR = $previousModelDirectory
    }
}
if (-not (Test-Path -LiteralPath $runtimeReport)) {
    throw "Frozen BodyLink runtime check did not complete."
}
$runtime = Get-Content -Raw -LiteralPath $runtimeReport | ConvertFrom-Json
if ($runtimeProcess.ExitCode -ne 0 -or $runtime.status -ne "ok" -or $runtime.body.provider -ne "CUDAExecutionProvider") {
    throw "Frozen BodyLink runtime check failed: $($runtime.error)"
}
Write-Host "Frozen runtime ready: $($runtime.body.provider), MediaPipe face session ready." -ForegroundColor Cyan

if ($SkipInstaller) {
    Write-Host "BodyLink application directory ready: $appExe" -ForegroundColor Green
    exit 0
}

$innoCandidates = @(
    (Join-Path ${env:LOCALAPPDATA} "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path ${env:ProgramFiles} "Inno Setup 6\ISCC.exe")
)
$iscc = $innoCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $iscc) {
    $command = Get-Command iscc.exe -ErrorAction SilentlyContinue
    if ($command) { $iscc = $command.Source }
}
if (-not $iscc) {
    throw "Inno Setup 6 not found. Install it with: winget install --id JRSoftware.InnoSetup -e"
}

& $iscc "/DMyAppVersion=$version" "/DModelPackSHA256=$modelPackHash" "/DModelPackSize=$modelPackSize" packaging\BodyLink.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup build failed." }

$installer = Join-Path $projectRoot "release\BodyLink-Setup-v$version.exe"
if (-not (Test-Path -LiteralPath $installer)) {
    throw "Installer was not created: $installer"
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
$checksum = Join-Path $projectRoot "release\BodyLink-Setup-v$version.exe.sha256"
Set-Content -LiteralPath $checksum -Encoding ascii -NoNewline -Value "$hash  BodyLink-Setup-v$version.exe`n"
$modelChecksum = Join-Path $projectRoot "release\BodyLink-Models-v$version.zip.sha256"
Set-Content -LiteralPath $modelChecksum -Encoding ascii -NoNewline -Value "$modelPackHash  BodyLink-Models-v$version.zip`n"
$checksums = Join-Path $projectRoot "release\SHA256SUMS.txt"
Set-Content -LiteralPath $checksums -Encoding ascii -Value @(
    "$hash  BodyLink-Setup-v$version.exe"
    "$modelPackHash  BodyLink-Models-v$version.zip"
)

$sizeMb = [math]::Round((Get-Item -LiteralPath $installer).Length / 1MB, 1)
$modelSizeMb = [math]::Round($modelPackSize / 1MB, 1)
Write-Host "BodyLink installer ready: $installer ($sizeMb MB)" -ForegroundColor Green
Write-Host "BodyLink model pack ready: $modelPack ($modelSizeMb MB)" -ForegroundColor Green
Write-Host "SHA256: $hash" -ForegroundColor Green

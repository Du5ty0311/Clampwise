# Build the standalone Windows program: dist\Clampwise\Clampwise.exe and dist\Clampwise-windows.zip.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 [-SkipTests]
#
# Tests run in envs\synaptoscope (the development env). The build runs in
# envs\clampwise-build, made from packaging\requirements-build.txt with PyPI wheels
# only: conda's PyQt5 keeps its Qt DLLs where PyInstaller cannot find them, and
# conda-forge's scipy crashed a frozen build on this machine.

param([switch]$SkipTests)
$ErrorActionPreference = "Stop"

# PyInstaller empties dist\ before it finds a locked file: building while the app
# is open would leave the open copy without its data files.
$running = Get-Process Clampwise -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "==> Clampwise is running (pid $($running.Id -join ', ')). Close it first." -ForegroundColor Yellow
    exit 1
}

# Paths follow the Windows user; set CLAMPWISE_CONDA or CLAMPWISE_DEMOS to override them on another PC.
$Conda = if ($env:CLAMPWISE_CONDA) { $env:CLAMPWISE_CONDA } else { Join-Path $env:USERPROFILE "miniconda3" }
$DevPython = Join-Path $Conda "envs\synaptoscope\python.exe"
$BuildEnv = Join-Path $Conda "envs\clampwise-build"
$BuildPython = Join-Path $BuildEnv "python.exe"
if (-not $env:CLAMPWISE_DEMOS) { $env:CLAMPWISE_DEMOS = Join-Path $env:USERPROFILE "synaptoscope-data\neuroexpress_tests" }
if (-not (Test-Path $env:CLAMPWISE_DEMOS)) { throw "The demo recordings are not at $env:CLAMPWISE_DEMOS (copy synaptoscope-data\neuroexpress_tests there)." }
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project

if (-not (Test-Path $BuildPython)) {
    Write-Host "==> Making the build env" -ForegroundColor Cyan
    & (Join-Path $Conda "Scripts\conda.exe") create -y -p $BuildEnv python=3.12 pip
    & $BuildPython -m pip install --only-binary=:all: -r packaging\requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not make the build env." }
}

# Conda's python keeps DLLs it needs itself (libffi, ...) in Library\bin;
# PyInstaller finds dependencies on PATH, which calling python.exe by path leaves out.
$env:PATH = (@($BuildEnv, "$BuildEnv\Library\bin", "$BuildEnv\DLLs", "$BuildEnv\Scripts") -join ";") + ";$env:PATH"

if (-not $SkipTests) {
    Write-Host "==> Tests" -ForegroundColor Cyan
    $env:QT_QPA_PLATFORM = "offscreen"
    $env:QT_QPA_FONTDIR = "C:\Windows\Fonts"
    & $DevPython -m pytest -q -p no:cacheprovider tests
    if ($LASTEXITCODE -ne 0) { throw "Tests failed: not building." }
    Remove-Item Env:\QT_QPA_PLATFORM, Env:\QT_QPA_FONTDIR
}

Write-Host "==> PyInstaller" -ForegroundColor Cyan
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
& $BuildPython -m PyInstaller --noconfirm --clean --log-level WARN --distpath dist --workpath build packaging\clampwise.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
$Exe = Join-Path $Project "dist\Clampwise\Clampwise.exe"

Write-Host "==> The 32-bit Python the Digidata's library needs (python.org's embeddable package, checked)" -ForegroundColor Cyan
# pCLAMP's AxDD1400.dll is 32-bit: recording runs it from this Python, shipped as dist\Clampwise\py32.
$Py32Url = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-win32.zip"
$Py32Sha = "084B9EB24CB848605C895D05B738FBC2572EFC8B4C18C415A824065864A2B853"   # its MD5, 93beb904..., is python.org's
$Py32Zip = Join-Path $env:LOCALAPPDATA "Clampwise\python-3.12.10-embed-win32.zip"
if (-not (Test-Path $Py32Zip)) {
    New-Item -ItemType Directory -Force (Split-Path $Py32Zip) | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing $Py32Url -OutFile $Py32Zip
}
if ((Get-FileHash $Py32Zip -Algorithm SHA256).Hash -ne $Py32Sha) { throw "$Py32Zip is not python.org's package (checksum)." }
Expand-Archive $Py32Zip -DestinationPath (Join-Path $Project "dist\Clampwise\py32") -Force
# the licence (GPL-3.0) beside the program, as the GPL asks
Copy-Item (Join-Path $Project "LICENSE") (Join-Path $Project "dist\Clampwise\LICENSE.txt")

Write-Host "==> Self-test of the built program" -ForegroundColor Cyan
# Windowed: no console, so the report is read from the file it writes.
$proc = Start-Process -FilePath $Exe -ArgumentList "--self-test" -Wait -PassThru
Get-Content (Join-Path $env:LOCALAPPDATA "Clampwise\self-test.txt")
if ($proc.ExitCode -ne 0) { throw "The built program failed its self-test (exit $($proc.ExitCode))." }

Write-Host "==> Start time (to the window on screen)" -ForegroundColor Cyan
$watch = [Diagnostics.Stopwatch]::StartNew()
$app = Start-Process -FilePath $Exe -PassThru
while ($app.MainWindowHandle -eq 0 -and $watch.Elapsed.TotalSeconds -lt 60 -and -not $app.HasExited) {
    Start-Sleep -Milliseconds 100
    $app.Refresh()
}
$seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1)
if ($app.MainWindowHandle -eq 0) { throw "No window within $seconds s." }
$app.CloseMainWindow() | Out-Null
$app.WaitForExit(10000) | Out-Null

Write-Host "==> Zip" -ForegroundColor Cyan
$Zip = Join-Path $Project "dist\Clampwise-windows.zip"
& (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $Zip -C (Join-Path $Project "dist") "Clampwise"
if ($LASTEXITCODE -ne 0) { throw "Zipping failed." }

$folderMB = [math]::Round((Get-ChildItem dist\Clampwise -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
$zipMB = [math]::Round((Get-Item $Zip).Length / 1MB)
Write-Host "==> Done: dist\Clampwise ($folderMB MB), $Zip ($zipMB MB), window in $seconds s" -ForegroundColor Green

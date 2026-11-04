# build.ps1 - Build Water Network Modeler Windows executable
# Usage: .\build.ps1   (run from D:\Projects)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root

# Use the project venv Python explicitly
$python = Join-Path $root ".venv\Scripts\python.exe"
$pyinstaller = Join-Path $root ".venv\Scripts\pyinstaller.exe"

Write-Host "=== Water Network Modeler PyInstaller Build ===" -ForegroundColor Cyan
Write-Host "Python: $python"

# --- 1. Clean ---
Write-Host "[1/4] Cleaning build/ and dist/" -ForegroundColor Yellow
foreach ($d in @("build", "dist")) {
    if (Test-Path $d) {
        Remove-Item $d -Recurse -Force
        Write-Host "  Removed $d/"
    }
}

# --- 2. Regenerate icons ---
Write-Host "[2/4] Generating icon assets" -ForegroundColor Yellow
& $python tools\make_icon.py
if ($LASTEXITCODE -ne 0) { throw "make_icon.py failed" }

# --- 3. PyInstaller ---
Write-Host "[3/4] Running PyInstaller" -ForegroundColor Yellow
& $python -m PyInstaller build.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

# --- 4. Report ---
Write-Host "[4/4] Build complete" -ForegroundColor Yellow
$distPath = Join-Path $root "dist\WaterNetworkModeler"
$sizeBytes = (Get-ChildItem $distPath -Recurse -File | Measure-Object -Property Length -Sum).Sum
$sizeMB = [math]::Round($sizeBytes / 1MB, 1)
$exePath = Join-Path $distPath "WaterNetworkModeler.exe"
Write-Host "  Folder size: $sizeMB MB"
if (Test-Path $exePath) {
    Write-Host "  EXE: $exePath" -ForegroundColor Green
} else {
    throw "WaterNetworkModeler.exe not found!"
}
Write-Host "=== Build finished ===" -ForegroundColor Green

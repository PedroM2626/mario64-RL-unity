# Script to download and install Python 3.8 automatically
# Python 3.8 is the most compatible version with ML-Agents 0.28.0

$pythonInstallerUrl = "https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe"
$installerPath = "$env:TEMP\python-3.8.10-amd64.exe"
$targetDir = "C:\Python38"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Installing Python 3.8 for ML-Agents" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Check if Python 3.8 is already installed
if (Test-Path "$targetDir\python.exe") {
    Write-Host "Python 3.8 is already installed at $targetDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "To continue setup:" -ForegroundColor Cyan
    Write-Host "1. Run: .\setup_training_env.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# Download the installer
Write-Host "Downloading Python 3.8..." -ForegroundColor Cyan
Write-Host "URL: $pythonInstallerUrl" -ForegroundColor Gray

try {
    Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download completed!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to download Python 3.8" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative: Download manually from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3810/" -ForegroundColor Cyan
    exit 1
}

# Install Python silently
Write-Host ""
Write-Host "Installing Python 3.8 at $targetDir..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

$installArgs = "/quiet InstallAllUsers=1 PrependPath=1 TargetDir=`"$targetDir`""
$process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host "ERROR: Python installation failed (Exit code: $($process.ExitCode))" -ForegroundColor Red
    exit 1
}

# Clean up temp file
Remove-Item $installerPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Python 3.8 Installed Successfully!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Location: $targetDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next step:" -ForegroundColor Cyan
Write-Host "Run: .\setup_training_env.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "This will create the virtual environment and install ML-Agents" -ForegroundColor Cyan

# Script to configure ML-Agents training environment with venv
# This script creates an isolated Python virtual environment for training
# IMPORTANT: Uses Python 3.9 for compatibility with ML-Agents 0.28.0

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
$venvPath = "$projectPath\venv_mlagents"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Configuring ML-Agents Environment" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Look for Python 3.8 (best compatibility with ML-Agents 0.28.0)
$python38Path = "C:\Python38\python.exe"
$python39Path = "C:\Python39\python.exe"
$pythonCmd = $null

if (Test-Path $python38Path) {
    $pythonCmd = $python38Path
    Write-Host "Python 3.8 found: $pythonCmd" -ForegroundColor Green
} elseif (Test-Path $python39Path) {
    $pythonCmd = $python39Path
    Write-Host "Python 3.9 found: $pythonCmd" -ForegroundColor Yellow
    Write-Host "WARNING: Python 3.9 may have compatibility issues with ML-Agents 0.28.0" -ForegroundColor Yellow
} else {
    # Try to find via py launcher
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $pyVersion = & py -3.8 --version 2>&1
            if ($pyVersion -match "3.8") {
                $pythonCmd = "py -3.8"
                Write-Host "Python 3.8 found via py launcher" -ForegroundColor Green
            }
        } catch {
            try {
                $pyVersion = & py -3.9 --version 2>&1
                if ($pyVersion -match "3.9") {
                    $pythonCmd = "py -3.9"
                    Write-Host "Python 3.9 found via py launcher" -ForegroundColor Yellow
                }
            } catch {}
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "ERROR: Python 3.8 or 3.9 not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "ML-Agents 0.28.0 requires Python 3.8 or 3.9." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options (run one of the scripts):" -ForegroundColor Cyan
    Write-Host "1. .\setup_python38.ps1  (recommended - most compatible)" -ForegroundColor Yellow
    Write-Host "2. .\setup_python39.ps1  (alternative)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press ENTER to exit..."
    Read-Host
    exit 1
}
Write-Host ""

Set-Location $projectPath

# Remove old venv if it exists
if (Test-Path $venvPath) {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

# Create new venv
Write-Host "Creating virtual environment at: $venvPath" -ForegroundColor Cyan
& $pythonCmd -m venv $venvPath

if (-not (Test-Path $venvPath)) {
    Write-Host "ERROR: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}

Write-Host "Virtual environment created successfully!" -ForegroundColor Green
Write-Host ""

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$activateScript = "$venvPath\Scripts\Activate.ps1"
& $activateScript

# Update pip
Write-Host "Updating pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "Installing ML-Agents and dependencies..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

pip install mlagents==0.28.0 --quiet
pip install torch --quiet
pip install tensorboard --quiet
pip install protobuf==3.20.3 --quiet

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Environment Configured Successfully!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "To use the environment:" -ForegroundColor Cyan
Write-Host "1. Run: .\venv_mlagents\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. Then run: mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or run the training script:" -ForegroundColor Cyan
Write-Host ".\train_with_venv.ps1" -ForegroundColor Yellow
Write-Host ""

# Create training script with venv
$trainScriptContent = @'
# Script to train using the virtual environment
$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Activate venv
$activateScript = "$projectPath\venv_mlagents\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Virtual environment not found. Run setup_training_env.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Starting Mario Training" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Make sure that:" -ForegroundColor Yellow
Write-Host "1. Unity Editor is open with ParkourTraining.unity" -ForegroundColor Yellow
Write-Host "2. Scene is in Play mode" -ForegroundColor Yellow
Write-Host ""

mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Model saved at: results/mario_parkour_v1/" -ForegroundColor Cyan
'@

$trainScriptPath = "$projectPath\train_with_venv.ps1"
$trainScriptContent | Out-File -FilePath $trainScriptPath -Encoding UTF8

Write-Host "Training script created: train_with_venv.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "Press ENTER to exit..."
Read-Host

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

# PowerShell script to train Mario in Chase mode (Pursuer vs Fugitive)
# Run this script in PowerShell

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Set environment variables to work around incompatibilities
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Path to mlagents-learn
$mlagentsPath = "C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Starting Chase SAC Training" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Project: $projectPath" -ForegroundColor Cyan
Write-Host "Config: config/chase_training_sac.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure that:" -ForegroundColor Yellow
Write-Host "1. The Unity Editor is open with the ChaseTraining.unity scene" -ForegroundColor Yellow
Write-Host "2. The scene is in Play mode" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press ENTER to start training..." -ForegroundColor Magenta
Read-Host

& $mlagentsPath config/chase_training_sac.yaml --run-id=chase_sac_v2 --force

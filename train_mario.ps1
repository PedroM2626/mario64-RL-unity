# PowerShell script to train Mario
# Run this script in PowerShell

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Set environment variables to work around incompatibilities
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Path to mlagents-learn
$mlagentsPath = "C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Starting Mario RL Training" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Project: $projectPath" -ForegroundColor Cyan
Write-Host "Config: Assets/ParkourRL/Config/mario_parkour.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "Make sure that:" -ForegroundColor Yellow
Write-Host "1. The Unity Editor is open with the ParkourTraining.unity scene" -ForegroundColor Yellow
Write-Host "2. The scene is in Play mode" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press ENTER to start training..." -ForegroundColor Magenta
Read-Host

& $mlagentsPath Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

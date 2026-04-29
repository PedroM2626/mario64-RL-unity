# PowerShell script to train Mario in hybrid mode
# Run this script in PowerShell

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Set environment variables
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Path to mlagents-learn
$mlagentsPath = "C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Mario Hybrid Training" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Mode: Training (ML-Agents) + Recording (IL/Offline RL)" -ForegroundColor Cyan
Write-Host "Config: mario_parkour_hybrid.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Yellow
Write-Host "1. Open the HybridTraining.unity scene in Unity" -ForegroundColor Yellow
Write-Host "2. Click Play" -ForegroundColor Yellow
Write-Host "3. Press 'M' to toggle between Training/Recording" -ForegroundColor Yellow
Write-Host ""
Write-Host "Player Controls (Green Mario):" -ForegroundColor Yellow
Write-Host "  WASD/Arrows = move" -ForegroundColor Yellow
Write-Host "  Space = jump" -ForegroundColor Yellow
Write-Host ""

Write-Host "Press ENTER to start training..." -ForegroundColor Magenta
Read-Host

& $mlagentsPath Assets/ParkourRL/Config/mario_parkour_hybrid.yaml --run-id=mario_hybrid_v1 --force

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Model saved at: results/mario_hybrid_v1/" -ForegroundColor Cyan
Write-Host ""
Write-Host "To analyze recorded data:" -ForegroundColor Yellow
Write-Host "  cd python_trainers" -ForegroundColor Yellow
Write-Host "  python analyze_dataset.py --data ../HybridTrainingData/ --plots" -ForegroundColor Yellow

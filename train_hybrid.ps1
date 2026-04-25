# Script PowerShell para treinar Mario no modo híbrido
# Execute este script no PowerShell

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Configurar variáveis de ambiente
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Caminho do mlagents-learn
$mlagentsPath = "C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Treinamento Híbrido do Mario" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Modo: Training (ML-Agents) + Recording (IL/Offline RL)" -ForegroundColor Cyan
Write-Host "Config: mario_parkour_hybrid.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "Instruções:" -ForegroundColor Yellow
Write-Host "1. Abra a cena HybridTraining.unity no Unity" -ForegroundColor Yellow
Write-Host "2. Clique em Play" -ForegroundColor Yellow
Write-Host "3. Pressione 'M' para alternar entre Training/Recording" -ForegroundColor Yellow
Write-Host ""
Write-Host "Controles Player (Mario Verde):" -ForegroundColor Yellow
Write-Host "  WASD/Setas = mover" -ForegroundColor Yellow
Write-Host "  Espaço = pular" -ForegroundColor Yellow
Write-Host ""

Write-Host "Pressione ENTER para iniciar treinamento..." -ForegroundColor Magenta
Read-Host

& $mlagentsPath Assets/ParkourRL/Config/mario_parkour_hybrid.yaml --run-id=mario_hybrid_v1 --force

Write-Host ""
Write-Host "Treinamento concluído!" -ForegroundColor Green
Write-Host "Modelo salvo em: results/mario_hybrid_v1/" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para analisar dados gravados:" -ForegroundColor Yellow
Write-Host "  cd python_trainers" -ForegroundColor Yellow
Write-Host "  python analyze_dataset.py --data ../HybridTrainingData/ --plots" -ForegroundColor Yellow

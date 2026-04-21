# Script PowerShell para treinar o Mario
# Execute este script no PowerShell

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Configurar variáveis de ambiente para contornar incompatibilidades
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

# Caminho do mlagents-learn
$mlagentsPath = "C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Iniciando Treinamento do Mario RL" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Projeto: $projectPath" -ForegroundColor Cyan
Write-Host "Config: Assets/ParkourRL/Config/mario_parkour.yaml" -ForegroundColor Cyan
Write-Host ""
Write-Host "Certifique-se de que:" -ForegroundColor Yellow
Write-Host "1. O Unity Editor está aberto com a cena ParkourTraining.unity" -ForegroundColor Yellow
Write-Host "2. A cena está em modo Play" -ForegroundColor Yellow
Write-Host ""
Write-Host "Pressione ENTER para iniciar o treinamento..." -ForegroundColor Magenta
Read-Host

& $mlagentsPath Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

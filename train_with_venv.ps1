# Script para treinar usando o ambiente virtual
$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Ativar venv
$activateScript = "$projectPath\venv_mlagents\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Ambiente virtual ativado!" -ForegroundColor Green
} else {
    Write-Host "ERRO: Ambiente virtual nÃ£o encontrado. Execute setup_training_env.ps1 primeiro." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Iniciando Treinamento do Mario" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Certifique-se de que:" -ForegroundColor Yellow
Write-Host "1. Unity Editor estÃ¡ aberto com ParkourTraining.unity" -ForegroundColor Yellow
Write-Host "2. Cena estÃ¡ em modo Play" -ForegroundColor Yellow
Write-Host ""

mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

Write-Host ""
Write-Host "Treinamento concluÃ­do!" -ForegroundColor Green
Write-Host "Modelo salvo em: results/mario_parkour_v1/" -ForegroundColor Cyan

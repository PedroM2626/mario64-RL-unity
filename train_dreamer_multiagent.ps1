# Script para treinamento DreamerV3 Multi-Agent
# Configuração otimizada para GPU RTX 4070 com 4 agentes paralelos

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "DreamerV3 Multi-Agent Training" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Ativar ambiente virtual
$venvPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master\venv_mlagents\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "Ambiente virtual ativado" -ForegroundColor Green
} else {
    Write-Host "ERRO: Ambiente virtual não encontrado em $venvPath" -ForegroundColor Red
    exit 1
}

# Verificar CUDA
Write-Host "`nVerificando CUDA..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA disponivel: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Modo de execução (com ou sem gráficos)
$useGraphics = $false  # Mude para $true se quiser ver a simulação

# Configurações
$runId = "DreamerMultiAgent_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$timeScale = 8.0
$batchSize = 2048
$numAgents = 4
$seqLen = 64
$horizon = 20
$capacity = 150000

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Configurações do Treinamento:" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Run ID: $runId"
Write-Host "Time Scale: ${timeScale}x"
Write-Host "Batch Size: $batchSize"
Write-Host "Num Agents: $numAgents"
Write-Host "Sequence Length: $seqLen"
Write-Host "Horizon: $horizon"
Write-Host "Buffer Capacity: $capacity"
Write-Host "Graphics: $(if ($useGraphics) { 'ON' } else { 'OFF (mais rápido)' })"
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n==========================================" -ForegroundColor Yellow
Write-Host "FLUXO DE EXECUCAO:" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow

if ($useGraphics) {
    Write-Host "MODO COM GRAFICOS:" -ForegroundColor Cyan
    Write-Host "1. Abra a cena DreamerParkour.unity no Unity" -ForegroundColor White
    Write-Host "2. Verifique se 'Agent Count' = $numAgents no DreamerEnvironment" -ForegroundColor White
    Write-Host "3. Pressione PLAY no Unity" -ForegroundColor White
    Write-Host "4. Execute este script" -ForegroundColor White
} else {
    Write-Host "MODO SEM GRAFICOS (HEADLESS):" -ForegroundColor Cyan
    Write-Host "1. Exporte a cena como Build (Windows x86_64)" -ForegroundColor White
    Write-Host "   - File > Build Settings > Build" -ForegroundColor Gray
    Write-Host "2. Coloque o caminho do .exe na variável `$envPath abaixo" -ForegroundColor White
    Write-Host "3. Execute este script" -ForegroundColor White
    Write-Host ""
    Write-Host "Para treinar com Editor Unity (com gráficos):" -ForegroundColor Yellow
    Write-Host "   Mude `$useGraphics = `$true no script" -ForegroundColor Gray
}

Write-Host "`n==========================================" -ForegroundColor Red
Write-Host "ATENCAO: Unity deve estar rodando antes!" -ForegroundColor Red
Write-Host "==========================================" -ForegroundColor Red

# Perguntar se Unity está pronto
$confirmation = Read-Host "`nUnity está rodando com a cena DreamerParkour? (S/N)"
if ($confirmation -ne 'S' -and $confirmation -ne 's') {
    Write-Host "`nPor favor, inicie o Unity primeiro com a cena DreamerParkour!" -ForegroundColor Red
    exit 1
}

# Construir comando
$envPath = $null  # Mude para "C:\Caminho\Para\Seu\Build.exe" para modo headless

$cmdArgs = @(
    "--run-id", $runId
    "--time-scale", $timeScale
    "--batch-size", $batchSize
    "--num-agents", $numAgents
    "--seq-len", $seqLen
    "--horizon", $horizon
    "--capacity", $capacity
    "--timeout", 600
    "--tb-logdir", "./tensorboard_logs"
)

if ($envPath) {
    $cmdArgs += @("--env", $envPath)
}

if (-not $useGraphics) {
    $cmdArgs += "--no-graphics"
}

Write-Host "`nIniciando treinamento DreamerV3..." -ForegroundColor Green
Write-Host "Comando: python train_dreamer.py $($cmdArgs -join ' ')" -ForegroundColor Gray

# Iniciar treinamento
python train_dreamer.py @cmdArgs

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "Treinamento concluído!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Para visualizar métricas:" -ForegroundColor Yellow
Write-Host "  tensorboard --logdir ./tensorboard_logs" -ForegroundColor Cyan
Write-Host "  mlflow ui" -ForegroundColor Cyan
Write-Host ""
Write-Host "Modelos salvos em: models\$runId\" -ForegroundColor Cyan

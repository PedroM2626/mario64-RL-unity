# Script para configurar ambiente de treinamento ML-Agents com venv
# Este script cria um ambiente virtual Python isolado para o treinamento
# IMPORTANTE: Usa Python 3.9 por compatibilidade com ML-Agents 0.28.0

$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
$venvPath = "$projectPath\venv_mlagents"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Configurando Ambiente ML-Agents" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Procurar Python 3.8 (melhor compatibilidade com ML-Agents 0.28.0)
$python38Path = "C:\Python38\python.exe"
$python39Path = "C:\Python39\python.exe"
$pythonCmd = $null

if (Test-Path $python38Path) {
    $pythonCmd = $python38Path
    Write-Host "Python 3.8 encontrado: $pythonCmd" -ForegroundColor Green
} elseif (Test-Path $python39Path) {
    $pythonCmd = $python39Path
    Write-Host "Python 3.9 encontrado: $pythonCmd" -ForegroundColor Yellow
    Write-Host "AVISO: Python 3.9 pode ter problemas de compatibilidade com ML-Agents 0.28.0" -ForegroundColor Yellow
} else {
    # Tentar encontrar via py launcher
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $pyVersion = & py -3.8 --version 2>&1
            if ($pyVersion -match "3.8") {
                $pythonCmd = "py -3.8"
                Write-Host "Python 3.8 encontrado via py launcher" -ForegroundColor Green
            }
        } catch {
            try {
                $pyVersion = & py -3.9 --version 2>&1
                if ($pyVersion -match "3.9") {
                    $pythonCmd = "py -3.9"
                    Write-Host "Python 3.9 encontrado via py launcher" -ForegroundColor Yellow
                }
            } catch {}
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "ERRO: Python 3.8 ou 3.9 não encontrado!" -ForegroundColor Red
    Write-Host ""
    Write-Host "O ML-Agents 0.28.0 requer Python 3.8 ou 3.9." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Opções (execute um dos scripts):" -ForegroundColor Cyan
    Write-Host "1. .\setup_python38.ps1  (recomendado - mais compatível)" -ForegroundColor Yellow
    Write-Host "2. .\setup_python39.ps1  (alternativa)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Pressione ENTER para sair..."
    Read-Host
    exit 1
}
Write-Host ""

Set-Location $projectPath

# Remover venv antigo se existir
if (Test-Path $venvPath) {
    Write-Host "Removendo ambiente virtual antigo..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venvPath
}

# Criar novo venv
Write-Host "Criando ambiente virtual em: $venvPath" -ForegroundColor Cyan
& $pythonCmd -m venv $venvPath

if (-not (Test-Path $venvPath)) {
    Write-Host "ERRO: Falha ao criar ambiente virtual!" -ForegroundColor Red
    exit 1
}

Write-Host "Ambiente virtual criado com sucesso!" -ForegroundColor Green
Write-Host ""

# Ativar venv
Write-Host "Ativando ambiente virtual..." -ForegroundColor Cyan
$activateScript = "$venvPath\Scripts\Activate.ps1"
& $activateScript

# Atualizar pip
Write-Host "Atualizando pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# Instalar dependências
Write-Host "Instalando ML-Agents e dependências..." -ForegroundColor Cyan
Write-Host "Isso pode levar alguns minutos..." -ForegroundColor Yellow

pip install mlagents==0.28.0 --quiet
pip install torch --quiet
pip install tensorboard --quiet
pip install protobuf==3.20.3 --quiet

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Ambiente Configurado com Sucesso!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para usar o ambiente:" -ForegroundColor Cyan
Write-Host "1. Execute: .\venv_mlagents\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "2. Depois execute: mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ou execute o script de treinamento:" -ForegroundColor Cyan
Write-Host ".\train_with_venv.ps1" -ForegroundColor Yellow
Write-Host ""

# Criar script de treinamento com venv
$trainScriptContent = @'
# Script para treinar usando o ambiente virtual
$projectPath = "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
Set-Location $projectPath

# Ativar venv
$activateScript = "$projectPath\venv_mlagents\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Ambiente virtual ativado!" -ForegroundColor Green
} else {
    Write-Host "ERRO: Ambiente virtual não encontrado. Execute setup_training_env.ps1 primeiro." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Iniciando Treinamento do Mario" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Certifique-se de que:" -ForegroundColor Yellow
Write-Host "1. Unity Editor está aberto com ParkourTraining.unity" -ForegroundColor Yellow
Write-Host "2. Cena está em modo Play" -ForegroundColor Yellow
Write-Host ""

mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

Write-Host ""
Write-Host "Treinamento concluído!" -ForegroundColor Green
Write-Host "Modelo salvo em: results/mario_parkour_v1/" -ForegroundColor Cyan
'@

$trainScriptPath = "$projectPath\train_with_venv.ps1"
$trainScriptContent | Out-File -FilePath $trainScriptPath -Encoding UTF8

Write-Host "Script de treinamento criado: train_with_venv.ps1" -ForegroundColor Green
Write-Host ""
Write-Host "Pressione ENTER para sair..."
Read-Host

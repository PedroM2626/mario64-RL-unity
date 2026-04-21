# Script para baixar e instalar Python 3.9 automaticamente
# Isso resolve a incompatibilidade do ML-Agents com Python 3.10+

$pythonInstallerUrl = "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
$installerPath = "$env:TEMP\python-3.9.13-amd64.exe"
$targetDir = "C:\Python39"

Write-Host "======================================" -ForegroundColor Green
Write-Host "  Instalando Python 3.9 para ML-Agents" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""

# Verificar se Python 3.9 já está instalado
if (Test-Path "$targetDir\python.exe") {
    Write-Host "Python 3.9 já está instalado em $targetDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "Para continuar a configuração:" -ForegroundColor Cyan
    Write-Host "1. Execute: .\setup_training_env.ps1" -ForegroundColor Yellow
    Write-Host ""
    exit 0
}

# Baixar o instalador
Write-Host "Baixando Python 3.9..." -ForegroundColor Cyan
Write-Host "URL: $pythonInstallerUrl" -ForegroundColor Gray
Write-Host "Destino: $installerPath" -ForegroundColor Gray

try {
    Invoke-WebRequest -Uri $pythonInstallerUrl -OutFile $installerPath -UseBasicParsing
    Write-Host "Download concluído!" -ForegroundColor Green
} catch {
    Write-Host "ERRO: Falha ao baixar Python 3.9" -ForegroundColor Red
    Write-Host "Erro: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternativa: Baixe manualmente de:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3913/" -ForegroundColor Cyan
    exit 1
}

# Instalar Python silenciosamente
Write-Host ""
Write-Host "Instalando Python 3.9 em $targetDir..." -ForegroundColor Cyan
Write-Host "Isso pode levar alguns minutos..." -ForegroundColor Yellow

$installArgs = "/quiet InstallAllUsers=1 PrependPath=1 TargetDir=`"$targetDir`""
$process = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host "ERRO: Falha na instalação do Python (Exit code: $($process.ExitCode))" -ForegroundColor Red
    exit 1
}

# Limpar arquivo temporário
Remove-Item $installerPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "  Python 3.9 Instalado com Sucesso!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "Local: $targetDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "Próximo passo:" -ForegroundColor Cyan
Write-Host "Execute: .\setup_training_env.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Isso vai criar o ambiente virtual e instalar o ML-Agents" -ForegroundColor Cyan

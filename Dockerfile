# Usando uma imagem base do Windows com Python já que os scripts usam PowerShell/Windows e ML-Agents precisa do ambiente correto
FROM python:3.9-slim-buster

# Instalar dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar bibliotecas Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o projeto
COPY . .

# Comando padrão
CMD ["python", "mlflow_tracker.py"]

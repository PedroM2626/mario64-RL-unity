FROM python:3.9-slim

# Instalar dependências de sistema necessárias
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar projeto
COPY . .

# Expor porta do MLflow
EXPOSE 5000

# Comando padrão
CMD ["python", "train_mlops.py"]

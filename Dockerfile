FROM python:3.9-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "train_mlops.py"]

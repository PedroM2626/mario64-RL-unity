FROM python:3.9-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (pinned in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (training outputs like models/, results/, mlruns/ are
# .dockerignored/.gitignored locally; mount them as volumes at runtime)
COPY . .

# MLflow (:5000) + TensorBoard (:6006)
EXPOSE 5000 6006

# NOTE: this image contains ONLY the Python trainers. Unity Editor/Build must
# run separately and connect back, e.g.:
#   docker run -it --rm -v ${PWD}/results:/app/results -v ${PWD}/models:/app/models \
#     -p 5000:5000 -p 6006:6006 mariorl:latest python train_mlops.py --run-id dockerrun
# For an offline smoke test without Unity: python evaluate.py --help
CMD ["python", "train_mlops.py", "--help"]

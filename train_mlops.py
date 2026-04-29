import mlflow
import subprocess
import os
import argparse
import yaml
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Assets/ParkourRL/Config/mario_parkour.yaml")
    parser.add_argument("--run-id", default=f"Parkour_Simultaneous_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--env", help="Path to Unity build (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # MLOps: Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_Parkour_Simultaneous_PPO_SAC_DQN")
    
    with mlflow.start_run(run_name=args.run_id) as run:
        print(f"[*] Started MLflow run: {run.info.run_id}")
        
        # MLOps: Register artifacts and hyperparameters
        mlflow.log_artifact(args.config)
        mlflow.log_param("config_file", args.config)
        mlflow.log_param("model_type", "Multi-Agent PPO+SAC+DQN(PPO-variant)")
        
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            # Log some basic configurations
            try:
                for behavior, data in config_data.get("behaviors", {}).items():
                    mlflow.log_param(f"{behavior}_trainer", data.get("trainer_type"))
            except Exception as e:
                print(f"Warning: could not parse behaviors for MLflow logging: {e}")
        
        # Prepare mlagents command
        cmd = [
            "mlagents-learn",
            args.config,
            "--run-id", args.run_id,
            "--force"
        ]
        if args.env:
            cmd.extend(["--env", args.env])
            
        print(f"[*] Running command: {' '.join(cmd)}")
        
        # Start training
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        # MLOps: Finalize experiment and register metrics/models
        if process.returncode == 0:
            print("[*] Training completed successfully.")
            mlflow.log_param("status", "completed")
            
            # Register all generated .onnx models
            results_dir = os.path.join("results", args.run_id)
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith(".onnx"):
                            model_path = os.path.join(root, file)
                            mlflow.log_artifact(model_path, "models")
                            print(f"[*] Logged model to MLflow: {file}")
                            
        else:
            print("[!] Training failed or interrupted.")
            mlflow.log_param("status", "failed")

if __name__ == "__main__":
    main()

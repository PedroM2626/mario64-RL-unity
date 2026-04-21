import os
import subprocess
import yaml
import mlflow
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento com MLOps (MLflow)")
    parser.add_argument("--config", type=str, default="Assets/ParkourRL/Config/mario_parkour.yaml", help="Path to config yaml")
    parser.add_argument("--run-id", type=str, default=f"mario_parkour_run_{int(time.time())}", help="Run ID for ML-Agents")
    parser.add_argument("--env", type=str, default=None, help="Path to Unity build (None for Editor training)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Iniciar experimento no MLflow
    mlflow.set_experiment("Mario_Parkour_RL")
    
    with mlflow.start_run(run_name=args.run_id):
        print(f"======================================")
        print(f"Iniciando Treinamento com MLOps...")
        print(f"Run ID: {args.run_id}")
        print(f"Config: {args.config}")
        print(f"======================================")
        
        # Registrar os parâmetros do YAML
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Logar alguns meta-dados chave
        mlflow.log_param("trainer_type", config_data['behaviors']['MarioParkour']['trainer_type'])
        mlflow.log_param("max_steps", config_data['behaviors']['MarioParkour']['max_steps'])
        mlflow.log_param("batch_size", config_data['behaviors']['MarioParkour']['hyperparameters']['batch_size'])
        mlflow.log_param("learning_rate", config_data['behaviors']['MarioParkour']['hyperparameters']['learning_rate'])
        mlflow.log_param("hidden_units", config_data['behaviors']['MarioParkour']['network_settings']['hidden_units'])
        
        mlflow.log_artifact(args.config, artifact_path="configs")
        
        # Autolog para Tensorboard (pega as métricas geradas pelo ML-Agents)
        mlflow.tensorboard.autolog()
        
        # Comando para mlagents-learn
        cmd = ["mlagents-learn", args.config, "--run-id", args.run_id, "--force"]
        if args.env:
            cmd.extend(["--env", args.env])
            
        print(f"Executando processo filho: {' '.join(cmd)}")
        print("Aguardando o Unity Editor ou fechamento do ambiente...\n")
        
        try:
            # Processo bloqueante
            subprocess.run(cmd, check=True)
            print("\nTreinamento Finalizado com Sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"\nErro durante o treinamento: {e}")
            mlflow.log_param("status", "failed")
        except KeyboardInterrupt:
            print(f"\nTreinamento interrompido pelo usuário.")
            mlflow.log_param("status", "interrupted")
        
        # No final, o modelo ONNX e os logs do tensorboard estarão em results/run_id/
        model_path = f"results/{args.run_id}/MarioParkour.onnx"
        if os.path.exists(model_path):
            print(f"Salvando modelo gerado no MLOps: {model_path}")
            mlflow.log_artifact(model_path, artifact_path="models")
            
        # Salvar as métricas finais / summaries 
        pt_path = f"results/{args.run_id}/MarioParkour"
        if os.path.exists(pt_path):
            mlflow.log_artifacts(pt_path, artifact_path="pytorch_models")

if __name__ == "__main__":
    main()

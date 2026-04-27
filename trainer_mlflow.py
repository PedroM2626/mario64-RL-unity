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
            
        # Logar alguns meta-dados chave (pega o primeiro behavior disponivel)
        behaviors = config_data.get('behaviors', {})
        first_behavior_name = next(iter(behaviors.keys()), None) if behaviors else None
        if first_behavior_name:
            first_behavior = behaviors[first_behavior_name]
            mlflow.log_param("behavior_name", first_behavior_name)
            mlflow.log_param("trainer_type", first_behavior.get('trainer_type', 'unknown'))
            mlflow.log_param("max_steps", first_behavior.get('max_steps', 0))
            mlflow.log_param("batch_size", first_behavior.get('hyperparameters', {}).get('batch_size', 0))
            mlflow.log_param("learning_rate", first_behavior.get('hyperparameters', {}).get('learning_rate', 0))
            mlflow.log_param("hidden_units", first_behavior.get('network_settings', {}).get('hidden_units', 0))
        else:
            mlflow.log_param("trainer_type", config_data.get('default_settings', {}).get('trainer_type', 'unknown'))
        
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
        
        # No final, tentar salvar modelos de todos os behaviors encontrados
        behaviors = config_data.get('behaviors', {})
        for behavior_name in behaviors.keys():
            model_path = f"results/{args.run_id}/{behavior_name}.onnx"
            if os.path.exists(model_path):
                print(f"Salvando modelo gerado no MLOps: {model_path}")
                mlflow.log_artifact(model_path, artifact_path="models")
            pt_path = f"results/{args.run_id}/{behavior_name}"
            if os.path.exists(pt_path):
                mlflow.log_artifacts(pt_path, artifact_path="pytorch_models")

if __name__ == "__main__":
    main()

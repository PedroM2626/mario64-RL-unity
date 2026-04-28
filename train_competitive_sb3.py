import os
import argparse
import numpy as np
import torch
import mlflow
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.logger import Logger
import torch as th


def export_onnx(model, path, is_dqn=False):
    """
    Exporta o modelo SB3 para formato ONNX simples.
    """
    class OnnxWrapper(th.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        def forward(self, obs):
            return self.policy._predict(obs, deterministic=True)
            
    dummy_input = th.randn(1, 42).to(model.device)
    wrapper = OnnxWrapper(model.policy).to(model.device)
    wrapper.eval()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    th.onnx.export(
        wrapper,
        dummy_input,
        path,
        opset_version=11,
        input_names=["vector_observation"],
        output_names=["action"]
    )


class CompetitiveParkourEnv(gym.Env):
    """
    Ambiente Gymnasium dedicado a CompetitiveParkour.
    Usado apenas para inicializar a arquitetura das redes neurais do SB3.
    As transicoes sao injetadas manualmente no loop principal.
    """
    def __init__(self, is_dqn=False):
        super().__init__()
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(42,), dtype=np.float32)
        
        if is_dqn:
            self.action_space = spaces.Discrete(18)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
            
    def step(self, action):
        return np.zeros(42), 0.0, False, False, {}
        
    def reset(self, seed=None):
        return np.zeros(42), {}


def convert_dqn_action(act):
    """ Mapeia uma das 18 acoes discretas do DQN para ActionTuple da Unity """
    joy_x_map = [-1.0, 0.0, 1.0]
    joy_y_map = [-1.0, 0.0, 1.0]
    
    j = act % 2
    act //= 2
    y = act % 3
    act //= 3
    x = act % 3
    
    cont = np.array([[joy_x_map[x], joy_y_map[y]]], dtype=np.float32)
    disc = np.array([[j, 0, 0]], dtype=np.int32)
    return ActionTuple(continuous=cont, discrete=disc)


def convert_box_action(act):
    """ Converte saida Box do SAC e PPO para ActionTuple da Unity """
    cont = np.array([[act[0], act[1]]], dtype=np.float32)
    j = 1 if act[2] > 0 else 0
    k = 1 if act[3] > 0 else 0
    s = 1 if act[4] > 0 else 0
    disc = np.array([[j, k, s]], dtype=np.int32)
    return ActionTuple(continuous=cont, discrete=disc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, help="Caminho do executavel (None para rodar via Play no Editor da Unity)")
    parser.add_argument("--run-id", type=str, default=f"CompetitiveParkour_SB3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--force", action="store_true", help="Overwrite (ignore previous models)")
    parser.add_argument("--tb-logdir", type=str, default="./tensorboard_logs", help="Diretorio do TensorBoard")
    parser.add_argument("--time-scale", type=float, default=3.0, help="Time scale da Unity")
    args = parser.parse_args()

    # MLOps: Inicializando Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_CompetitiveParkour_SB3")

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, args.run_id))
    print(f"[*] TensorBoard logs em: {os.path.join(args.tb_logdir, args.run_id)}")

    with mlflow.start_run(run_name=args.run_id) as run:
        print(f"[*] Started MLflow run: {run.info.run_id}")
        mlflow.log_param("Framework", "Stable-Baselines3 + Unity ML-Agents Bridge")
        mlflow.log_param("Scene", "CompetitiveParkour")
        mlflow.log_param("time_scale", args.time_scale)
        mlflow.log_param("obs_size", 42)
        mlflow.log_param("ppo_n_steps", 2048)
        mlflow.log_param("sac_buffer", 50000)
        mlflow.log_param("dqn_buffer", 50000)
        
        print("[*] Instanciando verdadeiros algoritmos no PyTorch...")
        env_ppo_sac = CompetitiveParkourEnv(is_dqn=False)
        env_dqn = CompetitiveParkourEnv(is_dqn=True)
        
        ppo_path = f"models/{args.run_id}/MarioParkourPPO_model.zip"
        sac_path = f"models/{args.run_id}/MarioParkourSAC_model.zip"
        dqn_path = f"models/{args.run_id}/MarioParkourDQN_model.zip"
        
        if args.resume and not args.force:
            print("[*] Tentando restaurar checkpoints anteriores...")
            ppo = PPO.load(ppo_path, env=env_ppo_sac) if os.path.exists(ppo_path) else PPO("MlpPolicy", env_ppo_sac, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto")
            sac = SAC.load(sac_path, env=env_ppo_sac) if os.path.exists(sac_path) else SAC("MlpPolicy", env_ppo_sac, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto")
            dqn = DQN.load(dqn_path, env=env_dqn) if os.path.exists(dqn_path) else DQN("MlpPolicy", env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto")
            # Configurar logger para modelos carregados tambem
            if not hasattr(sac, '_logger') or sac._logger is None:
                sac.set_logger(Logger(folder=None, output_formats=["stdout"]))
            if not hasattr(dqn, '_logger') or dqn._logger is None:
                dqn.set_logger(Logger(folder=None, output_formats=["stdout"]))
        else:
            ppo = PPO("MlpPolicy", env_ppo_sac, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto", tensorboard_log=args.tb_logdir)
            sac = SAC("MlpPolicy", env_ppo_sac, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto", tensorboard_log=args.tb_logdir)
            dqn = DQN("MlpPolicy", env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto", tensorboard_log=args.tb_logdir)
        
        # Configurar logger para SAC e DQN (necessario para train())
        sac.set_logger(Logger(folder=None, output_formats=["stdout"]))
        dqn.set_logger(Logger(folder=None, output_formats=["stdout"]))

        # Conectar na Unity
        print("[*] Aguardando conexao com a Unity (De Play na cena CompetitiveParkour)...")
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=args.time_scale)
        env = UnityEnvironment(file_name=args.env, side_channels=[channel], no_graphics=False)
        env.reset()
        
        behavior_names = list(env.behavior_specs.keys())
        print(f"[*] Behaviors detectados na cena: {behavior_names}")

        obs_dict = {}
        for name in behavior_names:
            dec, term = env.get_steps(name)
            if len(dec) > 0:
                obs_dict[name] = dec.obs[0][0]
            else:
                obs_dict[name] = np.zeros(42, dtype=np.float32)

        ep_rewards = {name: 0.0 for name in behavior_names}
        ep_counts = {name: 0 for name in behavior_names}
        ep_lengths = {name: 0 for name in behavior_names}
        
        # Metricas agregadas por modelo
        model_types = {}
        for name in behavior_names:
            if "PPO" in name:
                model_types[name] = "PPO"
            elif "SAC" in name:
                model_types[name] = "SAC"
            elif "DQN" in name:
                model_types[name] = "DQN"
            else:
                model_types[name] = "Unknown"

        step = 0
        try:
            print("[*] Treinamento competitivo iniciado com sucesso!")
            while True:
                actions_to_send = {}
                sb3_actions = {}

                for name in behavior_names:
                    obs = obs_dict[name]
                    
                    if "PPO" in name:
                        with torch.no_grad():
                            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(ppo.device)
                            action, value, log_prob = ppo.policy.forward(obs_t)
                        act_np = action.cpu().numpy()[0]
                        # Manter tensores para o buffer, mas numpy para envio
                        sb3_actions[name] = (act_np, action, value, log_prob)
                        actions_to_send[name] = convert_box_action(act_np)
                        
                    elif "SAC" in name:
                        action, _ = sac.predict(obs, deterministic=False)
                        sb3_actions[name] = action
                        actions_to_send[name] = convert_box_action(action)
                        
                    elif "DQN" in name:
                        epsilon = max(0.05, 1.0 - step / (dqn.exploration_fraction * 100000))
                        if np.random.rand() < epsilon:
                            action = np.random.randint(18)
                        else:
                            action, _ = dqn.predict(obs, deterministic=True)
                            action = int(action)
                        sb3_actions[name] = action
                        actions_to_send[name] = convert_dqn_action(action)

                for name in behavior_names:
                    if name in actions_to_send:
                        env.set_actions(name, actions_to_send[name])
                env.step()

                for name in behavior_names:
                    if name not in obs_dict:
                        continue
                    
                    dec, term = env.get_steps(name)
                    if len(term) > 0:
                        next_obs = term.obs[0][0]
                        reward = term.reward[0]
                        done = True
                    elif len(dec) > 0:
                        next_obs = dec.obs[0][0]
                        reward = dec.reward[0]
                        done = False
                    else:
                        continue
                        
                    ep_rewards[name] += reward
                    ep_lengths[name] += 1
                    old_obs = obs_dict[name]

                    if "PPO" in name:
                        act_np, action_t, value_t, log_prob_t = sb3_actions[name]
                        # Converter para tensores se necessario
                        if not isinstance(value_t, torch.Tensor):
                            value_t = torch.tensor(value_t).to(ppo.device)
                        if not isinstance(log_prob_t, torch.Tensor):
                            log_prob_t = torch.tensor(log_prob_t).to(ppo.device)
                        ppo.rollout_buffer.add(old_obs, act_np, reward, done, value_t, log_prob_t)
                        if ppo.rollout_buffer.full:
                            with torch.no_grad():
                                next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(ppo.device)
                                last_value = ppo.policy.predict_values(next_obs_t).flatten()
                            ppo.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.array([done]))
                            ppo.train()
                            ppo.rollout_buffer.reset()
                            
                    elif "SAC" in name:
                        sac.replay_buffer.add(old_obs, next_obs, sb3_actions[name], reward, done, [{}])
                        if step > sac.learning_starts:
                            sac.train(batch_size=sac.batch_size, gradient_steps=1)
                            
                    elif "DQN" in name:
                        dqn.replay_buffer.add(old_obs, next_obs, np.array([sb3_actions[name]]), reward, done, [{}])
                        if step > dqn.learning_starts:
                            dqn.train(batch_size=dqn.batch_size, gradient_steps=1)

                    obs_dict[name] = next_obs
                    
                    if done:
                        ep_counts[name] += 1
                        model = model_types.get(name, "Unknown")
                        print(f"[{name}] Episodio {ep_counts[name]} finalizado | Recompensa: {ep_rewards[name]:.2f} | Length: {ep_lengths[name]}")
                        
                        clean_name = name.replace("?", "_").replace("=", "_").replace("-", "_")
                        mlflow.log_metric(f"{clean_name}_reward", ep_rewards[name], step=ep_counts[name])
                        mlflow.log_metric(f"{clean_name}_length", ep_lengths[name], step=ep_counts[name])
                        
                        # TensorBoard
                        tb_writer.add_scalar(f"Rewards/{clean_name}", ep_rewards[name], ep_counts[name])
                        tb_writer.add_scalar(f"EpisodeLength/{clean_name}", ep_lengths[name], ep_counts[name])
                        tb_writer.add_scalar(f"ModelRewards/{model}", ep_rewards[name], ep_counts[name])
                        
                        ep_rewards[name] = 0.0
                        ep_lengths[name] = 0
                        
                        if ep_counts[name] % 50 == 0:
                            model_path = f"models/{args.run_id}/{name}_model.zip"
                            onnx_path = f"models/{args.run_id}/{name}_model.onnx"
                            os.makedirs(os.path.dirname(model_path), exist_ok=True)
                            
                            if "PPO" in name: 
                                ppo.save(model_path)
                                export_onnx(ppo, onnx_path)
                            elif "SAC" in name: 
                                sac.save(model_path)
                                export_onnx(sac, onnx_path)
                            elif "DQN" in name: 
                                dqn.save(model_path)
                                export_onnx(dqn, onnx_path, is_dqn=True)
                                
                            mlflow.log_artifact(model_path, "models_checkpoints")
                            mlflow.log_artifact(onnx_path, "models_checkpoints")

                step += 1

        except KeyboardInterrupt:
            print("[!] Treinamento interrompido pelo usuario. Salvando modelos finais...")
            os.makedirs(f"models/{args.run_id}", exist_ok=True)
            ppo.save(f"models/{args.run_id}/MarioParkourPPO_final.zip")
            sac.save(f"models/{args.run_id}/MarioParkourSAC_final.zip")
            dqn.save(f"models/{args.run_id}/MarioParkourDQN_final.zip")
            
            export_onnx(ppo, f"models/{args.run_id}/MarioParkourPPO_final.onnx")
            export_onnx(sac, f"models/{args.run_id}/MarioParkourSAC_final.onnx")
            export_onnx(dqn, f"models/{args.run_id}/MarioParkourDQN_final.onnx", is_dqn=True)
            
            mlflow.log_artifact(f"models/{args.run_id}", "models_final")
        finally:
            env.close()
            tb_writer.close()


if __name__ == "__main__":
    main()

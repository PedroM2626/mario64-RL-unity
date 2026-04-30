import os
import argparse
import numpy as np
import torch
import mlflow
import time as time_module
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
    Export the SB3 model to a simple ONNX format.
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
    Gymnasium environment dedicated to CompetitiveParkour.
    Used only to initialize the SB3 neural network architecture.
    Transitions are manually injected in the main loop.
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
    """ Maps one of the 18 discrete DQN actions to a Unity ActionTuple """
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
    """ Converts Box output from SAC and PPO to Unity ActionTuple """
    cont = np.array([[act[0], act[1]]], dtype=np.float32)
    j = 1 if act[2] > 0 else 0
    k = 1 if act[3] > 0 else 0
    s = 1 if act[4] > 0 else 0
    disc = np.array([[j, k, s]], dtype=np.int32)
    return ActionTuple(continuous=cont, discrete=disc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, help="Path to Unity executable (None to run via Play in the Unity Editor)")
    parser.add_argument("--run-id", type=str, default=f"CompetitiveParkour_SB3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--force", action="store_true", help="Overwrite (ignore previous models)")
    parser.add_argument("--tb-logdir", type=str, default="./tensorboard_logs", help="TensorBoard log directory")
    parser.add_argument("--time-scale", type=float, default=3.0, help="Unity time scale")
    args = parser.parse_args()

    # MLOps: Initialize Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_CompetitiveParkour_SB3")

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, args.run_id))
    print(f"[*] TensorBoard logs at: {os.path.join(args.tb_logdir, args.run_id)}")

    with mlflow.start_run(run_name=args.run_id) as run:
        print(f"[*] Started MLflow run: {run.info.run_id}")
        mlflow.log_param("Framework", "Stable-Baselines3 + Unity ML-Agents Bridge")
        mlflow.log_param("Scene", "CompetitiveParkour")
        mlflow.log_param("time_scale", args.time_scale)
        mlflow.log_param("obs_size", 42)
        mlflow.log_param("ppo_n_steps", 2048)
        mlflow.log_param("sac_buffer", 50000)
        mlflow.log_param("dqn_buffer", 50000)
        
        print("[*] Instantiating real algorithms in PyTorch...")
        env_ppo_sac = CompetitiveParkourEnv(is_dqn=False)
        env_dqn = CompetitiveParkourEnv(is_dqn=True)
        
        ppo_path = f"models/{args.run_id}/MarioParkourPPO_model.zip"
        sac_path = f"models/{args.run_id}/MarioParkourSAC_model.zip"
        dqn_path = f"models/{args.run_id}/MarioParkourDQN_model.zip"
        
        if args.resume and not args.force:
            print("[*] Attempting to restore previous checkpoints...")
            ppo = PPO.load(ppo_path, env=env_ppo_sac) if os.path.exists(ppo_path) else PPO("MlpPolicy", env_ppo_sac, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto")
            sac = SAC.load(sac_path, env=env_ppo_sac) if os.path.exists(sac_path) else SAC("MlpPolicy", env_ppo_sac, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto")
            dqn = DQN.load(dqn_path, env=env_dqn) if os.path.exists(dqn_path) else DQN("MlpPolicy", env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto")
        else:
            ppo = PPO("MlpPolicy", env_ppo_sac, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto", tensorboard_log=args.tb_logdir)
            sac = SAC("MlpPolicy", env_ppo_sac, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto", tensorboard_log=args.tb_logdir)
            dqn = DQN("MlpPolicy", env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto", tensorboard_log=args.tb_logdir)
        
        # Configure logger for PPO, SAC and DQN (required for train())
        ppo.set_logger(Logger(folder=None, output_formats=["stdout"]))
        sac.set_logger(Logger(folder=None, output_formats=["stdout"]))
        dqn.set_logger(Logger(folder=None, output_formats=["stdout"]))

        # Connect to Unity
        print("[*] Waiting for Unity connection (Press Play in the CompetitiveParkour scene)...")
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=args.time_scale)
        env = UnityEnvironment(file_name=args.env, side_channels=[channel], no_graphics=False)
        env.reset()
        
        behavior_names = list(env.behavior_specs.keys())
        print(f"[*] Behaviors detected in scene: {behavior_names}")

        # Initial observations
        obs_dict = {}
        for name in behavior_names:
            dec, term = env.get_steps(name)
            if len(dec) > 0:
                obs_dict[name] = dec.obs[0][0]
            else:
                obs_dict[name] = np.zeros(42, dtype=np.float32)

        # Per-agent metrics
        ep_rewards = {name: 0.0 for name in behavior_names}
        ep_counts = {name: 0 for name in behavior_names}
        ep_lengths = {name: 0 for name in behavior_names}
        ep_start_times = {name: time_module.time() for name in behavior_names}
        
        # Last action data for PPO (needs to be stored between steps)
        last_ppo_data = {}
        
        # Aggregated metrics per model
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

        # PPO episode_start counters
        ppo_episode_start = {name: True for name in behavior_names if "PPO" in name}

        step = 0
        last_progress_time = time_module.time()
        training_start_time = time_module.time()
        last_sb3_actions = {}  # Actions from previous iteration for reward processing
        
        try:
            print("[*] Competitive training started successfully!")
            while True:
                # ---- Single get_steps per iteration (standard ML-Agents pattern) ----
                # This returns results from the PREVIOUS env.step() (or env.reset()).
                # We process rewards/terminals AND compute new actions in one pass.


                actions_to_send = {}
                sb3_actions = {}

                for name in behavior_names:
                    dec, term = env.get_steps(name)



                    # --- 1) Process terminal steps (episode ended) ---
                    if len(term) > 0:
                        reward = term.reward[0]
                        ep_rewards[name] += reward
                        ep_lengths[name] += 1
                        old_obs = obs_dict.get(name, term.obs[0][0])

                        # Add final transition
                        if name in last_sb3_actions:
                            if "PPO" in name:
                                act_np, action_t, value_t, log_prob_t = last_sb3_actions[name]
                                if not isinstance(value_t, torch.Tensor):
                                    value_t = torch.tensor(value_t).to(ppo.device)
                                if not isinstance(log_prob_t, torch.Tensor):
                                    log_prob_t = torch.tensor(log_prob_t).to(ppo.device)
                                is_episode_start = ppo_episode_start.get(name, False)
                                ppo.rollout_buffer.add(
                                    old_obs, act_np, reward,
                                    is_episode_start,
                                    value_t, log_prob_t
                                )
                                ppo_episode_start[name] = False
                                if ppo.rollout_buffer.full:
                                    last_value = torch.zeros(1, device=ppo.device)
                                    ppo.rollout_buffer.compute_returns_and_advantage(
                                        last_values=last_value,
                                        dones=np.array([True])
                                    )
                                    ppo.train()
                                    ppo.rollout_buffer.reset()
                            elif "SAC" in name:
                                next_obs_term = term.obs[0][0]
                                sac.replay_buffer.add(old_obs, next_obs_term, last_sb3_actions[name], reward, True, [{}])
                                if step > sac.learning_starts and step % 10 == 0:
                                    sac.train(batch_size=sac.batch_size, gradient_steps=1)
                            elif "DQN" in name:
                                next_obs_term = term.obs[0][0]
                                dqn.replay_buffer.add(old_obs, next_obs_term, np.array([last_sb3_actions[name]]), reward, True, [{}])
                                if step > dqn.learning_starts and step % 10 == 0:
                                    dqn.train(batch_size=dqn.batch_size, gradient_steps=1)

                        # Clear the last action so we don't bleed into the next episode
                        if name in last_sb3_actions:
                            del last_sb3_actions[name]

                        # Episode-end bookkeeping
                        ep_counts[name] += 1
                        model = model_types.get(name, "Unknown")
                        elapsed = time_module.time() - ep_start_times[name]

                        print(f"[{name}] Ep {ep_counts[name]} | Reward: {ep_rewards[name]:.2f} | Steps: {ep_lengths[name]} | Time: {elapsed:.1f}s | Total steps: {step}")
                        clean_name = name.replace("?", "_").replace("=", "_").replace("-", "_")
                        mlflow.log_metric(f"{clean_name}_reward", ep_rewards[name], step=ep_counts[name])
                        mlflow.log_metric(f"{clean_name}_length", ep_lengths[name], step=ep_counts[name])
                        mlflow.log_metric(f"{clean_name}_time", elapsed, step=ep_counts[name])
                        tb_writer.add_scalar(f"Rewards/{clean_name}", ep_rewards[name], ep_counts[name])
                        tb_writer.add_scalar(f"EpisodeLength/{clean_name}", ep_lengths[name], ep_counts[name])
                        
                        ep_rewards[name] = 0.0
                        ep_lengths[name] = 0
                        ep_start_times[name] = time_module.time()

                        if "PPO" in name:
                            ppo_episode_start[name] = True

                        if ep_counts[name] % 50 == 0:
                            model_path = f"models/{args.run_id}/{clean_name}_model.zip"
                            onnx_path = f"models/{args.run_id}/{clean_name}_model.onnx"
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

                    # --- 2) Process decision steps (agent needs next action) ---
                    if len(dec) > 0:
                        new_obs = dec.obs[0][0]

                        # Add transition from the previous decision period
                        if name in last_sb3_actions:
                            reward = dec.reward[0]
                            ep_rewards[name] += reward
                            ep_lengths[name] += 1
                            old_obs = obs_dict[name]

                            if "PPO" in name:
                                act_np, action_t, value_t, log_prob_t = last_sb3_actions[name]
                                if not isinstance(value_t, torch.Tensor):
                                    value_t = torch.tensor(value_t).to(ppo.device)
                                if not isinstance(log_prob_t, torch.Tensor):
                                    log_prob_t = torch.tensor(log_prob_t).to(ppo.device)
                                is_episode_start = ppo_episode_start.get(name, False)
                                ppo.rollout_buffer.add(
                                    old_obs, act_np, reward,
                                    is_episode_start,
                                    value_t, log_prob_t
                                )
                                ppo_episode_start[name] = False
                                if ppo.rollout_buffer.full:
                                    with torch.no_grad():
                                        next_obs_t = torch.tensor(new_obs, dtype=torch.float32).unsqueeze(0).to(ppo.device)
                                        last_value = ppo.policy.predict_values(next_obs_t).flatten()
                                    ppo.rollout_buffer.compute_returns_and_advantage(
                                        last_values=last_value,
                                        dones=np.array([False])
                                    )
                                    ppo.train()
                                    ppo.rollout_buffer.reset()

                            elif "SAC" in name:
                                sac.replay_buffer.add(old_obs, new_obs, last_sb3_actions[name], reward, False, [{}])
                                if step > sac.learning_starts and step % 10 == 0:
                                    sac.train(batch_size=sac.batch_size, gradient_steps=1)

                            elif "DQN" in name:
                                dqn.replay_buffer.add(old_obs, new_obs, np.array([last_sb3_actions[name]]), reward, False, [{}])
                                if step > dqn.learning_starts and step % 10 == 0:
                                    dqn.train(batch_size=dqn.batch_size, gradient_steps=1)

                        obs_dict[name] = new_obs
                        
                        # --- 3) Compute new action ---
                        obs = new_obs
                        if "PPO" in name:
                            with torch.no_grad():
                                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(ppo.device)
                                action, value, log_prob = ppo.policy.forward(obs_t)
                            act_np = action.cpu().numpy()[0]
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

                # --- 4) Send actions and advance simulation ---
                for name in actions_to_send:
                    env.set_actions(name, actions_to_send[name])
                
                env.step()

                # Merge new actions into persistent store
                for name in sb3_actions:
                    last_sb3_actions[name] = sb3_actions[name]

                step += 1
                
                # Progress log every 30 seconds
                now = time_module.time()
                if now - last_progress_time > 30.0:
                    total_elapsed = now - training_start_time
                    total_eps = sum(ep_counts.values())
                    eps_per_min = total_eps / (total_elapsed / 60.0) if total_elapsed > 0 else 0
                    ppo_name = next((n for n in behavior_names if "PPO" in n), None)
                    sac_name = next((n for n in behavior_names if "SAC" in n), None)
                    dqn_name = next((n for n in behavior_names if "DQN" in n), None)
                    
                    # In-progress episode info
                    in_progress = []
                    for n in behavior_names:
                        short = model_types.get(n, "?")
                        ep_time = now - ep_start_times[n]
                        in_progress.append(f"{short}:{ep_rewards[n]:.1f}r/{ep_time:.0f}s")
                    progress_str = ", ".join(in_progress)
                    
                    print(
                        f"[Progress] Step: {step} | "
                        f"Eps: {total_eps} ({eps_per_min:.1f}/min) | "
                        f"Elapsed: {total_elapsed:.0f}s | "
                        f"PPO:{ep_counts.get(ppo_name, 0)} SAC:{ep_counts.get(sac_name, 0)} DQN:{ep_counts.get(dqn_name, 0)} | "
                        f"Running: [{progress_str}]"
                    )
                    last_progress_time = now

        except KeyboardInterrupt:
            print("[!] Training interrupted by user. Saving final models...")
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

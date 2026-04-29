import os
import argparse
import numpy as np
import torch
import mlflow
from datetime import datetime

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, DQN
import torch as th

def export_onnx(model, path, is_dqn=False):
    """
    Export the SB3 model to a simple ONNX format.
    Unity ML-Agents may not fully recognize it natively without version_number,
    but the base inference structure will be accessible.
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

class MockEnv(gym.Env):
    """
    A mock environment purely to initialize the Stable-Baselines3 neural network architecture.
    We manually inject transitions into the buffers in the main loop.
    """
    def __init__(self, is_dqn=False):
        super().__init__()
        # VectorObservationSize = 42 from Unity
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(42,), dtype=np.float32)
        
        if is_dqn:
            # True DQN: DQN requires a strictly discrete action space.
            # We discretize Mario's control: 3 X directions, 3 Y directions, 2 Jump = 18 actions.
            self.action_space = spaces.Discrete(18)
        else:
            # PPO and SAC support continuous spaces. Box(5) -> 2 Joysticks + 3 Buttons simulated as float.
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
    parser.add_argument("--run-id", type=str, default=f"Parkour_SB3_TrueDQN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if exists")
    parser.add_argument("--force", action="store_true", help="Overwrite (ignore previous models)")
    args = parser.parse_args()

    # MLOps: Initialize Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_Parkour_TrueDQN_Simultaneous")

    with mlflow.start_run(run_name=args.run_id) as run:
        print(f"[*] Started MLflow run: {run.info.run_id}")
        mlflow.log_param("Framework", "Stable-Baselines3 + Unity ML-Agents Bridge")
        
        print("[*] Instantiating real algorithms in PyTorch...")
        mock_env_box = MockEnv(is_dqn=False)
        mock_env_dqn = MockEnv(is_dqn=True)
        
        ppo_path = f"models/{args.run_id}/MarioParkourPPO_model.zip"
        sac_path = f"models/{args.run_id}/MarioParkourSAC_model.zip"
        dqn_path = f"models/{args.run_id}/MarioParkourDQN_model.zip"
        
        if args.resume and not args.force:
            print("[*] Attempting to restore previous checkpoints...")
            ppo = PPO.load(ppo_path, env=mock_env_box) if os.path.exists(ppo_path) else PPO("MlpPolicy", mock_env_box, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto")
            sac = SAC.load(sac_path, env=mock_env_box) if os.path.exists(sac_path) else SAC("MlpPolicy", mock_env_box, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto")
            dqn = DQN.load(dqn_path, env=mock_env_dqn) if os.path.exists(dqn_path) else DQN("MlpPolicy", mock_env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto")
        else:
            ppo = PPO("MlpPolicy", mock_env_box, n_steps=2048, batch_size=64, learning_rate=0.0003, device="auto")
            sac = SAC("MlpPolicy", mock_env_box, buffer_size=50000, batch_size=256, learning_starts=1000, device="auto")
            dqn = DQN("MlpPolicy", mock_env_dqn, buffer_size=50000, batch_size=128, learning_starts=1000, exploration_fraction=0.2, device="auto")

        # Connect to Unity
        print("[*] Waiting for Unity connection (Press Play in the CompetitiveParkour scene)...")
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=3.0)
        env = UnityEnvironment(file_name=args.env, side_channels=[channel], no_graphics=False)
        env.reset()
        
        behavior_names = list(env.behavior_specs.keys())
        print(f"[*] Behaviors detected in scene: {behavior_names}")

        obs_dict = {}
        for name in behavior_names:
            dec, term = env.get_steps(name)
            if len(dec) > 0:
                obs_dict[name] = dec.obs[0][0]
            else:
                obs_dict[name] = np.zeros(42, dtype=np.float32)

        ep_rewards = {name: 0.0 for name in behavior_names}
        ep_counts = {name: 0 for name in behavior_names}

        step = 0
        try:
            print("[*] Simultaneous training started successfully!")
            while True:
                actions_to_send = {}
                sb3_actions = {}

                # 1. Get actions from the 3 neural networks (Forward Pass)
                for name in behavior_names:
                    obs = obs_dict[name]
                    
                    if "PPO" in name:
                        with torch.no_grad():
                            obs_t = torch.tensor(obs).unsqueeze(0).to(ppo.device)
                            action, value, log_prob = ppo.policy.forward(obs_t)
                        act_np = action.cpu().numpy()[0]
                        sb3_actions[name] = (act_np, value.cpu().numpy()[0], log_prob.cpu().numpy()[0])
                        actions_to_send[name] = convert_box_action(act_np)
                        
                    elif "SAC" in name:
                        action, _ = sac.predict(obs, deterministic=False)
                        sb3_actions[name] = action
                        actions_to_send[name] = convert_box_action(action)
                        
                    elif "DQN" in name:
                        # Manual epsilon-greedy exploration to inject into the Unity loop
                        epsilon = max(0.05, 1.0 - step / (dqn.exploration_fraction * 100000))
                        if np.random.rand() < epsilon:
                            action = np.random.randint(18)
                        else:
                            action, _ = dqn.predict(obs, deterministic=True)
                            action = int(action)
                        sb3_actions[name] = action
                        actions_to_send[name] = convert_dqn_action(action)

                # 2. Send actions through the ML-Agents bridge and advance 1 physics step
                for name in behavior_names:
                    if name in actions_to_send:
                        env.set_actions(name, actions_to_send[name])
                env.step()

                # 3. Process new states and rewards
                for name in behavior_names:
                    if name not in obs_dict: continue
                    
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
                        continue # The agent may have taken 1 extra frame to request a decision
                        
                    ep_rewards[name] += reward
                    old_obs = obs_dict[name]

                    # 4. Insert memory into SB3 Replay Buffers and trigger Backward Pass (Training)
                    if "PPO" in name:
                        action, value, log_prob = sb3_actions[name]
                        ppo.rollout_buffer.add(old_obs, action, reward, done, value, log_prob)
                        if ppo.rollout_buffer.full:
                            with torch.no_grad():
                                next_obs_t = torch.tensor(next_obs).unsqueeze(0).to(ppo.device)
                                last_value = ppo.policy.predict_values(next_obs_t).cpu().numpy()[0]
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
                    
                    # 5. MLOps: Performance tracking
                    if done:
                        ep_counts[name] += 1
                        print(f"[{name}] Episode {ep_counts[name]} finished | Cumulative Reward: {ep_rewards[name]:.2f}")
                        
                        # Clean invalid characters for MLflow
                        clean_name = name.replace("?", "_").replace("=", "_").replace("-", "_")
                        mlflow.log_metric(f"{clean_name}_reward", ep_rewards[name], step=ep_counts[name])
                        
                        ep_rewards[name] = 0.0
                        
                        # Checkpoint backup to MLflow every 50 episodes
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
            print("[!] Training interrupted by user. Saving final models (ZIP and ONNX) to MLflow...")
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

if __name__ == "__main__":
    main()

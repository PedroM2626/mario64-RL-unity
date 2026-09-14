"""SB3 PPO runner for the cross-framework benchmark (shared policy).

- Mock mode: native model.learn() on a random-reward gym env (no Unity needed).
- Unity mode: manual shared-policy rollout loop against CompetitiveParkour
  (all matched behaviors driven by ONE PPO), mirroring train_competitive_sb3.py.

Hyperparams come from benchmarks/ppo_common.yaml unless overridden by CLI.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def build_ppo(common, seed, logdir=None):
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces

    class InitEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                shape=(42,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(5,), dtype=np.float32)

        def reset(self, seed=None):
            super().reset(seed=seed)
            return np.zeros(42, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(42, dtype=np.float32), 0.0, False, False, {}

    sb3_cfg = (common.get("sb3", {}) or {})
    net_arch = sb3_cfg.get("policy_kwargs", {}).get("net_arch", [256, 256])
    device = sb3_cfg.get("device", "auto")
    c = common["common"]
    model = PPO(
        sb3_cfg.get("policy", "MlpPolicy"),
        InitEnv(),
        learning_rate=c["learning_rate"],
        n_steps=c["n_steps"],
        batch_size=c["minibatch_size"],
        n_epochs=c["n_epochs"],
        gamma=c["gamma"],
        gae_lambda=c["gae_lambda"],
        clip_range=c["clip_range"],
        ent_coef=c["ent_coef"],
        vf_coef=c["vf_coef"],
        max_grad_norm=c["max_grad_norm"],
        policy_kwargs={"net_arch": net_arch},
        device=device,
        seed=seed,
        tensorboard_log=logdir,
        verbose=0,
    )
    return model


def run_mock(common, seed, mock_steps, out_dir, logdir=None):
    import gymnasium as gym
    from gymnasium import spaces
    from benchmarks.unity_shared import CsvLogger, apply_seed

    apply_seed(seed)

    class RandomRewardEnv(gym.Env):
        def __init__(self):
            super().__init__()
            # Bounds must match InitEnv in build_ppo (SB3 validates obs space on set_env).
            self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                shape=(42,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(5,), dtype=np.float32)
            self.rng = np.random.default_rng(seed)

        def reset(self, seed=None):
            super().reset(seed=seed)
            return self.rng.uniform(-1, 1, size=(42,)).astype(np.float32), {}

        def step(self, action):
            obs = self.rng.uniform(-1, 1, size=(42,)).astype(np.float32)
            return obs, float(self.rng.normal()), self.rng.random() < 0.02, False, {}

    model = build_ppo(common, seed, logdir=logdir)
    model.set_env(RandomRewardEnv())
    model.learn(total_timesteps=mock_steps)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"ppo_sb3_seed{seed}.zip"))
    logger = CsvLogger(os.path.join(out_dir, f"ppo_sb3_seed{seed}.csv"))
    logger.log("mock", 0.0, mock_steps)
    logger.close()
    print(f"[sb3] mock done: {mock_steps} timesteps, saved to {out_dir}")


def run_unity(common, seed, out_dir, env_path=None, time_scale=None,
              total_transitions=None, logdir=None, no_graphics=False,
              worker_id=0, base_port=5005):
    import torch
    from stable_baselines3.common.logger import Logger
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.base_env import ActionTuple
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    from benchmarks.unity_shared import (CsvLogger, apply_seed, match_behaviors,
                                         maybe_mlflow_run)

    apply_seed(seed)
    c = common["common"]
    budget = int(total_transitions or c["total_transitions"])
    time_scale = common["env"]["time_scale"] if time_scale is None else time_scale
    behavior_filter = common["env"].get("behavior_filter", [])

    model = build_ppo(common, seed, logdir=logdir)
    model.set_logger(Logger(folder=None, output_formats=["stdout"]))

    flat = {"framework": "sb3", "seed": seed, **{f"ppo_{k}": v for k, v in c.items()
                                                 if k not in ("seeds",)}}
    with maybe_mlflow_run(f"ppo_sb3_seed{seed}", "Mario_PPO_Benchmark", flat):
        if env_path:
            env_path = os.path.abspath(env_path)
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)
        env = UnityEnvironment(file_name=env_path, worker_id=worker_id,
                               base_port=base_port, side_channels=[channel],
                               no_graphics=no_graphics)
        env.reset()
        behaviors = match_behaviors(list(env.behavior_specs.keys()), behavior_filter)
        print(f"[sb3] behaviors driven (shared PPO): {behaviors}")

        from benchmarks.unity_shared import convert_box_action
        obs_of, ep_reward, ep_len = {}, {}, {}
        ep_start = {}
        import time as _t
        for b in behaviors:
            dec, _ = env.get_steps(b)
            obs_of[b] = dec.obs[0][0] if len(dec) > 0 else np.zeros(42, dtype=np.float32)
            ep_reward[b], ep_len[b] = 0.0, 0
            ep_start[b] = _t.time()
        last = {}
        ep_start_flag = {b: True for b in behaviors}
        logger = CsvLogger(os.path.join(out_dir, f"ppo_sb3_seed{seed}.csv"))
        step = 0
        trans = 0  # transitions consumed (budget unit, shared across frameworks)
        try:
            while trans < budget:
                act_send, sb3_act = {}, {}
                for b in behaviors:
                    dec, term = env.get_steps(b)
                    if len(term) > 0:
                        r = float(term.reward[0])
                        ep_reward[b] += r
                        ep_len[b] += 1
                        old = obs_of.get(b, term.obs[0][0])
                        if b in last:
                            act_np, val, lp = last.pop(b)
                            model.rollout_buffer.add(
                                old, act_np, r, ep_start_flag.get(b, False),
                                torch.as_tensor(val).to(model.device),
                                torch.as_tensor(lp).to(model.device))
                            trans += 1
                            ep_start_flag[b] = False
                            if model.rollout_buffer.full:
                                last_v = torch.zeros(1, device=model.device)
                                model.rollout_buffer.compute_returns_and_advantage(
                                    last_values=last_v, dones=np.array([True]))
                                model.train()
                                print(f"[sb3] PPO update at env_step={step} "
                                      f"(buffer {model.rollout_buffer.buffer_size} transitions)",
                                      flush=True)
                                model.rollout_buffer.reset()
                        logger.log(b, ep_reward[b], ep_len[b])
                        print(f"[sb3][{b}] ep reward={ep_reward[b]:.2f} len={ep_len[b]} step={step}")
                        ep_reward[b], ep_len[b] = 0.0, 0
                        ep_start_flag[b] = True
                    if len(dec) > 0:
                        new_obs = dec.obs[0][0]
                        if b in last:
                            r = float(dec.reward[0])
                            ep_reward[b] += r
                            ep_len[b] += 1
                            old = obs_of[b]
                            act_np, val, lp = last[b]
                            model.rollout_buffer.add(
                                old, act_np, r, ep_start_flag.get(b, False),
                                torch.as_tensor(val).to(model.device),
                                torch.as_tensor(lp).to(model.device))
                            trans += 1
                            ep_start_flag[b] = False
                            if model.rollout_buffer.full:
                                with torch.no_grad():
                                    nv = model.policy.predict_values(
                                        torch.as_tensor(new_obs, dtype=torch.float32)
                                        .unsqueeze(0).to(model.device)).flatten()
                                model.rollout_buffer.compute_returns_and_advantage(
                                    last_values=nv, dones=np.array([False]))
                                model.train()
                                print(f"[sb3] PPO update at env_step={step} "
                                      f"(buffer {model.rollout_buffer.buffer_size} transitions)",
                                      flush=True)
                                model.rollout_buffer.reset()
                        obs_of[b] = new_obs
                        with torch.no_grad():
                            ot = torch.as_tensor(new_obs, dtype=torch.float32).unsqueeze(0).to(model.device)
                            action, value, log_prob = model.policy.forward(ot)
                        act_np = action.cpu().numpy()[0]
                        sb3_act[b] = (act_np, value, log_prob)
                        cont, disc = convert_box_action(act_np)
                        act_send[b] = ActionTuple(continuous=cont, discrete=disc)
                for b, a in act_send.items():
                    env.set_actions(b, a)
                env.step()
                last.update(sb3_act)
                step += 1
        finally:
            os.makedirs(out_dir, exist_ok=True)
            model.save(os.path.join(out_dir, f"ppo_sb3_seed{seed}.zip"))
            logger.close()
            env.close()
    print(f"[sb3] unity done: {step} env-steps, {trans} transitions, saved to {out_dir}")

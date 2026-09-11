"""Offline + Unity evaluation for SB3 checkpoints (PPO/SAC/DQN).

Two modes:
  1) --mock (default, no Unity): smoke test — loads the .zip, runs N forward
     passes on random obs of the model's expected size, optionally checks the
     sibling .onnx exists. Verifies the checkpoint is not corrupt.
  2) --env <build> or Editor (press Play): real rollout — connects via
     UnityEnvironment, runs deterministic episodes, reports mean reward/length.

Examples:
  python evaluate.py --checkpoint models/<run>/MarioParkourPPO_final.zip --mock
  python evaluate.py --checkpoint models/<run>/MarioParkourSAC_final.zip --episodes 5
  python evaluate.py --checkpoint models/<run>/MarioParkourDQN_final.zip --episodes 5 --env ./Builds/Game.exe
"""
import argparse
import os
import sys

import numpy as np


def detect_algo(checkpoint_path, forced):
    if forced and forced != "auto":
        return forced.upper()
    name = os.path.basename(checkpoint_path).upper()
    for algo in ("PPO", "SAC", "DQN"):
        if algo in name:
            return algo
    return "PPO"


def load_model(checkpoint, algo):
    from stable_baselines3 import PPO, SAC, DQN

    cls = {"PPO": PPO, "SAC": SAC, "DQN": DQN}[algo]
    # Load without env (policy-only) for mock; env attached later for Unity eval.
    return cls.load(checkpoint)


def mock_eval(checkpoint, algo, steps=200):
    model = load_model(checkpoint, algo)
    obs_size = int(np.prod(model.observation_space.shape))
    print(f"[*] Loaded {algo} from {checkpoint} (obs_size={obs_size})")

    rng = np.random.default_rng(0)
    for i in range(steps):
        obs = rng.uniform(-1, 1, size=(obs_size,)).astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        if i == 0:
            print(f"[*] Sample action[{i}]: {np.asarray(action).flatten()[:8]}")

    # Sibling ONNX check (export is external-inference only, see script headers)
    onnx_path = os.path.splitext(checkpoint)[0] + ".onnx"
    if os.path.exists(onnx_path):
        print(f"[*] Sibling ONNX found: {onnx_path} (external inference only, not Barracuda)")
    else:
        print(f"[!] No sibling ONNX at {onnx_path} (run training to export, or ignore)")

    print(f"[✓] Mock smoke test passed ({steps} forward passes).")


def unity_eval(checkpoint, algo, env_path, episodes=5, time_scale=1.0):
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

    model = load_model(checkpoint, algo)
    obs_size = int(np.prod(model.observation_space.shape))
    print(f"[*] Loaded {algo} from {checkpoint} (obs_size={obs_size})")

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=time_scale)
    env = UnityEnvironment(file_name=env_path, side_channels=[channel])
    env.reset()
    behavior_names = list(env.behavior_specs.keys())
    print(f"[*] Behaviors: {behavior_names}")
    if not behavior_names:
        print("[!] No behaviors detected. Press Play in Unity (Editor mode) and retry.")
        env.close()
        return 1

    # Pick the behavior matching the algo substring, fallback to first.
    target = next((b for b in behavior_names if algo in b.upper()), behavior_names[0])
    print(f"[*] Evaluating on behavior: {target} (deterministic, {episodes} episodes)")

    rewards, lengths = [], []
    ep_reward, ep_len = 0.0, 0
    done_eps = 0
    obs = np.zeros(obs_size, dtype=np.float32)

    try:
        while done_eps < episodes:
            dec, term = env.get_steps(target)
            # Terminal steps: one env episode finished per terminated agent.
            # Competitive scene has 1 agent per behavior; multi-agent behaviors
            # count each terminated agent as one episode.
            for i in range(len(term)):
                ep_reward += float(term.reward[i])
                ep_len += 1
                rewards.append(ep_reward)
                lengths.append(ep_len)
                done_eps += 1
                print(f"[Ep {done_eps}] reward={ep_reward:.2f} len={ep_len}")
                ep_reward, ep_len = 0.0, 0
                if done_eps >= episodes:
                    break
            if done_eps >= episodes:
                break
            if len(dec) == 0:
                env.step()
                continue
            # Decision steps: predict per-agent action (handles N agents/behavior).
            from mlagents_envs.base_env import ActionTuple

            cont_list, disc_list = [], []
            for i in range(len(dec)):
                obs = dec.obs[0][i]
                ep_reward += float(dec.reward[i])
                ep_len += 1
                action, _ = model.predict(obs, deterministic=True)
                if algo == "DQN":
                    act = int(np.asarray(action).flatten()[0])
                    j = act % 2
                    act //= 2
                    y = act % 3
                    act //= 3
                    x = act % 3
                    cont_list.append([-1.0 + x, -1.0 + y])
                    disc_list.append([j, 0, 0])
                else:
                    a = np.asarray(action).flatten()
                    cont_list.append([a[0], a[1]])
                    disc_list.append([1 if a[2] > 0 else 0,
                                      1 if a[3] > 0 else 0,
                                      1 if a[4] > 0 else 0])
            env.set_actions(target, ActionTuple(
                continuous=np.array(cont_list, dtype=np.float32),
                discrete=np.array(disc_list, dtype=np.int32),
            ))
            env.step()
    except KeyboardInterrupt:
        print("[!] Interrupted.")
    finally:
        env.close()

    if rewards:
        print(f"[✓] Episodes={len(rewards)} mean_reward={np.mean(rewards):.2f} "
              f"mean_len={np.mean(lengths):.1f}")
        return 0
    print("[!] No episodes completed.")
    return 2


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate SB3 checkpoints (mock or Unity).")
    p.add_argument("--checkpoint", required=True, help="Path to SB3 .zip (PPO/SAC/DQN)")
    p.add_argument("--algo", default="auto", choices=["auto", "ppo", "sac", "dqn"])
    p.add_argument("--mock", action="store_true",
                   help="Smoke test without Unity (load + forward passes only)")
    p.add_argument("--env", default=None, help="Unity build path (None = Editor with Play pressed)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=200, help="Forward passes for --mock")
    p.add_argument("--time-scale", type=float, default=1.0)
    args = p.parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"[X] Checkpoint not found: {args.checkpoint}")
        return 1

    algo = detect_algo(args.checkpoint, args.algo)
    # --mock = offline smoke test. Otherwise connect to Unity:
    # --env <build> for builds, or env=None + Play pressed for Editor.
    if args.mock:
        mock_eval(args.checkpoint, algo, steps=args.steps)
        return 0
    return unity_eval(args.checkpoint, algo, args.env, episodes=args.episodes,
                      time_scale=args.time_scale)


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark evaluation: ONE protocol for all 4 PPO frameworks (SB3/CleanRL/RLlib/ML-Agents).

Design (fairness):
- The Unity player (Builds/EvalBuild, with OnnxEvalRunner in OBSERVER mode)
  owns the OFFICIAL metrics: exact goal success, completion time, min distance.
  Launch it with -evalObserve 1 -evalEpisodes N -evalOut <csv> (pass through
  --additional-args so the SAME player serves every framework).
- This script only runs policy inference (deterministic) for the chosen
  framework and drives ALL detected behaviors with that ONE shared policy —
  the same shared-policy protocol as training. Per-episode rewards/lengths
  printed here are supplementary; the Unity CSV is the verdict.

Policy loaders (all deterministic):
- sb3:      PPO/SAC/DQN .zip, predict(deterministic=True)
- cleanrl:  .pt into the VENDORED CleanRL Agent, actor_mean (their deterministic path)
- rllib:    Tuner checkpoint dir, shared policy compute_single_action(explore=False)
- mlagents: .onnx via onnxruntime (vector_observation + all-ones action_masks in;
  deterministic_continuous_actions + discrete 0/1 branches out)

Examples:
  python evaluate.py --framework sb3 --checkpoint benchmarks/runs/one_m/ppo_sb3_seed0.zip --env Builds/EvalBuild/CompetitiveParkourEval.exe --no-graphics --episodes 60 --additional-args -evalObserve 1 -evalEpisodes 60 -evalOut D:/libsm64-unity-master/benchmarks/runs/eval_sb3.csv
  (--additional-args MUST be last; everything after it goes to Unity verbatim.)
"""
import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------- policies ---
class SB3Policy:
    def __init__(self, checkpoint):
        from stable_baselines3 import PPO
        self.model = PPO.load(checkpoint)

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).flatten()


class CleanRLPolicy:
    def __init__(self, checkpoint):
        import torch
        from gymnasium import spaces

        from benchmarks._vendor.cleanrl_ppo_continuous_master import Agent

        class ShimEnvs:
            single_observation_space = spaces.Box(low=-10.0, high=10.0, shape=(42,),
                                                  dtype=np.float32)
            single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.torch = torch
        self.agent = Agent(ShimEnvs())
        self.agent.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.agent.eval()

    def __call__(self, obs):
        with self.torch.no_grad():
            mean = self.agent.actor_mean(
                self.torch.as_tensor(obs, dtype=self.torch.float32).unsqueeze(0))
        return mean.cpu().numpy()[0]


class RLLibPolicy:
    def __init__(self, checkpoint):
        from ray.rllib.algorithms.ppo import PPO as PPOAlgo

        algo = PPOAlgo.from_checkpoint(checkpoint)
        self.policy = algo.get_policy("shared")

    def __call__(self, obs):
        # Deterministic (explore=False) single action from the shared policy.
        return np.asarray(
            self.policy.compute_single_action(obs, explore=False)[0],
            dtype=np.float32).flatten()


class MLAagentsPolicy:
    def __init__(self, checkpoint):
        import onnxruntime as rt

        self.sess = rt.InferenceSession(checkpoint, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        # ML-Agents export names the obs "obs_0" (not "vector_observation").
        non_mask = [n for n in names if "mask" not in n]
        self.obs_name = next((n for n in ("vector_observation", "obs_0", "obs") if n in names),
                             sorted(non_mask)[0] if non_mask else sorted(names)[0])
        self.mask_name = next((n for n in ("action_masks", "action_mask") if n in names), None)
        outs = [(o.name, o.shape) for o in self.sess.get_outputs()]
        # Prefer deterministic_* heads when present (ML-Agents export convention).
        self.cont_name = next((n for n, s in outs if "deterministic_continuous" in n),
                              next((n for n, s in outs if "continuous" in n and "shape" not in n), None))
        self.disc_name = next((n for n, s in outs if "deterministic_discrete" in n or
                               (n.startswith("discrete") and "shape" not in n)), None)
        if self.cont_name is None:
            raise RuntimeError(f"Cannot find continuous head in {outs}")

    def __call__(self, obs):
        feed = {self.obs_name: obs.reshape(1, -1).astype(np.float32)}
        if self.mask_name is not None:
            feed[self.mask_name] = np.ones((1, 6), dtype=np.float32)
        outs = self.sess.run([self.cont_name] + ([self.disc_name] if self.disc_name else []), feed)
        cont = np.asarray(outs[0]).flatten()
        disc = (np.asarray(outs[1]).flatten() > 0.5).astype(np.float32) if len(outs) > 1 else np.zeros(3, np.float32)
        return np.concatenate([cont[:2], disc[:3]]).astype(np.float32)


def load_policy(framework, checkpoint):
    makers = {"sb3": SB3Policy, "cleanrl": CleanRLPolicy,
              "rllib": RLLibPolicy, "mlagents": MLAagentsPolicy}
    print(f"[*] Loading {framework} policy from {checkpoint}", flush=True)
    return makers[framework](checkpoint)


# ------------------------------------------------------------------ driver ---
def box5_to_unity(act):
    from mlagents_envs.base_env import ActionTuple

    a = np.asarray(act, dtype=np.float32).flatten()
    return ActionTuple(
        continuous=np.array([[a[0], a[1]]], dtype=np.float32),
        discrete=np.array([[1 if a[2] > 0 else 0,
                            1 if a[3] > 0 else 0,
                            1 if a[4] > 0 else 0]], dtype=np.int32))


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark eval: one protocol, all frameworks.")
    p.add_argument("--framework", required=True, choices=["sb3", "cleanrl", "rllib", "mlagents"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env", default=None, help="Unity build (launched with --additional-args)")
    p.add_argument("--episodes", type=int, default=60, help="Total terminal episodes across all slots")
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--no-graphics", action="store_true")
    p.add_argument("--worker-id", type=int, default=3)
    p.add_argument("--base-port", type=int, default=5045)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--additional-args", nargs=argparse.REMAINDER, default=[],
                   help="Extra CLI args forwarded to the Unity player; MUST come last, e.g. --additional-args -evalObserve 1 -evalEpisodes 60 -evalOut <csv>")
    args = p.parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print(f"[X] Checkpoint not found: {args.checkpoint}")
        return 1

    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

    policy = load_policy(args.framework, args.checkpoint)
    # Smoke the policy once before launching Unity (fail fast on bad checkpoints).
    print(f"[*] Sample action: {policy(np.zeros(42, dtype=np.float32))}", flush=True)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=args.time_scale)
    env = UnityEnvironment(file_name=args.env, worker_id=args.worker_id,
                           base_port=args.base_port, side_channels=[channel],
                           no_graphics=args.no_graphics, timeout_wait=args.timeout,
                           additional_args=args.additional_args or None)
    env.reset()
    behaviors = list(env.behavior_specs.keys())
    print(f"[*] Driving (shared {args.framework} policy): {behaviors}", flush=True)

    rewards, lengths = [], []
    ep_reward = {b: 0.0 for b in behaviors}
    ep_len = {b: 0 for b in behaviors}
    done_eps = 0
    try:
        while done_eps < args.episodes:
            send = {}
            for b in behaviors:
                dec, term = env.get_steps(b)
                for i in range(len(term)):
                    ep_reward[b] += float(term.reward[i])
                    ep_len[b] += 1
                    rewards.append(ep_reward[b])
                    lengths.append(ep_len[b])
                    done_eps += 1
                    print(f"[{b}] ep {done_eps}/{args.episodes} "
                          f"reward={ep_reward[b]:.2f} len={ep_len[b]}", flush=True)
                    ep_reward[b], ep_len[b] = 0.0, 0
                    if done_eps >= args.episodes:
                        break
                if done_eps >= args.episodes:
                    break
                if len(dec) > 0:
                    cont, disc = [], []
                    for i in range(len(dec)):
                        ep_reward[b] += float(dec.reward[i])
                        ep_len[b] += 1
                        a = policy(dec.obs[0][i].astype(np.float32))
                        at = box5_to_unity(a)
                        cont.append(at.continuous[0])
                        disc.append(at.discrete[0])
                    from mlagents_envs.base_env import ActionTuple
                    send[b] = ActionTuple(continuous=np.stack(cont), discrete=np.stack(disc))
            for b, a in send.items():
                env.set_actions(b, a)
            env.step()
    except Exception as e:
        # Unity side (observer) may quit first once IT reaches its target —
        # that severs this connection. Partial results still reported.
        print(f"[!] Stopped early ({type(e).__name__}: {e}). "
              f"Reporting {len(rewards)} episodes.", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    if rewards:
        print(f"[OK] {args.framework}: episodes={len(rewards)} "
              f"mean_reward={np.mean(rewards):.2f} mean_len={np.mean(lengths):.1f}", flush=True)
        return 0
    print("[X] No episodes completed.")
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""RLlib PPO (REAL framework) on Unity CompetitiveParkour.

- Env: UnityMultiAgentEnv (ray MultiAgentEnv) with 3 agents (one per behavior
  slot), all mapped to ONE shared PPO policy -> same shared-policy protocol as
  the SB3 / CleanRL-real runners.
- Training: native ray.rllib PPOConfig + Tuner.fit() (algo.train() loop).
  Hyperparams from benchmarks/ppo_common.yaml (lr, gamma, lambda, clip_param,
  entropy_coeff, vf_loss_coeff, train_batch_size 2048, sgd_minibatch 256,
  num_sgd_iter 3, fcnet [256,256] tanh, torch).
- Budget: stop={"timesteps_total": N} with N = transition budget (RLlib counts
  agent-steps in multi-agent).
- Episode CSV (for cross-framework compare) is written ENV-side on each agent
  termination (RLlib's native per-agent episode stats with a never-ending
  multi-agent env are awkward; our CSV is the source of truth, documented).

Requires: pip install -r requirements-benchmark.txt (ray[rllib]).
Venv notes (documented): setuptools==70.3.0 (ray 2.9 imports
pkg_resources._vendor, removed in setuptools 75) + opencv-python-headless
(ray imports atari wrappers unconditionally). Pins otherwise untouched.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class UnityMultiAgentEnv:
    """Lazy MultiAgentEnv base import so --help works without ray installed."""

    @staticmethod
    def _base():
        from ray.rllib.env.multi_agent_env import MultiAgentEnv

        return MultiAgentEnv


def make_env_class():
    from gymnasium import spaces

    Base = UnityMultiAgentEnv._base()

    class UnityMultiAgentEnvImpl(Base):
        def __init__(self, config=None):
            super().__init__()
            from mlagents_envs.environment import UnityEnvironment
            from mlagents_envs.side_channel.engine_configuration_channel import (
                EngineConfigurationChannel,
            )
            from benchmarks.unity_shared import apply_seed

            config = config or {}
            apply_seed(config.get("seed", 0))
            self._agent_ids = ["MarioParkourPPO", "MarioParkourSAC", "MarioParkourDQN"]
            self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                shape=(42,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(5,), dtype=np.float32)
            self._bname = {}  # short id -> full Unity behavior name
            channel = EngineConfigurationChannel()
            channel.set_configuration_parameters(
                time_scale=config.get("time_scale", 5.0))
            self._env = UnityEnvironment(
                file_name=config.get("exe_path"),
                worker_id=config.get("worker_id", 0),
                base_port=config.get("base_port", 5005),
                side_channels=[channel],
                no_graphics=config.get("no_graphics", True),
                timeout_wait=config.get("timeout", 300),
            )
            self._env.reset()
            specs = list(self._env.behavior_specs.keys())
            for short in self._agent_ids:
                hit = next((s for s in specs if short in s), None)
                if hit is None:
                    raise RuntimeError(f"Behavior '{short}' not in scene: {specs}")
                self._bname[short] = hit
            self._obs = {a: np.zeros(42, dtype=np.float32) for a in self._agent_ids}
            self._ep_rew = {a: 0.0 for a in self._agent_ids}
            self._ep_len = {a: 0 for a in self._agent_ids}
            # RLlib multi-agent protocol: a terminated agent stays terminated
            # (NO internal autoreset) and, crucially, is reported as done EXACTLY
            # ONCE — afterwards it is omitted from step outputs until __all__.
            # Re-emitting done=True rows would create one "trajectory" per row and
            # break RLlib postprocessing (single-trajectory batches). Physically
            # the Unity agent respawns immediately and keeps running; we keep
            # driving it with its cached last action (on-policy limbo) and drain
            # its rewards into a discard bin, so no slice leaks across logical
            # episodes. This matches RLlib's expected trajectory model exactly.
            self._alive = {a: True for a in self._agent_ids}
            self._limbo_rew = {a: 0.0 for a in self._agent_ids}
            self._last_act = {}  # last policy action per agent (limbo driving)
            self._csv = None
            if config.get("csv_path"):
                from benchmarks.unity_shared import CsvLogger

                self._csv = CsvLogger(config["csv_path"])

        def reset(self, *, seed=None, options=None):
            obs = {}
            for a in self._agent_ids:
                dec, _ = self._env.get_steps(self._bname[a])
                if len(dec) > 0:
                    self._obs[a] = dec.obs[0][0].astype(np.float32)
                self._ep_rew[a], self._ep_len[a] = 0.0, 0
                self._limbo_rew[a] = 0.0
                self._alive[a] = True
                obs[a] = self._obs[a].copy()
            return obs, {}

        def step(self, action_dict):
            from mlagents_envs.base_env import ActionTuple
            from benchmarks.unity_shared import convert_box_action

            # Guard: only send to slots currently holding a decision request
            # (ML-Agents validates set_actions shape; empty slots would raise).
            # Dead (limbo) agents are still driven with policy actions so their
            # physical behavior stays on-policy; their rewards go to the bin.
            # Dead (limbo) agents vanish from RLlib's action_dict; keep driving
            # them with their last policy action (cached) so limbo physics stays
            # on-policy instead of degrading to Unity fallback-random.
            send = dict(action_dict)
            for a in self._agent_ids:
                if a not in send and a in self._last_act:
                    send[a] = self._last_act[a]
            for a, act in send.items():
                self._last_act[a] = act
                dec, _ = self._env.get_steps(self._bname[a])
                if len(dec) == 0:
                    continue
                cont, disc = convert_box_action(act)
                self._env.set_actions(self._bname[a],
                                      ActionTuple(continuous=cont, discrete=disc))
            self._env.step()
            obs, rew, term, trunc, infos = {}, {}, {}, {}, {}
            for a in self._agent_ids:
                if not self._alive[a]:
                    # Limbo: drive silently (cached action), drain to discard bin,
                    # OMIT from outputs (already reported done).
                    dec, tm = self._env.get_steps(self._bname[a])
                    if len(tm) > 0:
                        self._limbo_rew[a] += float(tm.reward[0])
                    elif len(dec) > 0:
                        self._limbo_rew[a] += float(dec.reward[0])
                    continue
                dec, tm = self._env.get_steps(self._bname[a])
                if len(tm) > 0:
                    r = float(tm.reward[0])
                    self._ep_rew[a] += r
                    self._ep_len[a] += 1
                    self._obs[a] = tm.obs[0][0].astype(np.float32)
                    if self._csv is not None:
                        self._csv.log(a, self._ep_rew[a], self._ep_len[a])
                    print(f"[rllib][{a}] ep reward={self._ep_rew[a]:.2f} "
                          f"len={self._ep_len[a]}", flush=True)
                    self._ep_rew[a], self._ep_len[a] = 0.0, 0
                    self._alive[a] = False
                    obs[a], rew[a] = self._obs[a].copy(), r
                    term[a], trunc[a], infos[a] = True, False, {}
                elif len(dec) > 0:
                    r = float(dec.reward[0])
                    self._ep_rew[a] += r
                    self._ep_len[a] += 1
                    self._obs[a] = dec.obs[0][0].astype(np.float32)
                    obs[a], rew[a] = self._obs[a].copy(), r
                    term[a], trunc[a], infos[a] = False, False, {}
                else:  # reset gap: hold last obs, zero reward
                    obs[a], rew[a] = self._obs[a].copy(), 0.0
                    term[a], trunc[a], infos[a] = False, False, {}
            all_done = all(not self._alive[a] for a in self._agent_ids)
            term["__all__"] = all_done
            trunc["__all__"] = False
            return obs, rew, term, trunc, infos

        def close(self):
            try:
                if self._csv is not None:
                    self._csv.close()
            finally:
                try:
                    self._env.close()
                except Exception:
                    pass

    return UnityMultiAgentEnvImpl


def rllib_config(native_cfg, common, seed):
    from ray.rllib.algorithms.ppo import PPOConfig

    c = common["common"]
    r = common.get("rllib", {}) or {}
    cfg = PPOConfig()
    cfg = cfg.training(
        lr=c["learning_rate"],
        gamma=c["gamma"],
        lambda_=c["gae_lambda"],
        clip_param=c["clip_range"],
        vf_loss_coeff=c["vf_coef"],
        entropy_coeff=c["ent_coef"],
        train_batch_size=r.get("train_batch_size", 2048),
        sgd_minibatch_size=r.get("sgd_minibatch_size", c["minibatch_size"]),
        num_sgd_iter=r.get("num_sgd_iter", c["n_epochs"]),
        model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"},
    )
    cfg = cfg.framework(r.get("framework", "torch"))
    # FRAGMENT MATH (multi-agent!): one Unity env-step yields ~3 transitions
    # (one per slot). train_batch_size=2048 counts AGENT transitions, while
    # rollout_fragment_length counts ENV steps. fragment=683 env-steps ≈ 1900
    # agent-transitions ≈ train_batch → one update per ~683 env-steps, mirroring
    # SB3's shared 2048-buffer cadence. (Fragment 2048 was WRONG here: it sampled
    # ~6k/iter and trained on 2048, wasting 2/3 of data.)
    cfg = cfg.rollouts(
        num_rollout_workers=r.get("num_workers", 0),
        num_envs_per_worker=r.get("num_envs_per_worker", 1),
        rollout_fragment_length=683,
    )
    cfg = cfg.multi_agent(
        policies={"shared": (None,
                             native_cfg["obs_space"],
                             native_cfg["act_space"],
                             {})},
        policy_mapping_fn=lambda agent_id, *a, **k: "shared",
        policies_to_train=["shared"],
    )
    cfg = cfg.debugging(seed=seed)
    cfg = cfg.environment(env=native_cfg["env_cls"], env_config=native_cfg["env_config"])
    return cfg


def run_mock(common, seed, mock_steps, out_dir):
    """Smoke test of the REAL RLlib PPO on a random single-agent env (no Unity)."""
    import gymnasium as gym
    from gymnasium import spaces
    from ray.rllib.algorithms.ppo import PPOConfig
    from benchmarks.unity_shared import CsvLogger, apply_seed

    apply_seed(seed)

    class RandomEnv(gym.Env):
        def __init__(self, config=None):
            super().__init__()
            self.rng = np.random.default_rng((config or {}).get("seed", 0))
            self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                                shape=(42,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0,
                                           shape=(5,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.rng.uniform(-1, 1, size=(42,)).astype(np.float32), {}

        def step(self, action):
            obs = self.rng.uniform(-1, 1, size=(42,)).astype(np.float32)
            return obs, float(self.rng.normal()), self.rng.random() < 0.02, False, {}

    c = common["common"]
    cfg = PPOConfig()
    cfg = cfg.training(lr=c["learning_rate"], gamma=c["gamma"], lambda_=c["gae_lambda"],
                       clip_param=c["clip_range"], vf_loss_coeff=c["vf_coef"],
                       entropy_coeff=c["ent_coef"], train_batch_size=2048,
                       sgd_minibatch_size=c["minibatch_size"], num_sgd_iter=c["n_epochs"],
                       model={"fcnet_hiddens": [256, 256], "fcnet_activation": "tanh"})
    cfg = cfg.framework("torch")
    cfg = cfg.rollouts(num_rollout_workers=0, rollout_fragment_length=512)
    cfg = cfg.environment(env=RandomEnv, env_config={"seed": seed})
    cfg = cfg.debugging(seed=seed)
    algo = cfg.build()
    for _ in range(max(1, mock_steps // 2048)):
        algo.train()
    os.makedirs(out_dir, exist_ok=True)
    algo.save(os.path.join(out_dir, f"ppo_rllib_seed{seed}"))
    logger = CsvLogger(os.path.join(out_dir, f"ppo_rllib_seed{seed}.csv"))
    logger.log("mock", 0.0, mock_steps)
    logger.close()
    algo.stop()
    print(f"[rllib] mock done: saved to {out_dir}")


def run_unity(common, seed, out_dir, env_path=None, time_scale=None, total_transitions=None,
              no_graphics=True, worker_id=0, base_port=5025):
    """REAL RLlib PPO Tuner.fit() on Unity (multi-agent, one shared policy)."""
    import ray
    from ray import tune
    from gymnasium import spaces

    from benchmarks.unity_shared import apply_seed, maybe_mlflow_run

    apply_seed(seed)
    c = common["common"]
    budget = int(total_transitions or c["total_transitions"])
    time_scale = common["env"]["time_scale"] if time_scale is None else time_scale
    # Absolute paths: Ray workers must not depend on the launcher CWD.
    out_dir = os.path.abspath(out_dir)
    if env_path:
        env_path = os.path.abspath(env_path)

    env_cls = make_env_class()
    obs_space = spaces.Box(low=-10.0, high=10.0, shape=(42,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    csv_path = os.path.abspath(os.path.join(out_dir, f"ppo_rllib_seed{seed}.csv"))
    os.makedirs(out_dir, exist_ok=True)
    native_cfg = {
        "env_cls": env_cls,
        "obs_space": obs_space,
        "act_space": act_space,
        "env_config": {"exe_path": env_path, "time_scale": time_scale,
                       "worker_id": worker_id, "base_port": base_port,
                       "no_graphics": no_graphics, "seed": seed,
                       "csv_path": csv_path},
    }
    flat = {"framework": "rllib-real", "seed": seed,
            **{f"ppo_{k}": v for k, v in c.items() if k != "seeds"}}
    # STOP SEMANTICS (verified empirically: trial table showed
    # iter=986, ts=2019328=986*2048): RLlib's `timesteps_total` counts ENV steps,
    # not agent transitions. Budget is in transitions (3 slots/env-step), so stop
    # at budget//3 env-steps ≈ budget agent-transitions.
    stop_env_steps = budget // 3
    with maybe_mlflow_run(f"ppo_rllib_seed{seed}", "Mario_PPO_Benchmark", flat):
        if not ray.is_initialized():
            ray.init(num_gpus=0, include_dashboard=False,
                     _temp_dir=os.path.abspath(os.path.join(out_dir, "ray_tmp")))
        cfg = rllib_config(native_cfg, common, seed)
        tuner = tune.Tuner(
            "PPO",
            param_space=cfg.to_dict(),
            run_config=ray.air.RunConfig(
                stop={"timesteps_total": stop_env_steps},
                verbose=1,
                checkpoint_config=ray.air.CheckpointConfig(
                    checkpoint_frequency=0, checkpoint_at_end=True),
            ),
        )
        results = tuner.fit()
        # Fail LOUD on trial errors (Tuner may return partial results otherwise).
        errors = [r for r in results if getattr(r, "error", None)]
        if errors:
            raise RuntimeError(f"RLlib trial(s) errored: {[str(e.error)[:300] for e in errors]}")
        try:
            best = results.get_best_result()
            if best.checkpoint:
                import shutil

                dst = os.path.join(out_dir, f"ppo_rllib_seed{seed}_final")
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(best.checkpoint.path, dst)
                print(f"[rllib] final checkpoint -> {dst}")
        except Exception as e:
            print(f"[rllib] checkpoint export skipped ({e})")
    print(f"[rllib] unity done: stop={stop_env_steps} env-steps "
          f"(~{budget} agent-steps), csv -> {csv_path}")

"""Unity VecEnv exposing the SyncVectorEnv interface subset used by CleanRL's ppo loop.

Maps num_envs sub-envs onto behavior slots of ONE Unity CompetitiveParkour instance
(shared policy, same protocol as the SB3 benchmark runner):
- reset(seed) -> (obs [num_envs, 42], infos {})
- step(actions [num_envs, 5]) -> (obs, reward, terminations, truncations, infos)
  with infos["final_info"][i] = {"episode": {"r": R, "l": L}} on termination rows
  (RecordEpisodeStatistics-style, so CleanRL's `if "final_info" in infos` block
  works unmodified).
- single_observation_space / single_action_space attributes, close().

Unity timing note: ML-Agents decisions are async — a slot occasionally has neither
a fresh decision nor a terminal for one Unity step (agent resetting after EndEpisode).
Since CleanRL's loop stores one row per sub-env per step unconditionally, this env
sub-steps Unity (cap 10, uncounted) until every slot has fresh data; only as a last
resort it repeats the previous obs with reward 0. Sub-steps advance extra physics
but add ZERO storage rows, so the transition budget (global_step) stays exact.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class UnityVectorEnv:
    def __init__(self, exe_path, behavior_names, time_scale=5.0, worker_id=0,
                 base_port=5005, no_graphics=True, seed=0, timeout=300):
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from gymnasium import spaces

        from benchmarks.unity_shared import apply_seed

        apply_seed(seed)
        if exe_path:
            exe_path = os.path.abspath(exe_path)
        self.behavior_names = list(behavior_names)
        self.num_envs = len(behavior_names)
        self.single_observation_space = spaces.Box(low=-10.0, high=10.0,
                                                   shape=(42,), dtype=np.float32)
        self.single_action_space = spaces.Box(low=-1.0, high=1.0,
                                              shape=(5,), dtype=np.float32)
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)
        self._env = UnityEnvironment(file_name=exe_path, worker_id=worker_id,
                                     base_port=base_port, side_channels=[channel],
                                     no_graphics=no_graphics, timeout_wait=timeout)
        self._env.reset()
        # Resolve wanted behavior substrings (e.g. "MarioParkourPPO") against the
        # actual spec keys (which carry ML-Agents suffixes like "?team=0").
        specs = list(self._env.behavior_specs.keys())
        resolved = []
        for want in behavior_names:
            hit = want if want in specs else next((s for s in specs if want in s), None)
            if hit is None:
                raise RuntimeError(f"Behavior '{want}' not in scene: {specs}")
            resolved.append(hit)
        self.behavior_names = resolved
        self.num_envs = len(resolved)
        self._last_obs = [np.zeros(42, dtype=np.float32) for _ in resolved]
        self._ep_rew = [0.0 for _ in resolved]
        self._ep_len = [0 for _ in resolved]

    # -- helpers -----------------------------------------------------------
    def _convert(self, act):
        from mlagents_envs.base_env import ActionTuple

        a = np.asarray(act, dtype=np.float32).flatten()
        return ActionTuple(
            continuous=np.array([[a[0], a[1]]], dtype=np.float32),
            discrete=np.array([[1 if a[2] > 0 else 0,
                                1 if a[3] > 0 else 0,
                                1 if a[4] > 0 else 0]], dtype=np.int32),
        )

    def _drain_fresh(self, max_substeps=10):
        """Sub-step until every slot has a fresh decision or terminal (uncounted)."""
        fresh = [None] * self.num_envs
        terms = [None] * self.num_envs
        for _ in range(max_substeps):
            pending = False
            for i, b in enumerate(self.behavior_names):
                if fresh[i] is not None or terms[i] is not None:
                    continue
                dec, term = self._env.get_steps(b)
                if len(term) > 0:
                    terms[i] = term
                elif len(dec) > 0:
                    fresh[i] = dec
                else:
                    pending = True
            if not pending:
                break
            self._env.step()
        return fresh, terms

    # -- SyncVectorEnv-compatible API --------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            from benchmarks.unity_shared import apply_seed

            apply_seed(seed)
        fresh, _ = self._drain_fresh()
        obs = []
        for i in range(self.num_envs):
            if fresh[i] is not None:
                self._last_obs[i] = fresh[i].obs[0][0].astype(np.float32)
            self._ep_rew[i], self._ep_len[i] = 0.0, 0
            obs.append(self._last_obs[i].copy())
        return np.stack(obs), {}

    def step(self, actions):
        # NOTE on ML-Agents semantics: get_steps() is NON-destructive (it only reads
        # the current decision buffer; env.step() advances and refreshes it), while
        # set_actions() unconditionally overwrites the pending action consumed at the
        # agent's next decision request. So: send to ALL slots every call (no read
        # needed first), advance exactly one counted Unity step, then collect.
        # This consumes every reward slice exactly once — no loss, no double count.
        # Send ONLY to slots currently holding a decision request: ML-Agents
        # validates set_actions shape against the live agent count, so sending
        # to an empty (resetting) slot raises. The len() pre-read is safe:
        # get_steps() is non-destructive and we accumulate no rewards here.
        actions = np.asarray(actions, dtype=np.float32)
        for i, b in enumerate(self.behavior_names):
            dec, _ = self._env.get_steps(b)
            if len(dec) > 0:
                self._env.set_actions(b, self._convert(actions[i]))
        # 2) advance exactly one counted Unity step
        self._env.step()
        # 3) collect results, sub-stepping (uncounted) until every slot is fresh
        obs, rew, term, trunc, final_info = [], [], [], [], []
        fresh2, terms2 = self._drain_fresh()
        for i, b in enumerate(self.behavior_names):
            if terms2[i] is not None:
                t = terms2[i]
                r = float(t.reward[0])
                self._ep_rew[i] += r
                self._ep_len[i] += 1
                o = t.obs[0][0].astype(np.float32)
                self._last_obs[i] = o.copy()
                obs.append(o)
                rew.append(r)
                term.append(True)
                trunc.append(False)
                final_info.append({"episode": {"r": self._ep_rew[i], "l": self._ep_len[i]}})
                self._ep_rew[i], self._ep_len[i] = 0.0, 0
            elif fresh2[i] is not None:
                d = fresh2[i]
                r = float(d.reward[0])
                self._ep_rew[i] += r
                self._ep_len[i] += 1
                o = d.obs[0][0].astype(np.float32)
                self._last_obs[i] = o.copy()
                obs.append(o)
                rew.append(r)
                term.append(False)
                trunc.append(False)
                final_info.append(None)
            else:  # reset gap even after sub-steps: repeat last obs, zero reward
                obs.append(self._last_obs[i].copy())
                rew.append(0.0)
                term.append(False)
                trunc.append(False)
                final_info.append(None)
        infos = {"final_info": final_info}
        return (np.stack(obs).astype(np.float32),
                np.array(rew, dtype=np.float32),
                np.array(term, dtype=bool),
                np.array(trunc, dtype=bool),
                infos)

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

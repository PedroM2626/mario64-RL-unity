"""Shared helpers for the cross-framework PPO benchmark.

Dependency-light on import (stdlib + numpy only). torch / mlagents_envs /
mlflow / tensorboard are imported lazily inside the functions that need them,
so `--help` and mock smoke tests work on a minimal install.

Conventions (must match train_competitive_sb3.py):
- obs: Box(42,) float32
- action: Box(5,) float32 in [-1, 1]; buttons thresholded at >0:
    cont = [a0, a1], disc = [a2>0, a3>0, a4>0]
"""
import csv
import os

import numpy as np

OBS_SIZE = 42
ACTION_SIZE = 5


def convert_box_action(act):
    """Box(5) SB3/CleanRL/RLlib action -> (cont, disc) numpy pair.

    Returns (cont, disc) with shapes (1, 2) float32 and (1, 3) int32,
    mirroring train_competitive_sb3.convert_box_action without importing it
    (which would pull mlflow/sb3 at module import time).
    """
    a = np.asarray(act, dtype=np.float32).flatten()
    cont = np.array([[a[0], a[1]]], dtype=np.float32)
    disc = np.array(
        [[1 if a[2] > 0 else 0, 1 if a[3] > 0 else 0, 1 if a[4] > 0 else 0]],
        dtype=np.int32,
    )
    return cont, disc


def to_unity_action_tuple(act):
    """Box(5) action -> mlagents_envs ActionTuple (lazy import)."""
    from mlagents_envs.base_env import ActionTuple

    cont, disc = convert_box_action(act)
    return ActionTuple(continuous=cont, discrete=disc)


def match_behaviors(behavior_names, behavior_filter):
    """Select which Unity behaviors the shared policy drives."""
    if not behavior_filter:
        return list(behavior_names)
    out = []
    for name in behavior_names:
        for f in behavior_filter:
            if f.lower() in name.lower():
                out.append(name)
                break
    return out or list(behavior_names)


def load_common_config(path):
    """Load benchmarks/ppo_common.yaml (requires pyyaml, standard in this repo)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_seed(seed):
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class CsvLogger:
    """Per-episode CSV: episode, behavior, reward, length. One file per (framework, seed)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=["episode", "behavior", "reward", "length"])
        self._w.writeheader()
        self._n = 0

    def log(self, behavior, reward, length):
        self._n += 1
        self._w.writerow({"episode": self._n, "behavior": behavior,
                          "reward": f"{reward:.4f}", "length": length})
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


class MockUnityStream:
    """Random-obs stand-in for the Unity decision/terminal stream (smoke tests).

    Yields (obs, reward, done) triples so each framework's rollout + update math
    can be exercised without Unity or mlagents_envs installed.
    """

    def __init__(self, obs_size=OBS_SIZE, seed=0, done_prob=0.02):
        self.obs_size = obs_size
        self.rng = np.random.default_rng(seed)
        self.done_prob = done_prob
        self._obs = self.rng.uniform(-1, 1, size=(obs_size,)).astype(np.float32)

    def step(self):
        obs = self._obs
        reward = float(self.rng.normal(0, 1))
        done = bool(self.rng.random() < self.done_prob)
        self._obs = self.rng.uniform(-1, 1, size=(self.obs_size,)).astype(np.float32)
        return obs, reward, done


def maybe_mlflow_run(run_name, experiment, params):
    """Start an MLflow run if mlflow is installed, else a no-op context."""

    class _NoOp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def log_metric(self, *a, **k):
            pass

    try:
        import mlflow

        mlflow.set_experiment(experiment)
        run = mlflow.start_run(run_name=run_name)
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass
        return run
    except ImportError:
        print("[i] mlflow not installed — running without MLflow tracking.")
        return _NoOp()
    except Exception as e:
        print(f"[!] MLflow disabled ({e}).")
        return _NoOp()

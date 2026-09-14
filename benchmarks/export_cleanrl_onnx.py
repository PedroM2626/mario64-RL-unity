"""Export a trained CleanRL-real checkpoint (.pt) to Box(5) ONNX for in-Unity eval.

Loads the state dict saved by benchmarks/ppo_cleanrl.py into the VENDORED
CleanRL Agent (same architecture/HIDDEN) and traces ONLY the deterministic
path: obs [1,42] -> actor_mean -> action [1,5] (CleanRL deterministic eval =
mean, no sampling). Buttons thresholded (>0) by the Unity runner, exactly like
training. Run AFTER the 1M benchmark:
  venv_mlagents\\Scripts\\python.exe benchmarks/export_cleanrl_onnx.py \
    --checkpoint benchmarks/runs/one_m/ppo_cleanrl_seed0.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    import numpy as np
    import torch
    import torch.nn as nn
    from gymnasium import spaces

    from benchmarks._vendor.cleanrl_ppo_continuous_master import Agent

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="CleanRL .pt state dict")
    p.add_argument("--out", default=None, help="Output .onnx (default: sibling of checkpoint)")
    args = p.parse_args(argv)

    class ShimEnvs:
        single_observation_space = spaces.Box(low=-10.0, high=10.0, shape=(42,),
                                              dtype=np.float32)
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

    agent = Agent(ShimEnvs())
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()

    class DeterministicMean(nn.Module):
        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def forward(self, obs):
            return self.agent.actor_mean(obs)

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".onnx"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    # NOTE: NO dynamic_axes — fixed batch=1 (see train_competitive_sb3.export_onnx:
    # symbolic batch dims break Barracuda with all-zero outputs).
    torch.onnx.export(
        DeterministicMean(agent),
        torch.randn(1, 42),
        out,
        opset_version=11,
        input_names=["vector_observation"],
        output_names=["action"],
    )
    print(f"[OK] CleanRL ONNX (actor_mean, in-Unity eval ready) -> {out}")


if __name__ == "__main__":
    main()

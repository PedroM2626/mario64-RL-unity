"""Export a trained SB3 PPO checkpoint to Box(5) ONNX for in-Unity eval.

Reuses train_competitive_sb3.export_onnx (single [1,42] in -> [1,5] out,
deterministic). Run AFTER the 1M benchmark:
  venv_mlagents\\Scripts\\python.exe benchmarks/export_sb3_onnx.py \
    --checkpoint benchmarks/runs/one_m/ppo_sb3_seed0.zip
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="SB3 .zip")
    p.add_argument("--out", default=None, help="Output .onnx (default: sibling of checkpoint)")
    args = p.parse_args(argv)

    from stable_baselines3 import PPO
    from train_competitive_sb3 import export_onnx

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".onnx"
    model = PPO.load(args.checkpoint)
    export_onnx(model, out)
    print(f"[OK] SB3 ONNX (Box5, in-Unity eval ready) -> {out}")


if __name__ == "__main__":
    main()

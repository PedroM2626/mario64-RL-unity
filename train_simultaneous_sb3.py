"""Deprecated shim — use train_competitive_sb3.py instead.

train_simultaneous_sb3.py was a ~90% duplicate of train_competitive_sb3.py with
an older loop (send-then-get, no TensorBoard writer, no --time-scale/--tb-logdir,
fragile PPO rollout handling). It is kept only so old commands/scripts don't break.

Canonical: python train_competitive_sb3.py --time-scale 3.0 --tb-logdir ./tensorboard_logs
"""
import sys
import warnings

warnings.warn(
    "train_simultaneous_sb3.py is deprecated; delegating to train_competitive_sb3.main(). "
    "Use 'python train_competitive_sb3.py' directly.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from train_competitive_sb3 import main
except ImportError:  # fallback when run as script from another cwd
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_competitive_sb3 import main

if __name__ == "__main__":
    sys.argv[0] = "train_competitive_sb3.py"
    main()

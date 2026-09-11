"""Deprecated shim — use train_mlops.py instead.

trainer_mlflow.py duplicated train_mlops.py (both wrap `mlagents-learn` with MLflow).
Canonical implementation lives in train_mlops.py.
"""
import sys
import warnings

warnings.warn(
    "trainer_mlflow.py is deprecated; delegating to train_mlops.main(). "
    "Use 'python train_mlops.py' directly.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    from train_mlops import main
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_mlops import main

if __name__ == "__main__":
    sys.argv[0] = "train_mlops.py"
    main()

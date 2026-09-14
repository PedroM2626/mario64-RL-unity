"""Compare PPO benchmark runs (CSVITLE).

Reads benchmarks/runs/<run>/*_seed<S>.csv files (written by every runner) and
prints a per-framework summary + saves a reward-curve plot (matplotlib).

Usage:
  python benchmarks/compare_ppo.py --dir benchmarks/runs/v1 --out benchmarks/runs/v1/compare.png
  python benchmarks/compare_ppo.py --dir benchmarks/runs/v1  # table only
"""
import argparse
import csv
import glob
import os
import re


def load_series(path):
    eps, rewards, trans = [], [], []
    cum = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                r = float(row["reward"])
                ln = int(float(row.get("length", 1)))
            except (KeyError, ValueError):
                continue
            cum += ln
            eps.append(len(eps) + 1)
            rewards.append(r)
            trans.append(cum)
    return eps, rewards, trans


def moving_avg(xs, w=20):
    w = max(1, min(w, len(xs)))
    out = []
    for i in range(len(xs)):
        out.append(sum(xs[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--window", type=int, default=20)
    args = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    # Exclude forensic appendices (e.g. *_overshoot.csv) from the official plot.
    files = [f for f in files if "overshoot" not in os.path.basename(f)]
    if not files:
        print(f"[X] No CSVs in {args.dir}")
        return 1
    print(f"{'framework':<12}{'seed':<6}{'episodes':<10}{'mean_r':<10}{'last50_r':<10}file")
    series = {}
    for fp in files:
        base = os.path.basename(fp)
        m = re.match(r"ppo_(\w+)_seed(\d+)\.csv", base)
        fw, seed = (m.group(1), m.group(2)) if m else ("?", "?")
        eps, rew, trans = load_series(fp)
        if not rew:
            print(f"{fw:<12}{seed:<6}{0:<10}{'-':<10}{'-':<10}{base} (empty/mock)")
            continue
        mean_r = sum(rew) / len(rew)
        last = sum(rew[-50:]) / len(rew[-50:])
        print(f"{fw:<12}{seed:<6}{len(rew):<10}{mean_r:<10.2f}{last:<10.2f}{base}")
        series.setdefault(fw, []).append((seed, trans, rew))

    if args.out and series:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not installed — table only.")
            return 0
        plt.figure(figsize=(11, 6))
        for fw, runs in sorted(series.items()):
            # X = transitions (cumulative budget unit); aligns per-episode rows
            # with TB summary points (mlagents) on the same axis.
            for seed, trans, rew in runs:
                plt.plot(trans, moving_avg(rew, args.window), alpha=0.35,
                         label=f"{fw} s{seed}" if len(runs) == 1 else None)
            min_len = min(len(r) for _, _, r in runs)
            mean_curve = [sum(r[i] for _, _, r in runs) / len(runs) for i in range(min_len)]
            min_x = min(t[:min_len] for _, t, _ in runs)
            plt.plot(min_x, moving_avg(mean_curve, args.window),
                     linewidth=2.5, label=f"{fw} mean({len(runs)})")
        plt.xlabel("transitions (agent-steps, budget unit)")
        plt.ylabel(f"reward (moving avg w={args.window})")
        plt.title("PPO benchmark: ML-Agents vs SB3 vs RLlib vs CleanRL-real (seed 0, 3M transitions)")
        plt.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plt.savefig(args.out, dpi=120)
        print(f"[OK] Plot saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Convert ML-Agents TensorBoard curves to the benchmark's per-row CSV format.

ML-Agents logs summaries (not per-episode rows): we extract
  Environment/Cumulative Reward  -> reward column (running mean over episodes)
  Environment/Episode Length     -> length column (nearest-step aligned)
one row per summary point, behavior=MarioParkourPPO (the PPO slot).

This is a SMOOTHED curve (summary_freq=2000 steps), so compare it against the
moving-average curves of the other frameworks, not raw per-episode points.
Table stats (mean/last50) computed over these points approximate the same
quantities. SAC/DQN-slot series are left in TB (same file) for inspection.
"""
import glob
import os


def extract(run_dir, behavior="MarioParkourPPO", out_csv=None, summary_freq=2000, n_behaviors=3):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    pattern = os.path.join(run_dir, behavior, "events.out.tfevents.*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TB events for {behavior} in {run_dir}")
    acc = EventAccumulator(files[0], size_guidance={"scalars": 0})
    acc.Reload()
    rewards = {s.step: s.value for s in acc.Scalars("Environment/Cumulative Reward")}
    steps = sorted(rewards)
    rows = []
    # NOTE: each TB row summarizes a summary_freq window, NOT one episode.
    # For axis parity with per-episode CSVs, length = window transitions
    # (summary_freq agent-steps/behavior x n_behaviors), so cumsum(length)
    # reconstructs the transition budget. Table means over these rows are
    # windowed means, not episode means — documented, not hidden.
    window_trans = summary_freq * n_behaviors
    for i, st in enumerate(steps):
        rows.append((i + 1, behavior, rewards[st], window_trans, st))
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("episode,behavior,reward,length\n")
            for i, b, r, ln, _ in rows:
                f.write(f"{i},{b},{r:.4f},{ln}\n")
    return rows


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="results/<run-id>")
    p.add_argument("--behavior", default="MarioParkourPPO")
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args(argv)
    rows = extract(args.run_dir, args.behavior, args.out)
    print(f"[OK] {len(rows)} TB points -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())

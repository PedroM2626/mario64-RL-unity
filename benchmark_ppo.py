"""Cross-framework PPO benchmark dispatcher (ML-Agents vs SB3 vs RLlib vs CleanRL).

All frameworks use the SAME nominal hyperparameters from benchmarks/ppo_common.yaml.
Scene: CompetitiveParkour (42 obs, Box(5) action, shared policy where applicable).
Budget unit: TRANSITIONS (consumed agent-steps), identical for every runner.
3M transitions ~= 1M Unity env-steps x 3 behavior slots.

Examples:
  python benchmark_ppo.py --framework sb3 --mock --mock-steps 512 --out benchmarks/runs/smoke
  python benchmark_ppo.py --framework cleanrl --mock --mock-steps 2048 --out benchmarks/runs/smoke
  python benchmark_ppo.py --framework sb3 --seed 0 --transitions 3000000 --out benchmarks/runs/v1 --env Builds/CompetitiveHeadless/CompetitiveParkour.exe --no-graphics --worker-id 0 --base-port 5005
  python benchmark_ppo.py --framework cleanrl --seed 0 --transitions 3000000 --out benchmarks/runs/v1 --env Builds/CompetitiveHeadless/CompetitiveParkour.exe --no-graphics --worker-id 1 --base-port 5015
  python benchmark_ppo.py --framework rllib --seed 0 --transitions 3000000 --out benchmarks/runs/v1 --env Builds/CompetitiveHeadless/CompetitiveParkour.exe --no-graphics --worker-id 2 --base-port 5025
  python benchmark_ppo.py --framework mlagents --out benchmarks/runs/v1 --run-id ppo_bench --transitions 3000000 --env Builds/CompetitiveHeadless/CompetitiveParkour.exe --no-graphics --base-port 5035
  python benchmark_ppo.py --list  # show hyperparams + mapping
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.unity_shared import load_common_config

DEFAULT_CONFIG = os.path.join("benchmarks", "ppo_common.yaml")
FRAMEWORKS = ("sb3", "cleanrl", "rllib", "mlagents")
DEFAULT_PORTS = {"sb3": 5005, "cleanrl": 5015, "rllib": 5025, "mlagents": 5035}


def show_config(common):
    c = common["common"]
    print("Canonical PPO hyperparams (benchmarks/ppo_common.yaml [common]):")
    for k in ("learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
              "vf_coef", "max_grad_norm", "n_steps", "minibatch_size", "n_epochs",
              "total_transitions", "seeds"):
        print(f"  {k}: {c[k]}")
    print("Mapping: sb3(clip_range,n_steps,batch_size,n_epochs) | "
          "rllib(clip_param,train_batch,sgd_minibatch,num_sgd_iter) | "
          "cleanrl(clip_coef,num_steps,num_minibatches,update_epochs) | "
          "mlagents(epsilon,buffer_size,batch_size,num_epoch; beta=ent_coef; schedule constant).")


def main(argv=None):
    p = argparse.ArgumentParser(description="PPO cross-framework benchmark.")
    p.add_argument("--framework", choices=FRAMEWORKS + ("all",), default="sb3")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transitions", type=int, default=None,
                   help="Transition (agent-step) budget (default: config total_transitions)")
    p.add_argument("--mock", action="store_true", help="Smoke test without Unity")
    p.add_argument("--mock-steps", type=int, default=512)
    p.add_argument("--out", default=os.path.join("benchmarks", "runs", "bench"))
    p.add_argument("--run-id", default="ppo_bench")
    p.add_argument("--env", default=None, help="Unity build path (None = Editor + Play)")
    p.add_argument("--no-graphics", action="store_true",
                   help="Launch player builds with -nographics (faster; needs a player build, not Editor)")
    p.add_argument("--time-scale", type=float, default=None)
    p.add_argument("--worker-id", type=int, default=0,
                   help="Unity worker id (distinct per parallel run)")
    p.add_argument("--base-port", type=int, default=None,
                   help="Unity base port (default per framework: sb3 5005, cleanrl 5015, rllib 5025, mlagents 5035)")
    p.add_argument("--list", action="store_true", help="Print canonical hyperparams and exit")
    p.add_argument("--tb-logdir", default=None, help="TensorBoard root (sb3 only)")
    args = p.parse_args(argv)

    common = load_common_config(args.config)
    if args.list:
        show_config(common)
        return 0

    frameworks = list(FRAMEWORKS) if args.framework == "all" else [args.framework]
    rc = 0
    for fw in frameworks:
        base_port = args.base_port if args.base_port is not None else DEFAULT_PORTS[fw]
        try:
            if fw == "sb3":
                from benchmarks import ppo_sb3
                if args.mock:
                    ppo_sb3.run_mock(common, args.seed, args.mock_steps, args.out, logdir=args.tb_logdir)
                else:
                    ppo_sb3.run_unity(common, args.seed, args.out, env_path=args.env,
                                      time_scale=args.time_scale, total_transitions=args.transitions,
                                      logdir=args.tb_logdir, no_graphics=args.no_graphics,
                                      worker_id=args.worker_id, base_port=base_port)
            elif fw == "cleanrl":
                from benchmarks import ppo_cleanrl
                if args.mock:
                    ppo_cleanrl.run_mock(common, args.seed, args.mock_steps, args.out)
                else:
                    ppo_cleanrl.run_unity(common, args.seed, args.out, env_path=args.env,
                                          time_scale=args.time_scale, total_transitions=args.transitions,
                                          no_graphics=args.no_graphics,
                                          worker_id=args.worker_id, base_port=base_port)
            elif fw == "rllib":
                from benchmarks import ppo_rllib
                if args.mock:
                    ppo_rllib.run_mock(common, args.seed, args.mock_steps, args.out)
                else:
                    ppo_rllib.run_unity(common, args.seed, args.out, env_path=args.env,
                                        time_scale=args.time_scale, total_transitions=args.transitions,
                                        no_graphics=args.no_graphics,
                                        worker_id=args.worker_id, base_port=base_port)
            elif fw == "mlagents":
                from benchmarks import ppo_mlagents
                budget = int(args.transitions or common["common"]["total_transitions"])
                if args.mock:
                    print("[mlagents] mock N/A (config generator only) — YAML still generated for inspection.")
                ppo_mlagents.run(common, args.out, args.run_id,
                                 max_steps_per_behavior=budget // 3,
                                 base_port=base_port, env_path=args.env,
                                 no_graphics=args.no_graphics, seed=args.seed)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[{fw}] FAILED: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

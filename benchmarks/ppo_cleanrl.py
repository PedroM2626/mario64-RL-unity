"""CleanRL PPO (REAL algorithm) on Unity CompetitiveParkour.

What is genuinely CleanRL here:
- Agent architecture + layer_init + get_action_and_value: imported from
  benchmarks/_vendor/cleanrl_ppo_continuous_master.py (byte-faithful vendor of
  upstream ppo_continuous_action.py; sole numeric change HIDDEN 64->256).
- Rollout storage layout [num_steps, num_envs], GAE loop, minibatch SGD loop,
  clipped surrogate + clipped value loss, entropy bonus, grad clipping,
  approx_kl/clipfrac/explained_var logging: replicated VERBATIM from upstream
  __main__ (see UNITY-SWAP marks for the only differing lines).
- Seeding block verbatim. Adam(eps=1e-5) verbatim.

UNITY-SWAP lines (compatibility only, no algorithmic change):
- env: UnityVectorEnv (3 behavior slots as num_envs=3) instead of SyncVectorEnv.
- hyperparams: values from benchmarks/ppo_common.yaml
  (lr 3e-4, ent 0.01, num_minibatches 8, update_epochs 3, anneal_lr False,
  num_steps 2048, gamma/lambda/clip/vf/maxgrad native names).
- TB writer path -> out_dir; model save -> ppo_cleanrl_seed{S}.pt.
- ADDITIVE: CsvLogger rows (for cross-framework compare) + MLflow params.

Budget: total TRANSITIONS (agent-steps). 3 slots x 1M env-steps equivalent.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _fill_args(common, seed, total_transitions):
    from benchmarks._vendor.cleanrl_ppo_continuous_master import Args

    c = common["common"]
    clean = common.get("cleanrl", {}) or {}
    args = Args()
    args.seed = seed
    args.torch_deterministic = True
    args.cuda = True
    args.track = False
    args.capture_video = False
    args.save_model = True
    args.total_timesteps = int(total_transitions)
    args.learning_rate = c["learning_rate"]
    args.num_envs = 3
    args.num_steps = c["n_steps"]
    args.anneal_lr = False  # constant LR like SB3/RLlib/ML-Agents-constant
    args.gamma = c["gamma"]
    args.gae_lambda = c["gae_lambda"]
    args.num_minibatches = max(1, (args.num_envs * args.num_steps) // c["minibatch_size"])
    args.update_epochs = c["n_epochs"]
    args.norm_adv = bool(clean.get("norm_adv", True))
    args.clip_coef = c["clip_range"]
    args.clip_vloss = bool(clean.get("clip_vloss", True))
    args.ent_coef = c["ent_coef"]
    args.vf_coef = c["vf_coef"]
    args.max_grad_norm = c["max_grad_norm"]
    args.target_kl = clean.get("target_kl", None)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def _run_loop(args, envs, run_name, out_dir, seed, framework_tag="ppo_cleanrl"):
    """Upstream __main__ training loop, verbatim except UNITY-SWAP lines."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from benchmarks._vendor.cleanrl_ppo_continuous_master import Agent
    from benchmarks.unity_shared import CsvLogger, maybe_mlflow_run

    # UNITY-SWAP: writer under out_dir (upstream: runs/{run_name})
    writer = SummaryWriter(os.path.join(out_dir, f"tb_{framework_tag}_seed{seed}"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    csv_logger = CsvLogger(os.path.join(out_dir, f"{framework_tag}_seed{seed}.csv"))
    flat = {"framework": "cleanrl-real", "seed": seed,
            **{f"ppo_{k}": v for k, v in vars(args).items()
               if k not in ("batch_size", "minibatch_size", "num_iterations")}}

    with maybe_mlflow_run(f"{framework_tag}_seed{seed}", "Mario_PPO_Benchmark", flat):
        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        # UNITY-SWAP: envs passed in (UnityVectorEnv); upstream builds SyncVectorEnv here.
        assert isinstance(envs.single_action_space.shape, tuple), "Box action space required"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for i, info in enumerate(infos["final_info"]):
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                            # UNITY-SWAP (additive): unified CSV for cross-framework compare
                            csv_logger.log(f"slot{i}", info["episode"]["r"], info["episode"]["l"])

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            print(f"[cleanrl] update {iteration}/{args.num_iterations} "
                  f"pg_loss={pg_loss.item():.4f} v_loss={v_loss.item():.4f} "
                  f"global_step={global_step}", flush=True)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # UNITY-SWAP: save .pt checkpoint (upstream: optional .cleanrl_model + eval/upload)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(agent.state_dict(), os.path.join(out_dir, f"{framework_tag}_seed{seed}.pt"))
        print(f"[cleanrl] done: global_step={global_step}, saved to {out_dir}")

        envs.close()
        writer.close()
        csv_logger.close()


def run_mock(common, seed, mock_steps, out_dir):
    """Smoke test of the REAL CleanRL loop on a random VecEnv (no Unity)."""
    import gymnasium as gym

    c = common["common"]
    total = max(6144, (mock_steps // 6144) * 6144)

    class RandomVecEnv:
        single_observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(42,), dtype=np.float32)
        single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        def __init__(self):
            self.rng = np.random.default_rng(seed)

        def reset(self, seed=None):
            return self.rng.uniform(-1, 1, size=(3, 42)).astype(np.float32), {}

        def step(self, actions):
            obs = self.rng.uniform(-1, 1, size=(3, 42)).astype(np.float32)
            rew = self.rng.normal(size=(3,)).astype(np.float32)
            done = self.rng.random(3) < 0.02
            infos = {"final_info": [
                {"episode": {"r": float(r), "l": 10}} if d else None
                for r, d in zip(rew, done)]}
            return obs, rew, done, np.zeros(3, dtype=bool), infos

        def close(self):
            pass

    args = _fill_args(common, seed, total)
    _run_loop(args, RandomVecEnv(), f"mock_cleanrl_{seed}", out_dir, seed)


def run_unity(common, seed, out_dir, env_path=None, time_scale=None, total_transitions=None,
              no_graphics=True, worker_id=0, base_port=5015):
    """REAL CleanRL PPO against Unity (3 behavior slots as num_envs=3)."""
    from benchmarks.unity_vec_env import UnityVectorEnv

    c = common["common"]
    total_transitions = int(total_transitions or c["total_transitions"])
    time_scale = common["env"]["time_scale"] if time_scale is None else time_scale
    args = _fill_args(common, seed, total_transitions)
    behaviors = ["MarioParkourPPO", "MarioParkourSAC", "MarioParkourDQN"]
    envs = UnityVectorEnv(env_path, behaviors, time_scale=time_scale,
                          worker_id=worker_id, base_port=base_port,
                          no_graphics=no_graphics, seed=seed)
    _run_loop(args, envs, f"unity_cleanrl_{seed}", out_dir, seed)

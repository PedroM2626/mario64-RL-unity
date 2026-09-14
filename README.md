# libsm64-unity-dev - Parkour RL Training System

A complete system for training AI agents in Super Mario 64 parkour environments using **Reinforcement Learning** (model-free and model-based), powered by [libsm64](https://github.com/libsm64/libsm64-unity) and Unity ML-Agents.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training Systems](#training-systems)
  - [Parkour RL (Traditional)](#1-parkour-rl-traditional)
  - [Competitive Parkour (PPO, SAC, DQN)](#2-competitive-parkour-ppo-sac-dqn)
  - [Dreamer Parkour (World Model RL)](#3-dreamer-parkour-world-model-rl)
  - [Team Battle](#4-team-battle)
  - [Chase Training (Pursuer vs Fugitive)](#5-chase-training-pursuer-vs-fugitive)
- [PPO Benchmark](#ppo-benchmark-ml-agents-vs-sb3-vs-rllib-vs-cleanrl)
- [Available Scenes](#available-scenes)
- [Usage Instructions](#usage-instructions)
- [MLOps Integration](#mlops-integration)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Docker](#docker)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Features

- **Multi-Algorithm Competitive Training**: Simultaneously train PPO, SAC, and DQN agents on the same parkour map (via Stable-Baselines3 bridge; ML-Agents YAML natively supports PPO/SAC/POCA only)
- **DreamerV3 World Model RL**: Model-based RL with RSSM, actor-critic, and imagined trajectories
- **Parallel Environment Support**: Train with up to 4 parallel Mario agents
- **MLOps Integration**: Full experiment tracking via MLflow with TensorBoard visualization
- **ONNX Export**: Export trained SB3 models for external inference (see limitations in each script header)
- **Curriculum Learning**: Progressive difficulty with win-rate-based advancement (ParkourTraining scenes only — Competitive/Dreamer/Team/Chase do not read `spawn_lesson`)
- **Team Battle Mode**: Multi-agent cooperative/competitive training with MA-POCA
- **Chase Training**: Pursuer vs Fugitive SAC-based training
- **Docker Support**: Containerized training environment

---

## Project Architecture

```
libsm64-unity-dev/
├── Assets/
│   ├── ParkourRL/
│   │   ├── Scripts/
│   │   │   ├── BackupSystem/
│   │   │   │   ├── BackupParkourEnvironment.cs     [Legacy OldSystem]
│   │   │   │   ├── BackupMarioRLAgent.cs
│   │   │   │   └── BackupCheckpoint.cs
│   │   │   ├── ParkourEnvironment.cs               [Traditional RL]
│   │   │   ├── MarioRLAgent.cs
│   │   │   ├── MarioInputProvider.cs
│   │   │   ├── PlayerMarioSpawner.cs
│   │   │   ├── CompetitiveParkourEnvironment.cs    [Competitive mode]
│   │   │   ├── MarioCompetitiveAgent.cs
│   │   │   ├── DreamerParkourEnvironment.cs        [DreamerV3 mode]
│   │   │   ├── MarioDreamerAgent.cs
│   │   │   ├── ChaseTrainingEnvironment.cs         [Chase mode]
│   │   │   ├── ChaseAgent.cs
│   │   │   ├── TeamBattleEnvironment.cs            [Team mode]
│   │   │   └── TeamBattleAgent.cs
│   │   ├── Scenes/
│   │   │   ├── ParkourTraining.unity               [Standard RL]
│   │   │   ├── ParkourTraining_OldSystem.unity
│   │   │   ├── CompetitiveParkour.unity
│   │   │   ├── DreamerParkour.unity                [DreamerV3 RL]
│   │   │   ├── ChaseTraining.unity
│   │   │   └── TeamBattle.unity
│   │   ├── Config/
│   │   │   └── mario_parkour.yaml                  [RL Config]
│   │   └── Models/
│   └── libsm64-unity/                              [Native wrapper]
│
├── config/
│   ├── chase_training_sac.yaml                     [Chase SAC config]
│   └── team_battle_poca.yaml                       [Team Battle config]
│
├── models/                                         [Trained models]
├── results/                                        [Training results]
├── tensorboard_logs/                               [TensorBoard logs]
│
├── train_competitive_sb3.py                        [SB3 competitive training w/ TensorBoard]
├── train_simultaneous_sb3.py                       [Deprecated shim -> train_competitive_sb3.py]
├── train_dreamer.py                                [DreamerV3 world model training]
├── train_dreamer_multiagent.ps1                    [Multi-agent Dreamer training]
├── train_mlops.py                                  [MLOps wrapper (canonical)]
├── trainer_mlflow.py                               [Deprecated shim -> train_mlops.py]
├── evaluate.py                                     [Offline evaluation of checkpoints]
├── benchmark_ppo.py                                [PPO benchmark dispatcher]
│
├── benchmarks/
│   ├── ppo_common.yaml                             [Canonical PPO hyperparams]
│   ├── unity_shared.py                             [Shared converters/CSV/mock]
│   ├── ppo_sb3.py / ppo_cleanrl.py                 [SB3 / CleanRL-real runners]
│   ├── ppo_rllib.py (optional ray)                 [RLlib runner]
│   ├── ppo_mlagents.py                             [ML-Agents YAML generator]
│   └── compare_ppo.py                              [CSV table + plot]
│
├── train_mario.ps1 / .bat                          [Standard training scripts]
├── train_chase.ps1                                 [Chase training script]
├── train_mario_fast.bat                            [Fast training script]
├── setup_training_env.ps1                          [Environment setup]
├── setup_python38.ps1 / setup_python39.ps1         [Python installers]
│
├── requirements.txt                                [Python dependencies, pinned]
├── requirements-benchmark.txt                      [Optional: ray[rllib] for benchmark]
├── Dockerfile                                      [Docker support]
└── README.md
```

---

## Prerequisites

- **Unity**: 2019.3.10 or higher
- **Python**: 3.8 or 3.9 (3.9 recommended)
- **ROM**: Super Mario 64 US (MD5: `20b854b239203baf6c961b850a4a51a2`)

---

## Installation

1. Clone with submodules:
    ```bash
    git clone https://github.com/libsm64/libsm64-unity-dev
    cd libsm64-unity-dev
    git submodule update --init --recursive
    ```

2. Place the ROM `baserom.us.z64` in the project root:
    ```
    libsm64-unity-dev/
    ├── baserom.us.z64    <-- Place here
    ├── Assets/
    └── ...
    ```

3. Open the project in Unity

4. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

5. (Optional) Configure virtual environment:
    ```powershell
    # PowerShell
    .\setup_python39.ps1  # or setup_python38.ps1
    .\setup_training_env.ps1
    ```

---

## Training Systems

### 1. **Parkour RL (Traditional)**

Training with **4 parallel Marios** using PPO.

**Components:**
- `ParkourEnvironment.cs` - Orchestrator
- `MarioRLAgent.cs` - ML-Agents agent
- `MarioInputProvider.cs` - Input control
- `BackupMarioRLAgent.cs` - Simplified version

**Quick Start:**
```powershell
# Terminal 1: Start trainer
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2: Play the ParkourTraining.unity scene
```

**Observations (30-dim):**
| Field | Indices | Values |
|-------|---------|--------|
| Position (x,y,z) | 0-2 | [-1, 1] |
| Direction to goal (x,y,z,dist) | 3-6 | [-1, 1] |
| Velocity (x,y,z) | 7-9 | [-1, 1] |
| Airborne? | 10 | {0, 1} |
| 8-direction raycasts | 11-26 | [0, 1] |
| Downward raycast | 27 | [0, 1] |
| Jumping? | 28 | {0, 1} |
| Time / 30s | 29 | [0, 1] |

**Actions:**
- `[0]` Joystick X: [-1, 1] (left/right)
- `[1]` Joystick Y: [-1, 1] (forward/back)
- `[2]` Jump: {0, 1} (discrete)

**Training Progress:**
- Step 0-100k: Mario learns to move
- Step 100k-500k: Mario attempts platform jumps
- Step 500k+: Mario can complete the parkour

**Curriculum:** `mario_parkour.yaml` / `mario_parkour_fast.yaml` define `spawn_lesson` (Lesson0..Phase8), consumed only by `ParkourEnvironment.cs` + `MarioRLAgent.cs` via `Academy.EnvironmentParameters`. Competitive/Dreamer/TeamBattle/Chase scenes ignore `spawn_lesson` — they always spawn at fixed points.

---

### 2. **Competitive Parkour (PPO, SAC, DQN)**

Three models (PPO, SAC, DQN) compete simultaneously to reach the goal first via the Stable-Baselines3 bridge (`train_competitive_sb3.py`). The environment is dedicated (does not reuse the old parkour) and does not use parallel training.

> Note: ML-Agents natively supports only PPO/SAC/POCA. DQN here is a true SB3-DQN running through the low-level `UnityEnvironment` bridge with a discretized 18-action space (joystick X/Y + Jump). Kick/Stomp are PPO/SAC-only. See `Assets/ParkourRL/Config/mario_parkour.yaml` for the ML-Agents PPO/SAC configs.

- Scene: `CompetitiveParkour.unity`
- Script: `CompetitiveParkourEnvironment.cs`
- Agent: `MarioCompetitiveAgent.cs`

**Features:**
- 3 configurable Spawn Points in the Inspector
- Custom colors and textures per model (exposed in Inspector: `ppoColor`, `sacColor`, `dqnColor`)
- On-screen victory HUD (UI) with per-model win counts
- Runtime-generated materials based on configured color

**Observations (42-dim):**
| Field | Indices | Values |
|-------|---------|--------|
| Position (x,y,z) | 0-2 | [-1, 1] |
| Direction to goal (x,y,z,dist) | 3-6 | [-1, 1] |
| Velocity (x,y,z) | 7-9 | [-1, 1] |
| Airborne? | 10 | {0, 1} |
| 8-direction raycasts | 11-26 | [0, 1] |
| Downward raycast | 27 | [0, 1] |
| Time / 45s | 28 | [0, 1] |
| Relative ranking | 29 | [0, 1] |
| Rivals (3x4 obs) | 30-41 | [-1, 1] |

**Actions:**
- `[0-1]` Joystick X/Y: [-1, 1] (continuous)
- `[2]` Jump: {0, 1} (discrete)
- `[3]` Kick/Punch: {0, 1} (discrete)
- `[4]` Slide/Stomp: {0, 1} (discrete)

**Training:**
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. In Unity, open the CompetitiveParkour.unity scene and press Play

# 3. Run training with TensorBoard and MLflow
python train_competitive_sb3.py --time-scale 3.0 --tb-logdir ./tensorboard_logs

# 4. Monitor metrics in real time (TensorBoard)
tensorboard --logdir ./tensorboard_logs
# Access in browser: http://localhost:6006

# 5. Monitor models and parameters (MLflow)
mlflow ui
# Access in browser: http://localhost:5000
```

**Script Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Path to Unity executable (empty for Editor) | `None` |
| `--run-id` | Run ID | timestamp |
| `--resume` | Resume from saved checkpoints | `false` |
| `--force` | Overwrite previous runs | `false` |
| `--tb-logdir` | TensorBoard log directory | `./tensorboard_logs` |
| `--time-scale` | Unity time scale | `3.0` |

The script creates a `CompetitiveParkourEnv` (gym.Env) dedicated to this scene. It logs to MLflow: reward per episode, episode length, checkpoints every 50 episodes, and final models in ONNX.

---

### 3. **Dreamer Parkour (World Model RL)**

Treinamento com **DreamerV3** - algoritmo de Model-Based RL que aprende um modelo do mundo e planeja no espaço latente.

**Características:**
- **RSSM** (Recurrent State-Space Model) - Modelo dinâmico do ambiente
- **Actor-Critic** com learned dynamics
- Predição de recompensas e terminações
- Treino eficiente em dados (menos interações necessárias)

**Arquivos:**
- Scene: `Assets/ParkourRL/Scenes/DreamerParkour.unity`
- Script: `DreamerParkourEnvironment.cs`
- Agent: `MarioDreamerAgent.cs`
- Trainer: `train_dreamer.py`

**Observations (47-dim):**
| Field | Indices | Description |
|-------|---------|-------------|
| Position (x,y,z) | 0-2 | Normalized position |
| Direction to goal | 3-6 | Vector to goal + distance |
| Current velocity | 7-9 | Velocity (current) |
| Previous velocity | 10-12 | Velocity (t-1) - temporal feature |
| Airborne + Moving | 13-14 | Binary states |
| 12 raycasts | 15-38 | 12 directions x 2 values |
| Ground height | 39 | Raycast down |
| Time + Ranking | 40-41 | Temporal + competitive |
| Last actions | 42-46 | 5 previous actions |

**Actions:**
- `[0-1]` Joystick X/Y: [-1, 1] (continuous)
- `[2]` Jump: {0, 1} (discrete)
- `[3]` Kick/Punch: {0, 1} (discrete)
- `[4]` Slide/Stomp: {0, 1} (discrete)

**Training:**
```powershell
# 1. In Unity, open DreamerParkour.unity and press Play

# 2. Run Dreamer training
python train_dreamer.py --time-scale 3.0 --tb-logdir ./tensorboard_logs

# 3. Monitor
# TensorBoard: http://localhost:6006
# MLflow: http://localhost:5000
```

**Script Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--env` | Path to Unity executable | `None` (Editor) |
| `--run-id` | Run ID | timestamp |
| `--resume` | Resume from checkpoint | `false` |
| `--force` | Overwrite previous runs | `false` |
| `--batch-size` | World model batch size | `2048` (GPU) |
| `--seq-len` | Sequence length | `50` |
| `--horizon` | Imagination horizon | `15` |
| `--num-agents` | Number of parallel agents | `4` |
| `--capacity` | Experience buffer capacity | `200000` |
| `--device` | Device (cuda/cpu) | `cuda` |

**Multi-Agent Training:**
```powershell
# Treinamento com 4 agentes em paralelo (batch size 2048 para GPU)
python train_dreamer.py --time-scale 5.0 --batch-size 2048 --num-agents 4

# Ou use o script PowerShell
.\train_dreamer_multiagent.ps1
```

**Configuração no Unity:**
1. Abra `DreamerParkour.unity`
2. Selecione o objeto `DreamerEnvironment`
3. No Inspector, configure:
   - `Agent Count`: 4 (número de agentes paralelos)
   - `Competitive Mode`: false (para treinamento cooperativo)
   - `Use Visual Observations`: false (mais rápido)

**Arquitetura do Modelo:**
```
Observation (47D) → RSSM Encoder → Latent State z_t (32D)
                          ↓
Previous Action + State → Recurrent Model → Hidden State h_t (256D)
                          ↓
              ┌─────────────────────────┐
              ↓                         ↓
       Observation Decoder      Reward Predictor
       (reconstruction)        (predicted reward)
              ↓                         ↓
       Actor Network          Critic Network
       (policy)               (value function)
```

**Métricas:**
- `world_model/obs_loss` - Erro de reconstrução
- `world_model/reward_loss` - Erro de predição de recompensa
- `world_model/kl_loss` - Divergência KL
- `policy/actor_loss` - Loss do ator
- `policy/critic_loss` - Loss do crítico
- `policy/returns_mean` - Retorno médio estimado

**Arquitetura Comparativa:**
| Aspecto | SB3 (PPO/SAC) | DreamerV3 |
|---------|---------------|-----------|
| Tipo | Model-free | Model-based |
| Dados | Alta amostragem | Eficiente em dados |
| Planejamento | Não | Sim (imaginado) |
| Observações | Atuais | Temporal (sequências) |
| World Model | Não aprende | RSSM aprende dinâmica |

---

### 4. **Team Battle**

Two teams of agents cooperate/compete using MA-POCA (Multi-Agent POsthumous Credit Assignment).

- Scene: `TeamBattle.unity`
- Script: `TeamBattleEnvironment.cs`
- Agent: `TeamBattleAgent.cs`
- Config: `config/team_battle_poca.yaml`

---

### 5. **Chase Training (Pursuer vs Fugitive)**

Decentralized training with two SAC agents: a pursuer and a fugitive.

```powershell
# Terminal 1
.\train_chase.ps1

# Terminal 2: In Unity
1. Open ChaseTraining.unity
2. Play
3. Wait for automatic agent spawning
```

**Observations (31-dim):**
| Field | Indices | Values |
|-------|---------|--------|
| Local position (x,y,z) | 0-2 | [-1, 1] |
| Velocity (x,y,z) | 3-5 | [-1, 1] |
| Grounded | 6 | {0, 1} |
| Relative opponent (x,y,z,dist) | 7-10 | [-1, 1] |
| Opponent velocity (x,y,z) | 11-13 | [-1, 1] |
| Normalized time | 14 | [0, 1] |
| Raycasts | 15-30 | [0, 1] |

**Actions (5 continuous):**
- `[0]` Joystick X: [-1, 1]
- `[1]` Joystick Y: [-1, 1]
- `[2]` Jump: [0, 1]
- `[3]` Kick/Punch: [0, 1]
- `[4]` Slide/Stomp: [0, 1]

**Performance Note:** The SAC algorithm performs neural network updates every step. To prevent freezes, the config uses `time_scale: 1.0`, `steps_per_update: 4`, and `DecisionPeriod: 8`.

---

## PPO Benchmark (ML-Agents vs SB3 vs RLlib vs CleanRL — all REAL frameworks)

Fair PPO comparison with the **same nominal hyperparameters** (`benchmarks/ppo_common.yaml`) on **CompetitiveParkour** (42-dim obs, shared policy, `Box(5)` action with `>0` button threshold — same convention as `train_competitive_sb3.py`). Budget unit: **transitions** (consumed agent-steps), identical for every runner: 3M ≈ 1M Unity env-steps × 3 slots (ML-Agents: 1M `max_steps` per behavior).

| Canonical | SB3 (real) | RLlib (real) | CleanRL (real) | ML-Agents (real) |
|-----------|-----------|--------------|----------------|------------------|
| `lr 3e-4` | `learning_rate` | `lr` | `lr` | `learning_rate` |
| `gamma 0.99` | `gamma` | `gamma` | `gamma` | `extrinsic.gamma` |
| `gae 0.95` | `gae_lambda` | `lambda` | `gae_lambda` | `lambd` |
| `clip 0.2` | `clip_range` | `clip_param` | `clip_coef` | `epsilon` |
| `ent 0.01` | `ent_coef` | `entropy_coeff` | `ent_coef` | `beta` |
| `vf 0.5` | `vf_coef` | `vf_loss_coeff` | `vf_coef` | — (strength 1.0) |
| `steps 2048` | `n_steps` (shared buffer) | `train_batch 2048` | `num_steps` × `num_envs=3` | `buffer 6144` |
| `mini 256` | `batch_size` | `sgd_minibatch` | `num_minibatches 24` | `batch_size` |
| `epochs 3` | `n_epochs` | `num_sgd_iter` | `update_epochs` | `num_epoch` |
| net 256×2 tanh | `policy_kwargs` | `fcnet [256,256] tanh` | vendored Agent, HIDDEN=256 | `hidden 256×2` |

Framework reality notes (all verified in-repo):
- **SB3**: native `PPO` class + `RolloutBuffer` + `train()`, manual Unity loop.
- **CleanRL**: byte-faithful vendor of upstream `ppo_continuous_action.py` (`benchmarks/_vendor/`, sole numeric change HIDDEN 64→256) driven through `UnityVectorEnv` (3 slots as `num_envs=3`); upstream seeding/GAE/loss code paths.
- **RLlib**: native `PPOConfig` + `Tuner.fit()` on `UnityMultiAgentEnv` (3 agents, one shared policy). `timesteps_total` counts ENV steps (verified: iter=986 → ts=2019328=986×2048), so stop = budget/3; `rollout_fragment_length=683` env-steps ≈ one `train_batch` of agent transitions.
- **ML-Agents**: native `mlagents-learn`, generated YAML, constant LR schedule, 3 identical PPO policies (one per behavior; ML-Agents has no cross-behavior sharing).

```powershell
# 0. Inspect canonical hyperparams
python benchmark_ppo.py --list

# 1. Smoke tests (no Unity needed)
python benchmark_ppo.py --framework sb3 --mock --mock-steps 512 --out benchmarks/runs/smoke
python benchmark_ppo.py --framework cleanrl --mock --mock-steps 2048 --out benchmarks/runs/smoke
python benchmark_ppo.py --framework rllib --mock --mock-steps 4096 --out benchmarks/runs/smoke
python benchmark_ppo.py --framework mlagents --out benchmarks/runs/smoke --run-id ppo_bench

# 2. Live headless runs, one worker/port per framework (example: 3M transitions, seed 0)
$exe = "Builds/CompetitiveHeadless/CompetitiveParkour.exe"
python benchmark_ppo.py --framework sb3 --seed 0 --transitions 3000000 --out benchmarks/runs/one_m --env $exe --no-graphics --worker-id 0 --base-port 5005
python benchmark_ppo.py --framework cleanrl --seed 0 --transitions 3000000 --out benchmarks/runs/one_m --env $exe --no-graphics --worker-id 1 --base-port 5015
python benchmark_ppo.py --framework rllib --seed 0 --transitions 3000000 --out benchmarks/runs/one_m --env $exe --no-graphics --worker-id 2 --base-port 5025
python benchmark_ppo.py --framework mlagents --out benchmarks/runs/one_m --run-id ppo_1M --transitions 3000000 --env $exe --no-graphics --base-port 5035 --seed 0
venv_mlagents\Scripts\python.exe -m mlagents.trainers.learn benchmarks/runs/one_m/ppo_1M_ppo_mlagents.yaml --run-id ppo_1M_mlagents --seed 0 --base-port 5035 --force --env $exe --no-graphics [--torch-device cpu]

# 3. Fair in-game eval, ONE protocol for all (Unity = official metrics, Python = inference):
#    3a. (done at build time) eval build contains OnnxEvalRunner in observer mode
#    3b. per framework (example SB3, 60 episodes, time_scale 5.0):
python evaluate.py --framework sb3 --checkpoint benchmarks/runs/one_m/ppo_sb3_seed0.zip --env Builds/EvalBuild/CompetitiveParkourEval.exe --no-graphics --episodes 60 --time-scale 5.0 --worker-id 3 --base-port 5045 --additional-args -evalObserve 1 -evalEpisodes 60 -evalOut <abs path>/eval_sb3.csv
#    checkpoints: sb3 .zip | cleanrl .pt (vendored Agent) | rllib Tuner dir (shared policy) | mlagents .onnx (onnxruntime + masks, deterministic heads)

# 4. Compare
python benchmarks/compare_ppo.py --dir benchmarks/runs/one_m --out benchmarks/runs/one_m/compare.png
```

Caveats:
- Same nominal hyperparams, framework-native rollout semantics; update cadence differs slightly (SB3/RLlib ≈ every 683 env-steps on 2048 transitions; CleanRL/ML-Agents ≈ every 2048 env-steps on ~6k) with identical data/use ratios and minibatch/epochs — documented in `benchmarks/ppo_common.yaml`.
- SB3/CleanRL/RLlib share **one** policy across the 3 slots; ML-Agents trains one policy **per** slot with identical hyperparams (no cross-behavior sharing exists).
- Network inits/activations stay framework-native (SB3/CleanRL tanh+orthogonal; RLlib fcnet tanh; ML-Agents default).
- Every runner writes `ppo_<fw>_seed<S>.csv` per episode — the comparison is built on those CSVs, not on any single framework's logger.

**Live results (seed 0, 3M transitions ≈ 1M env-steps × 3 slots, headless builds, `time_scale 5.0`):**

Training curves, episodic return, shared policy (ML-Agents curve = TB smoothed means, 500 windows; others = raw per-episode rows):

| Framework | Episodes | Mean reward | Last-50 | PPO updates |
|-----------|----------|-------------|---------|-------------|
| ML-Agents | 500 TB windows | 103.0 | 138.1 | native (≈1/2048 env-steps, 6k-transition batches) |
| CleanRL-real | 181699 | 84.0 | 119.4 | 488 (6144-batch) |
| RLlib-real | 145287 | 63.3 | 68.6 | train_batch 2048 (≈1/683 env-steps) |
| SB3 | 148441 | 57.6 | 54.2 | ~1465 (shared 2048-buffer) |

In-game eval (deterministic policies, shared across 3 slots, 60 episodes, `time_scale 5.0`, Unity = official metrics):

| Framework | Success | Min dist to goal | Mean return |
|-----------|---------|------------------|-------------|
| CleanRL-real | 0/60 | **13.8 m** | **175.1** |
| RLlib-real | 0/60 | 28.9 m | 65.8 |
| ML-Agents | 0/60 | 30.6 m | 54.6 |
| SB3 | 0/60 | 31.5 m | 46.7 |

Reading: 1M env-steps from scratch does not complete this parkour with any framework (agents fall within ~2 s; 0/60 everywhere — the map is hard). Two honest findings: (1) **CleanRL-real wins on both measures** — best training-final among bridge runs and 2× closer to the goal in eval; (2) **ML-Agents trains best stochastically but evals worst deterministically** (and vice-versa for CleanRL, whose training curve collapses ~2.5M yet whose deterministic mean excels) — same-hyperparams PPO can diverge between exploration and exploitation regimes. RLlib is the most stable curve with the lowest ceiling; SB3 collapses late (~2.2M) and plateaus ~50. Single seed — treat gaps <15% as noise; the CleanRL min-dist gap (>2×) is the only large effect.
Curves: `benchmarks/runs/one_m/compare.png` (gitignored). Checkpoints: `ppo_sb3_seed0.zip`, `ppo_cleanrl_seed0.pt`, `ppo_rllib_seed0_final/`, `results/ppo_1M_b_mlagents/`. Eval CSVs: `benchmarks/runs/eval_{sb3,cleanrl,rllib,mlagents}.csv`. Rerun seeds 1–2 for error bars.

---

## Available Scenes

| Scene | Purpose | Mode | Parallel |
|-------|---------|------|----------|
| **ParkourTraining.unity** | Standard RL | 4 Parallel Marios | Yes |
| **ParkourTraining_OldSystem.unity** | Legacy system | Backup | No |
| **CompetitiveParkour.unity** | 3-model competition | 1 PPO + 1 SAC + 1 DQN | No |
| **DreamerParkour.unity** | World Model RL | DreamerV3 Agent | No |
| **TeamBattle.unity** | Team 2v2 | 4 Agents | Yes |
| **ChaseTraining.unity** | 1v1 Chase (SAC) | 2 Agents (Pursuer + Fugitive) | Yes |

---

## Usage Instructions

### Option 1: Traditional RL Training (Parkour)

```powershell
# Terminal 1
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2
# Open ParkourTraining.unity in Unity -> Play
# Wait for message: "Agent initialized. Ready to accept experiences."
```

### Option 2: Competitive SB3 Training

```powershell
# 1. Open CompetitiveParkour.unity in Unity -> Play
# 2. Run
python train_competitive_sb3.py --time-scale 3.0 --tb-logdir ./tensorboard_logs
```

### Option 3: Dreamer Training

```powershell
# 1. Open DreamerParkour.unity in Unity -> Play
# 2. Run
python train_dreamer.py --time-scale 3.0 --tb-logdir ./tensorboard_logs
```

### Option 4: Chase / TeamBattle (ML-Agents)

```powershell
# Chase (SAC)
mlagents-learn config/chase_training_sac.yaml --run-id chase_sac
# Open ChaseTraining.unity -> Play

# TeamBattle (POCA)
mlagents-learn config/team_battle_poca.yaml --run-id team_v1
# Open TeamBattle.unity -> Play
```

---

## MLOps Integration

All training scripts integrate with **MLflow** for experiment tracking:

```powershell
# Automatically track metrics
python train_mlops.py --run-id mario_parkour_run1

# View dashboard
mlflow ui
# Access: http://localhost:5000
```

**What is tracked:**
- Hyperparameters (learning rate, batch size, network architecture)
- Per-episode rewards and episode lengths
- Model checkpoints (ZIP and ONNX) every 50 episodes
- Final trained models
- Configuration files as artifacts
- TensorBoard logs

---

## API Reference

### BackupParkourEnvironment.cs (Legacy OldSystem)

**Enum:**
```csharp
public enum StartupMode { Training = 0, Recording = 1 }
```

**Properties:**
```csharp
public Transform marioSpawnPointPublic { get; }  // Player spawn
public Transform goalPublic { get; }              // Target goal
```

**Methods:**
```csharp
public float[] CollectObservationVector()  // Returns 30-dim obs
public void ResetEnvironment()              // Episode reset
public void SetStartupMode(StartupMode mode)  // Change mode
```

### Evaluation

```bash
# Evaluate SB3 checkpoints offline (no Unity required for smoke test)
python evaluate.py --checkpoint models/<run>/MarioParkourPPO_final.zip --episodes 5

# Evaluate ML-Agents ONNX (requires Unity build or Editor)
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id eval --inference
```

---

## Configuration

### ML-Agents YAML (mario_parkour.yaml)

```yaml
behaviors:
  MarioParkour:
    trainer_type: ppo
    hyperparameters:
      batch_size: 512
      buffer_size: 4096
      learning_rate: 0.0003
      learning_rate_schedule: linear
      beta: 0.02
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
    network_settings:
      normalize: true
      hidden_units: 256
      num_layers: 2
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    max_steps: 5000000
    time_horizon: 128
    summary_freq: 2000
    threaded: true
```

### Requirements (pinned, see `requirements.txt`)

```
mlagents==0.28.0
mlagents-envs==0.28.0
torch==2.2.0
torchvision==0.17.0
stable-baselines3==2.3.2
gymnasium==0.29.1
shimmy>=1.0.0
protobuf<=3.20.3
onnx==1.16.0
onnxruntime>=1.17.0
pyyaml>=6.0
tensorboard>=2.12.0
numpy==1.26.4
pillow>=9.0.0
matplotlib>=3.5.0
mlflow>=2.0.0
```

---

## Docker

```bash
# Build
docker build -t mariorl:latest .

# Run (training still needs Unity Editor/Build connection unless using evaluate.py)
docker run -it --rm -v ${PWD}/results:/app/results -v ${PWD}/models:/app/models \
  -p 5000:5000 -p 6006:6006 \
  mariorl:latest python train_mlops.py --run-id dockerrun
```

> Limitation: this image contains only the Python trainers. Unity Editor/Build must run separately (or use `--env <build>`). TensorBoard on `:6006`, MLflow on `:5000`.

---

## Troubleshooting

### Problem: "Error: No agent found"

**Solution:**
```powershell
# Terminal 1: Start trainer
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2: THEN press Play in Unity (wait 3s)
```

### Problem: "ROM not found"

**Solution:**
1. Verify file: `baserom.us.z64`
2. Confirm MD5: `20b854b239203baf6c961b850a4a51a2`
3. Place in project root

### Problem: "Python module not found"

**Solution:**
```bash
pip install -r requirements.txt
# or
pip install mlagents==0.28.0 torch numpy scikit-learn matplotlib
```

### Problem: "ModuleNotFoundError: No module named 'mlagents_envs'"

**Solution:**
```powershell
# Windows
pip install --upgrade ml-agents==0.28.0 ml-agents-envs==0.28.0

# or use venv
.\setup_python39.ps1
```

### Problem: "Unity freezes during ChaseTraining (SAC)"

**Cause:** SAC with `steps_per_update: 1` performs neural network updates every environment step. With a high `time_scale`, the Python trainer cannot keep up with Unity and gets blocked in `DecideAction`.

**Solution (already applied in config):**
- `steps_per_update: 4` -- reduces network update frequency
- `time_scale: 1.0` -- lowers simulation speed
- `DecisionPeriod: 8` -- reduces decision requests per second
- `batch_size: 512` -- better hardware utilization per update

If it still freezes, further reduce `time_scale` or increase `steps_per_update`.

---

## References

- **ML-Agents**: https://github.com/Unity-Technologies/ml-agents
- **libsm64-unity**: https://github.com/libsm64/libsm64-unity
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **PyTorch**: https://pytorch.org/
- **MLflow**: https://mlflow.org/
- **Gymnasium**: https://gymnasium.farama.org/

---

## License

This project uses:
- libsm64-unity (MIT)
- ML-Agents Toolkit (Apache 2.0)
- PyTorch (BSD)
- Stable-Baselines3 (MIT)

Check the respective repositories for license details.

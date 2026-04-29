# libsm64-unity-dev - Parkour RL + Hybrid Training System

A complete system for training AI agents in Super Mario 64 parkour environments using **Reinforcement Learning**, **Imitation Learning**, and **Offline RL**, powered by [libsm64](https://github.com/libsm64/libsm64-unity) and Unity ML-Agents.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Training Systems](#training-systems)
  - [Parkour RL (Traditional)](#1-parkour-rl-traditional)
  - [Hybrid Training System](#2-hybrid-training-system)
  - [Competitive Parkour (PPO, SAC, DQN)](#3-competitive-parkour-ppo-sac-dqn)
  - [Team Battle](#4-team-battle)
  - [Chase Training (Pursuer vs Fugitive)](#5-chase-training-pursuer-vs-fugitive)
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

- **Multi-Algorithm Competitive Training**: Simultaneously train PPO, SAC, and DQN agents on the same parkour map
- **Hybrid Learning Pipeline**: Combine Imitation Learning, Offline RL (CQL/IQL), and Online RL
- **Parallel Environment Support**: Train with up to 4 parallel Mario agents
- **MLOps Integration**: Full experiment tracking via MLflow with TensorBoard visualization
- **ONNX Export**: Export trained models for inference in Unity
- **Curriculum Learning**: Progressive difficulty with win-rate-based advancement
- **Team Battle Mode**: Multi-agent cooperative/competitive training with MA-POCA
- **Chase Training**: Pursuer vs Fugitive SAC-based training
- **Docker Support**: Containerized training environment
- **Recording System**: Record human demonstrations for Imitation Learning

---

## Project Architecture

```
libsm64-unity-dev/
├── Assets/
│   ├── ParkourRL/
│   │   ├── Scripts/
│   │   │   ├── BackupSystem/
│   │   │   │   ├── BackupParkourEnvironment.cs     [Recording System]
│   │   │   │   ├── BackupMarioRLAgent.cs
│   │   │   │   └── BackupCheckpoint.cs
│   │   │   ├── HybridSystem/
│   │   │   │   ├── MarioHybridAgent.cs             [Hybrid Agent]
│   │   │   │   ├── HybridParkourEnvironment.cs
│   │   │   │   ├── HybridDataRecorder.cs           [Records data]
│   │   │   │   ├── HybridTrainingManager.cs        [Manages modes]
│   │   │   │   ├── README_HYBRID.md                [Docs]
│   │   │   │   ├── QUICKSTART.md
│   │   │   │   └── HybridSceneSetup.md
│   │   │   ├── ParkourEnvironment.cs               [Traditional RL]
│   │   │   ├── MarioRLAgent.cs
│   │   │   ├── MarioInputProvider.cs
│   │   │   ├── PlayerMarioSpawner.cs
│   │   │   ├── CompetitiveParkourEnvironment.cs    [Competitive mode]
│   │   │   ├── MarioCompetitiveAgent.cs
│   │   │   ├── ChaseTrainingEnvironment.cs         [Chase mode]
│   │   │   ├── ChaseAgent.cs
│   │   │   ├── TeamBattleEnvironment.cs            [Team mode]
│   │   │   └── TeamBattleAgent.cs
│   │   ├── Scenes/
│   │   │   ├── ParkourTraining.unity               [Standard RL]
│   │   │   ├── ParkourTraining_OldSystem.unity
│   │   │   ├── HybridTraining.unity                [IL + RL]
│   │   │   ├── CompetitiveParkour.unity
│   │   │   ├── ChaseTraining.unity
│   │   │   └── TeamBattle.unity
│   │   ├── Config/
│   │   │   ├── mario_parkour.yaml                  [RL Config]
│   │   │   └── mario_parkour_hybrid.yaml           [Hybrid Config]
│   │   └── Models/
│   └── libsm64-unity/                              [Native wrapper]
│
├── python_trainers/
│   ├── train_behavior_cloning.py                   [Imitation Learning]
│   ├── train_offline_rl.py                         [Offline RL (CQL/IQL)]
│   ├── analyze_dataset.py                          [Data analysis]
│   └── requirements.txt
│
├── config/
│   ├── chase_training_sac.yaml                     [Chase SAC config]
│   └── team_battle_poca.yaml                       [Team Battle config]
│
├── HybridTrainingData/                             [Recorded data]
├── models/                                         [Trained models]
├── results/                                        [Training results]
├── tensorboard_logs/                               [TensorBoard logs]
│
├── train_competitive_sb3.py                        [SB3 competitive training w/ TensorBoard]
├── train_simultaneous_sb3.py                       [Simultaneous multi-algo training]
├── train_mlops.py                                  [MLOps wrapper]
├── trainer_mlflow.py                               [MLflow trainer]
├── validate_recording_integration.py               [Recording validation]
│
├── train_mario.ps1 / .bat                          [Standard training scripts]
├── train_hybrid.ps1 / .bat                         [Hybrid training scripts]
├── train_chase.ps1                                 [Chase training script]
├── train_mario_fast.bat                            [Fast training script]
├── setup_training_env.ps1                          [Environment setup]
├── setup_python38.ps1 / setup_python39.ps1         [Python installers]
│
├── requirements.txt                                [Python dependencies]
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

---

### 2. **Hybrid Training System**

Combines **Imitation Learning (IL)**, **Offline RL**, and **Online RL** in a single pipeline.

**Architecture:**
```
Phase 1: Recording (Record player demos)
  |
Phase 2: Behavior Cloning (Train network to imitate)
  |
Phase 3: Offline RL (Refine with CQL/IQL)
  |
Phase 4: Online RL with Warm-Start (PPO + initial weights)
```

**Components:**
- `BackupParkourEnvironment.cs` - Controls Recording/Training mode
- `HybridDataRecorder.cs` - Saves data in JSON/CSV
- `RecordingInputProvider` - Captures human input
- `train_behavior_cloning.py` - BC training
- `train_offline_rl.py` - CQL/IQL training

**Data File Format:**
```json
{
  "metadata": {
    "createdAt": "2026-04-25T14:30:00",
    "episodeCount": 10,
    "totalSteps": 2500,
    "averageReward": 15.3,
    "successRate": 0.6
  },
  "episodes": [
    {
      "episodeId": 0,
      "startTime": "2026-04-25T14:30:00",
      "endTime": "2026-04-25T14:30:15",
      "duration": 15.2,
      "success": true,
      "stepCount": 300,
      "totalReward": 45.7,
      "transitions": [
        {
          "step": 0,
          "timestamp": 0.0,
          "observations": [0.1, 0.2, 0.0, ...],
          "actions": [0.5, 0.0, 1.0],
          "reward": 2.5,
          "nextObservations": [0.11, 0.21, ...],
          "done": false
        }
      ]
    }
  ]
}
```

**Full Workflow:**

```powershell
# 1. Generate data in Recording mode
# Unity: HybridTraining.unity -> Recording mode (press M) -> Play
# Data saved to: HybridTrainingData/hybrid_episodes_*.json

# 2. Analyze dataset
cd python_trainers
python analyze_dataset.py --data ../HybridTrainingData/ --plots

# 3. Train Behavior Cloning
python train_behavior_cloning.py \
    --data ../HybridTrainingData/ \
    --epochs 100 \
    --batch-size 32 \
    --output models/bc_mario.pth

# 4. Train Offline RL (CQL)
python train_offline_rl.py \
    --data ../HybridTrainingData/ \
    --algo CQL \
    --epochs 50 \
    --output models/cql_mario.pth

# 5. Train Online RL with warm-start (future version)
mlagents-learn Assets/ParkourRL/Config/mario_parkour_hybrid.yaml \
    --run-id=mario_hybrid_bc \
    --initialize-from=models/bc_mario.pth
```

---

### 3. **Competitive Parkour (PPO, SAC, DQN)**

Three models (PPO, SAC, DQN) compete simultaneously to reach the goal first. The environment is dedicated (does not reuse the old parkour) and does not use parallel training.

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

## Available Scenes

| Scene | Purpose | Mode | Parallel |
|-------|---------|------|----------|
| **ParkourTraining.unity** | Standard RL | 4 Parallel Marios | Yes |
| **ParkourTraining_OldSystem.unity** | Legacy system | Backup | No |
| **HybridTraining.unity** | IL + Offline RL | Recording mode | Yes |
| **CompetitiveParkour.unity** | 3-model competition | 1 PPO + 1 SAC + 1 DQN | No |
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

### Option 2: Recording Data (Recording Mode)

```powershell
# In Unity
1. Open Assets/ParkourRL/Scenes/HybridTraining.unity
2. In the Hierarchy, select: ParkourEnvironment
3. In Inspector, BackupParkourEnvironment: Startup Mode = Recording
4. Play (Ctrl+P)
5. Control Mario with WASD + Space
6. Complete the parkour (or fail)
7. Episode is saved to: HybridTrainingData/hybrid_episodes_*.json
```

**Repeat:** To record more episodes, simply play again.

### Option 3: Training with Recorded Data

```powershell
cd python_trainers

# Analyze data
python analyze_dataset.py --data ../HybridTrainingData/ --plots

# Behavior Cloning
python train_behavior_cloning.py \
    --data ../HybridTrainingData/ \
    --epochs 100 \
    --output models/bc_mario.pth

# Offline RL (CQL - Conservative Q-Learning)
python train_offline_rl.py \
    --data ../HybridTrainingData/ \
    --algo CQL \
    --epochs 50 \
    --output models/cql_mario.pth
```

**Output:**
- `models/bc_mario.pth` - Trained model (Behavior Cloning)
- `models/cql_mario.pth` - Trained model (Offline RL)
- Training plots in `plots/`

### Option 4: Hybrid Mode (Player + AI side by side)

```powershell
# Terminal 1
.\train_hybrid.ps1

# Terminal 2: In Unity
1. Open HybridTraining.unity
2. Play
3. Player Mario (green) = WASD + Space
4. AI Mario (blue) = ML-Agents (trains in real time)
5. Press M to toggle between Training/Recording
```

---

## MLOps Integration

All training scripts integrate with **MLflow** for experiment tracking:

```powershell
# Automatically track metrics
python trainer_mlflow.py --run-id mario_parkour_run1

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

### BackupParkourEnvironment.cs

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

### HybridDataRecorder.cs

**Public Methods:**
```csharp
void StartEpisode()  // Start new episode
void RecordStep(float[] obs, float[] actions, float reward, 
                float[] nextObs, bool done)  // Record one step
void SaveEpisode(Transition[] transitions, bool success)  // Save episode
void FlushBatch()  // Force flush to disk
```

**Events:**
```csharp
event Action OnEpisodeStarted
event Action<bool> OnEpisodeEnded  // bool = success
event Action<string> OnDataSaved  // string = filepath
```

### Python API

**analyze_dataset.py:**
```bash
python analyze_dataset.py \
    --data HybridTrainingData/ \
    --plots \
    --output analysis/
```

**train_behavior_cloning.py:**
```bash
python train_behavior_cloning.py \
    --data HybridTrainingData/ \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output models/bc.pth \
    --test-split 0.2
```

**train_offline_rl.py:**
```bash
python train_offline_rl.py \
    --data HybridTrainingData/ \
    --algo CQL  # or IQL
    --epochs 50 \
    --learning-rate 0.0003 \
    --output models/offline.pth
```

---

## Configuration

### ML-Agents YAML (mario_parkour_hybrid.yaml)

```yaml
behaviors:
  MarioHybrid:
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

### Requirements

```
mlflow>=2.0.0
mlagents==0.28.0
mlagents-envs==0.28.0
torch>=1.11.0
stable-baselines3>=2.0.0
gymnasium>=0.28.0
shimmy>=1.0.0
protobuf<=3.20.3
onnx
pyyaml
tensorboard>=2.12.0
numpy>=1.21.0
```

---

## Docker

```bash
# Build
docker build -t mariorl:latest .

# Run
docker run -it --rm -v ${PWD}/results:/app/results \
    mariorl:latest python train_mlops.py --run-id dockerrun
```

---

## Troubleshooting

### Problem: "No JSON files found"

**Solution:**
1. Verify you ran in Recording mode
2. Confirm the `HybridTrainingData/` folder exists
3. Run at least 1 complete episode (up to 30s or fall)

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

# HybridTraining Scene Setup

## Overview

The **HybridTraining** scene allows training an ML-Agents agent (AI Mario - blue) side by side with a player-controlled Mario (green). The system supports two modes:

- **Training Mode**: Standard RL training with ML-Agents
- **Recording Mode**: Records player data for Imitation Learning and Offline RL

## Automatic Method (Recommended)

### Using the Hybrid Scene Builder

1. In the Unity Editor, go to: **`ParkourRL > Build Hybrid Training Scene`**
2. Wait for the scene to be created automatically
3. When the "Setup Complete" message appears, press **Play** to test!

### Configure References (if needed)

If any reference was not assigned automatically:

1. In the menu: **`ParkourRL > Setup Hybrid Scene References`**
2. Check the Inspector on the "HybridEnvironment" GameObject:
   - `Goal` should point to "GoalPlatform"
   - `Player Spawn Point` should point to "PlayerSpawn"
   - `AI Spawn Point` should point to "AISpawn"
   - `Data Recorder` and `Training Manager` should be connected

## Manual Method (Alternative)

If you prefer to create the scene manually, follow these steps:

### 1. Create New Scene

1. In Unity: `File > New Scene`
2. Save as: `Assets/ParkourRL/Scenes/HybridTraining.unity`

### 2. GameObject Hierarchy

Create the following hierarchy:

```
HybridTraining (Scene)
+-- HybridEnvironment (Empty GameObject)
|   +-- HybridParkourEnvironment.cs
|   +-- HybridDataRecorder.cs
|   +-- HybridTrainingManager.cs
|   +-- Spawn Points:
|       +-- PlayerSpawn (Transform - position x: -7, y: 2, z: 0)
|       +-- AISpawn (Transform - position x: 7, y: 2, z: 0)
|
+-- Parkour Level (copy from ParkourTraining.unity)
|   +-- StartPlatform (SM64StaticTerrain)
|   +-- Platform_1 (SM64StaticTerrain)
|   +-- Platform_2 (SM64StaticTerrain)
|   +-- Platform_3 (SM64StaticTerrain)
|   +-- GoalPlatform (with trigger collider + tag "Goal")
|
+-- UI Canvas
|   +-- ModeText (Text UI - top corner)
|   +-- InfoText (Text UI - top center)
|   +-- TrainingButton (Button)
|   +-- RecordingButton (Button)
|   +-- StatsPanel (stats panel)
|
+-- Cameras
    +-- PlayerCamera (Camera - viewport rect: 0,0,0.5,1)
    +-- AICamera (Camera - viewport rect: 0.5,0,0.5,1)
```

### 3. Component Configuration

#### HybridParkourEnvironment

```csharp
// Inspector settings:
Mario Prefab: Assets/Mario.prefab
Player Material: (leave null - will be created automatically)
AI Material: (leave null - will be created automatically)
Goal: (drag GoalPlatform)
Player Spawn Point: (drag PlayerSpawn)
AI Spawn Point: (drag AISpawn)
Mario Spacing: 15
Sync Resets: true
Show Comparison UI: true
Data Recorder: (drag HybridDataRecorder from scene)
Training Manager: (drag HybridTrainingManager from scene)
```

#### HybridDataRecorder

```csharp
Output Directory: HybridTrainingData
Output Format: Both (JSON + CSV)
File Name: hybrid_episodes
Max Episodes Per File: 100
Include Metadata: true
Min Record Interval: 0.05
Max Steps Per Episode: 2000
```

#### HybridTrainingManager

```csharp
Initial Mode: Training
Allow Mode Switching: true
Mode Switch Key: M
UI References: (drag UI elements)
Environment: (drag HybridParkourEnvironment)
Data Recorder: (drag HybridDataRecorder)
```

### 4. Parkour Configuration

Copy platforms from the `ParkourTraining.unity` scene:

1. Open `ParkourTraining.unity`
2. Select all platforms (SM64StaticTerrain)
3. Copy (Ctrl+C)
4. Open `HybridTraining.unity`
5. Paste (Ctrl+V)

Add **SM64StaticTerrain** to all platforms if not already present.

### 5. Goal Configuration

The Goal must have:
- **Transform**: On the last platform
- **Collider**: BoxCollider or SphereCollider with `isTrigger = true`
- **Tag**: "Goal"

### 6. YAML Configuration

Create `Assets/ParkourRL/Config/mario_parkour_hybrid.yaml`:

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

## How to Use

### Training Mode (Default)

1. Open `HybridTraining.unity`
2. Press **Play**
3. Run `train_hybrid.bat` (or `.ps1`)
4. AI Mario (blue) trains with ML-Agents
5. Player Mario (green) can be controlled with **WASD + Space**
6. Both compete to complete the parkour first

### Recording Mode

1. Press **M** or click "Recording Mode"
2. Control green Mario with **WASD + Space**
3. Data is recorded automatically to:
   ```
   HybridTrainingData/hybrid_episodes_YYYYMMDD_HHMMSS_batch0.json
   HybridTrainingData/hybrid_episodes_YYYYMMDD_HHMMSS_batch0.csv
   ```
4. Complete the parkour (or fail) - the episode is saved
5. Data contains: observations, actions, rewards, termination flags

### Using Recorded Data

Data can be used for:

1. **Imitation Learning (Behavior Cloning)**
   ```bash
   python train_behavior_cloning.py --data HybridTrainingData/
   ```

2. **Offline RL (CQL, IQL)**
   ```bash
   python train_offline_rl.py --data HybridTrainingData/ --algo CQL
   ```

3. **Warm Start for RL**
   ```bash
   mlagents-learn config.yaml --initialize-from=models/bc_model.onnx
   ```

## Controls

| Key | Action |
|-----|--------|
| **WASD** / **Arrows** | Move Player Mario |
| **Space** | Jump |
| **M** | Toggle mode (Training/Recording) |
| **P** | Pause/Resume |
| **R** | Reset both Marios |

## Data Architecture

### JSON Format

```json
{
  "metadata": {
    "createdAt": "2026-04-25 14:30:00",
    "episodeCount": 50,
    "totalSteps": 5000,
    "averageReward": 23.5,
    "successRate": 0.75
  },
  "episodes": [
    {
      "episodeId": 0,
      "success": true,
      "duration": 12.5,
      "stepCount": 250,
      "totalReward": 45.2,
      "transitions": [
        {
          "step": 0,
          "observations": [0.1, 0.2, ...],
          "actions": [0.5, 0.3, 0.0],
          "reward": -0.01,
          "done": false
        }
      ]
    }
  ]
}
```

### CSV Format

```csv
episode_id,step,timestamp,obs_0,...,obs_29,action_0,action_1,action_2,reward,done
0,0,0.000,0.1,...,0.5,0.5,0.3,0.0,-0.01,0
0,1,0.050,0.2,...,0.6,0.4,0.2,0.0,-0.01,0
```

## Troubleshooting

### Problem: AI Mario does not move

**Solution**: Check that:
1. ML-Agents is running (`train_hybrid.bat`)
2. Behavior Name in YAML matches the agent (`MarioHybrid`)
3. Mode is "Training", not "Recording"

### Problem: Data is not being saved

**Solution**: Check that:
1. Directory `HybridTrainingData/` exists (create manually if needed)
2. Mode is "Recording"
3. Episode terminates (success or failure)
4. Check console for file permission errors

### Problem: Split camera does not work

**Solution**:
1. Check that cameras have different Viewport Rects
2. PlayerCamera: Rect(0, 0, 0.5, 1)
3. AICamera: Rect(0.5, 0, 0.5, 1)

## Next Steps

1. Record 50+ human demonstration episodes
2. Train Behavior Cloning with the data
3. Use BC model as warm-start for PPO
4. Apply DAgger to improve with new data

---

**Note**: This system is a complete implementation of a hybrid IL+RL architecture. Generated data is compatible with standard Offline RL algorithms (CQL, IQL, BCQ).

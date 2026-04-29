# Hybrid Training System - Mario Parkour

## Overview

This system implements a hybrid architecture that combines:

- **Imitation Learning (IL)**: Learning from human demonstrations
- **Offline RL (CQL/IQL)**: Training from recorded data without environment interaction
- **Online RL (PPO)**: Standard reinforcement learning with ML-Agents
- **DAgger**: Dataset Aggregation to iteratively improve policies

## Architecture

### Unity Scripts

| Script | Function |
|--------|----------|
| `MarioHybridAgent.cs` | Agent that supports Training and Recording modes |
| `HybridParkourEnvironment.cs` | Manages two Marios side by side |
| `HybridPlayerMario.cs` | Player-controlled Mario for recording demos |
| `HybridDataRecorder.cs` | Records and saves transitions in JSON/CSV |
| `HybridTrainingManager.cs` | Manages mode switching (Training/Recording) |

### Python Scripts

| Script | Function |
|--------|----------|
| `train_behavior_cloning.py` | Trains neural network to imitate demonstrations |
| `train_offline_rl.py` | Trains with CQL/IQL from recorded data |
| `analyze_dataset.py` | Dataset statistics and quality analysis |

## Workflow

### 1. Recording Phase

```
Scene: HybridTraining.unity
Mode: Recording (press M)
Action: Control green Mario with WASD+Space
Output: JSON/CSV files in HybridTrainingData/
```

### 2. Behavior Cloning Phase

```bash
cd python_trainers
python train_behavior_cloning.py \
    --data ../HybridTrainingData/ \
    --epochs 100 \
    --output models/bc_mario.pth
```

### 3. Offline RL Phase

```bash
python train_offline_rl.py \
    --data ../HybridTrainingData/ \
    --algo CQL \
    --epochs 50 \
    --output models/cql_mario.pth
```

### 4. DAgger (Advanced)

```bash
# Iteration 1: Agent attempts, human corrects
python train_behavior_cloning.py --data ../HybridTrainingData/ --output models/dagger_iter1.pth

# Iteration 2: More data collected with mixed policy
python train_behavior_cloning.py --data ../HybridTrainingData/ --output models/dagger_iter2.pth
```

### 5. Online RL with Warm-Start

```bash
mlagents-learn Assets/ParkourRL/Config/mario_parkour_hybrid.yaml \
    --run-id=mario_hybrid_warmstart \
    --initialize-from=results/prev_run/MarioHybrid.onnx
```

## Data Format

### JSON Structure

```json
{
  "metadata": {
    "createdAt": "2026-04-25T14:30:00",
    "episodeCount": 10,
    "totalSteps": 2500,
    "averageReward": 15.3,
    "successRate": 0.6
  },
  "episodes": [{
    "episodeId": 0,
    "success": true,
    "stepCount": 300,
    "totalReward": 45.7,
    "transitions": [{
      "step": 0,
      "observations": [0.1, 0.2, ...],
      "actions": [0.5, 0.0, 1.0],
      "reward": 2.5,
      "nextObservations": [0.11, ...],
      "done": false
    }]
  }]
}
```

## API Reference

### MarioHybridAgent

```csharp
// Operation modes
public enum HybridMode { Training, Recording }

// Public methods
void SetMode(HybridMode mode)
void SetGoal(Transform goal)
void SetRecorder(HybridDataRecorder recorder)
```

### HybridPlayerMario

```csharp
// Methods
void RecordTransition(float[] obs, float[] actions, float reward)
void EndEpisode(bool success)
float[] CollectObservations()
```

### HybridDataRecorder

```csharp
// Methods
void StartEpisode()
void RecordStep(float[] obs, float[] actions, float reward, float[] nextObs, bool done)
void SaveEpisode(Transition[] transitions, bool success)
void FlushBatch()
```

## ML-Agents Integration

The system is fully compatible with ML-Agents:

1. **BehaviorName**: `MarioHybrid`
2. **Observations**: 30 dimensions (same format as the standard MarioRLAgent)
3. **Actions**: 2 continuous (joystick) + 1 discrete (jump)
4. **Config**: `mario_parkour_hybrid.yaml`

## YAML Configuration

See `Assets/ParkourRL/Config/mario_parkour_hybrid.yaml`

### Differences from default parkour:
- **Beta**: 0.02 (higher exploration for imitation)
- **Buffer Size**: 4096 (smaller for faster adaptation)
- **Batch Size**: 512 (smaller for rapid adaptation)

## Future Extensions

### Reward Shaping from Demonstrations
```python
# Pseudocode
for transition in demo_data:
    if similar_to_expert(agent_state, transition):
        bonus_reward += 0.1
```

### GAIL (Generative Adversarial IL)
```yaml
# ML-Agents native support
reward_signals:
  gail:
    strength: 0.5
    demo_path: HybridTrainingData/demos.demo
```

### Reward Relabeling
- Human demonstrations: reward = 1 (maximum)
- Agent interactions: normal environment reward
- Mixed: weighted average

## Troubleshooting

### No data recorded
```bash
# Check directory
ls HybridTrainingData/

# Permissions
chmod 755 HybridTrainingData/

# Check data overlap
python analyze_dataset.py --data HybridTrainingData/
```

### BC model not converging
- Verify data quality with `analyze_dataset.py`
- Try reducing learning rate: `--lr 1e-4`
- Increase hidden layer size: `--hidden-size 512`

### Offline RL with high Q-values
- Increase CQL alpha: `--cql-alpha 5.0`
- Use IQL instead: `--algo IQL`

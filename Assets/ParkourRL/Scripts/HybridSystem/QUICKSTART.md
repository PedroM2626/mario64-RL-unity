# Quick Start Guide - Hybrid Training System

## Getting Started in 3 Steps

### 1. Build the Scene
In Unity, go to: **Tools > ParkourRL > Build Hybrid Scene**

This creates:
- A hybrid environment with all components
- Split-screen cameras
- Spawn points and goal

### 2. Configure (if needed)
Select `HybridEnvironment` in the Hierarchy:
- **marioSpawnPoint**: Where Marios spawn
- **goal**: Target destination
- **marioSeparation**: Distance between the two Marios

### 3. Play
1. Press Play (Ctrl+P)
2. Use `M` to toggle between Training/Recording modes
3. In Recording mode: play with WASD + Space

## Controls

| Key | Action |
|-----|--------|
| **WASD** | Move |
| **Space** | Jump |
| **M** | Toggle mode |
| **ESC** | Pause |

## Operation Modes

### Training Mode (Default)
- AI Mario trains via ML-Agents (PPO)
- Player Mario can be controlled for comparison

### Recording Mode
- Player Mario records demonstrations
- Data saved to `HybridTrainingData/`
- Used for Behavior Cloning and Offline RL

## After Recording

```bash
cd python_trainers

# Analyze the data
python analyze_dataset.py --data ../HybridTrainingData/ --plots

# Train Behavior Cloning
python train_behavior_cloning.py --data ../HybridTrainingData/ --epochs 100

# Train Offline RL (CQL)
python train_offline_rl.py --data ../HybridTrainingData/ --algo CQL
```

## File Locations

| Type | Location |
|------|----------|
| Recorded data | `HybridTrainingData/` |
| BC models | `python_trainers/models/` |
| Scenes | `Assets/ParkourRL/Scenes/` |
| Configs | `Assets/ParkourRL/Config/` |

## Troubleshooting

### "Cameras not appearing"
- Ensure cameras were created via the scene builder
- Check the Hierarchy for `PlayerCamera` and `AICamera`

### "Data not being recorded"
- Set mode to Recording (press M)
- Check that `HybridTrainingData/` directory exists
- Console should show: "[HybridRecorder] Episode X started"

### "ML-Agents not connecting"
- Start `mlagents-learn` before pressing Play in Unity
- Config: `mario_parkour_hybrid.yaml`

## Full Documentation
- [README_HYBRID.md](README_HYBRID.md) - Full system documentation
- [HybridSceneSetup.md](HybridSceneSetup.md) - Scene setup details

**Ready to start! Press Play and enjoy!**

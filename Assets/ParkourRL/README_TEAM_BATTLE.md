# 🎮 Team Battle System for libsm64-unity

## Overview

The **Team Battle System** is a new ML-Agents training environment where 2 teams of 5 Marios each compete in head-to-head combat using Mario 64's attack moves (Kick, Stomp).

### Key Features

✅ **Dynamic Team System**
- 2 configurable teams (Red vs Blue)
- 5 Marios per team (total 10 agents)
- Team colors are visually distinct

✅ **Combat Mechanics**
- **Kick (B button)**: 20 damage, 2m range
- **Stomp (Z button)**: 15 damage, 2.5m range
- **Knockback**: Enemies are pushed back on hit
- **Health System**: Each Mario starts with 100 HP

✅ **Team Cooperation**
- Agents receive observations about teammates and enemies
- Rewards for staying close to teammates (cooperation bonus)
- Penalties for being isolated
- Shared victory/defeat for the team

✅ **Victory Conditions**
- **Team Elimination**: One team loses all members
- **Time Limit**: Battle ends after 120 seconds
- **Victory Rewards**: Winning team gets +20 reward, losing team gets -5

✅ **Advanced ML-Agents Integration**
- 55 observations per agent (position, velocity, teammates, enemies, health)
- Self-play support for competitive learning
- Curiosity-based exploration rewards
- PPO trainer with optimized hyperparameters

---

## File Structure

```
Assets/ParkourRL/
├── Scripts/
│   ├── TeamBattleEnvironment.cs      # Main battle environment manager
│   ├── TeamBattleAgent.cs            # Individual agent with team logic
│   ├── Editor/
│   │   └── TeamBattleSceneBuilder.cs # Scene auto-builder script
│   └── ... (existing scripts)
│
├── Config/
│   ├── mario_team_battle.yaml        # ML-Agents training configuration
│   └── ... (existing configs)
│
└── Scenes/
    ├── TeamBattle.unity              # Team battle scene (generated)
    ├── TeamBattle_SETUP.md           # Manual setup guide
    └── ... (existing scenes)
```

---

## Quick Start: Creating the Team Battle Scene

### Option A: Automatic Scene Generation (Recommended)

1. **In the Unity Editor**, go to: `Assets` menu → `Create` → `Team Battle Scene`
2. The scene will be automatically generated and saved as `TeamBattle.unity`
3. Press **Play** to start!

### Option B: Manual Scene Setup

See `Assets/ParkourRL/Scenes/TeamBattle_SETUP.md` for detailed step-by-step instructions.

---

## Training Configuration

### ML-Agents Training

```bash
# Train the team battle agent
mlagents-learn Assets/ParkourRL/Config/mario_team_battle.yaml --run-id=team_battle_v1 --num-envs=1

# For faster training with multiple parallel environments (if hardware allows)
mlagents-learn Assets/ParkourRL/Config/mario_team_battle.yaml --run-id=team_battle_v2 --num-envs=4
```

### Training Configuration Details

**File**: `mario_team_battle.yaml`

- **Behavior Name**: `MarioTeamBattle`
- **Trainer**: PPO (Proximal Policy Optimization)
- **Self-Play**: Yes (teams learn against each other)
- **Network**: 3 layers × 512 hidden units
- **Observations**: 55 per agent
- **Actions**:
  - 2 continuous (joystick X, Y)
  - 3 discrete action groups (Jump, Kick, Stomp each with 2 choices)

---

## Component Reference

### TeamBattleEnvironment.cs

**Main responsibility**: Manages the 2 teams, spawning, combat resolution, and battle lifecycle.

**Key Methods**:
```csharp
// Called automatically
void ProcessCombatCollisions()    // Detects kicks/stomps and applies damage

public void DealDamage(TeamBattleAgent defender, float damage, TeamBattleAgent attacker)
public void EliminateAgent(TeamBattleAgent agent)
public void EndBattle()

// Accessors
public List<TeamBattleAgent> GetTeammates(TeamBattleAgent agent)
public List<TeamBattleAgent> GetRivals(TeamBattleAgent agent)
public bool IsOutOfBounds(Vector3 position)
```

**Inspector Settings**:
```
Mario Setup
├── Mario Prefab: Assets/Mario.prefab
└── Base Mario Material: (optional)

Arena Setup
├── Team A Spawn Points: [5 transforms]
├── Team B Spawn Points: [5 transforms]
├── Arena Center: Transform
└── Arena Radius: 35

Battle Settings
├── Mario Per Team: 5
├── Max Battle Time: 120
├── Mario Health: 100
├── Kick Damage: 20
├── Stomp Damage: 15
└── Knockback Force: 5
```

### TeamBattleAgent.cs

**Main responsibility**: Individual agent logic, observations, combat, team-based decision-making.

**Key Methods**:
```csharp
public void SetTeammates(List<TeamBattleAgent> allTeammates)
public void SetRivals(List<TeamBattleAgent> allRivals)
public void TakeDamage(float damage)
public void ApplyKnockback(Vector3 knockbackVector)
```

**Observations** (55 total):
- Self Position: 3 obs
- Self Velocity: 3 obs
- Grounded State: 1 obs
- Raycasts (terrain detection): 16 obs
- Ground Height: 1 obs
- Time Normalized: 1 obs
- Teammates Info: 15 obs (5 teammates × 3: posX, posZ, health)
- Rivals Info: 15 obs (5 enemies × 3: posX, posZ, health)
- Self Health: 1 obs

**Reward System**:
- `-0.01f` per step (encourage fast action)
- `-0.1f` for moving outside arena
- `+0.5f` for hitting an enemy
- `-1.0f` for taking damage
- `-0.05f` for being isolated (no alive teammates)
- `+20f` for team victory
- `-5f` for team defeat

---

## Combat System Details

### How Combat Works

1. **Attack Detection**:
   - When agent presses Kick (B) or Stomp (Z), detection radius expands
   - Kick range: 2.0m
   - Stomp range: 2.5m

2. **Damage Application**:
   - Kick: 20 damage
   - Stomp: 15 damage
   - Damage is cumulative; no armor or resistance

3. **Knockback**:
   - Victims are pushed 5 units in direction away from attacker
   - Knockback is immediate (no animation)

4. **Elimination**:
   - When health ≤ 0, agent is removed from battle
   - Receives -10 reward penalty
   - Episode ends for that agent

### Friendly Fire

- ❌ **NOT ENABLED** in current version
- All attacks only affect enemies
- Teammates are immune to damage

---

## Battle Flow

```
1. Scene loads → TeamBattleManager spawns 10 Marios (5 red, 5 blue)
2. ML-Agents Decision Requester feeds observations every 5 frames
3. Each agent outputs: joystick, kick, stomp actions
4. Combat collisions detected every frame
5. Damage applied, knockback computed
6. Rewards distributed
7. Victory condition checked:
   ├─ One team eliminated? → Battle ends, rewards distributed
   ├─ 120 seconds elapsed? → Battle ends, time-based rewards
   └─ Continue...
8. After 2 seconds delay, scene resets and new battle begins
```

---

## Customization Guide

### Change Number of Players Per Team

**File**: `TeamBattleEnvironment.cs` (line 33)
```csharp
[SerializeField] private int marioPerTeam = 5;  // Change to 4, 6, 10, etc.
```

Also update spawn points in the scene to match.

### Adjust Combat Damage

**File**: `TeamBattleEnvironment.cs` (lines 34-36)
```csharp
[SerializeField] private float kickDamage = 20f;        // Increase for more lethal combat
[SerializeField] private float stompDamage = 15f;       // Decrease for longer battles
[SerializeField] private float knockbackForce = 5f;     // Higher = more airborne
```

### Change Team Colors

**File**: `TeamBattleEnvironment.cs` (lines 45-46)
```csharp
private Color teamAColor = Color.red;
private Color teamBColor = new Color(0, 0.7f, 1f);  // RGB for blue
```

### Modify Reward System

**File**: `TeamBattleAgent.cs` (OnActionReceived method)
- Edit reward values for different behaviors
- Currently emphasizes hitting enemies and team cooperation

### Extend Battle Time

**File**: `TeamBattleEnvironment.cs` (line 27)
```csharp
[SerializeField] private float maxBattleTime = 120f;  // Increase to 180f for longer battles
```

---

## Performance Tips

1. **Reduce Raycast Count**: In `TeamBattleAgent`, change `raycastCount = 8` to `4`
2. **Increase Decision Period**: In scene setup, change Decision Requester `DecisionPeriod` from 5 to 10
3. **Disable Visualization**: In Project Settings > Graphics, reduce quality level
4. **Use Headless Mode**: Add `--no-graphics` flag to mlagents-learn command

---

## Troubleshooting

### "Mario Prefab not found"
- Check that `Assets/Mario.prefab` exists
- If not, verify libsm64-unity is properly imported

### Marios not spawning
- Open TeamBattle.unity scene
- Check that spawn points are correctly assigned in TeamBattleManager Inspector

### No combat happening
- Verify raycast distance in detection (2.0-2.5m)
- Check that Kick/Stomp buttons are being pressed
- Monitor console for combat debug messages

### Scene freezes
- Reduce number of agents per team
- Increase Decision Period to 10
- Check for infinite loops in OnActionReceived

### Training not converging
- Increase learning_rate from 0.0003 to 0.0005
- Adjust reward magnitudes (currently -10 to +20)
- Enable curiosity reward (already enabled)

---

## Future Enhancements

Possible improvements for future versions:

- [ ] **Team Communication**: Agents observe team intentions/coordination status
- [ ] **Power-ups**: Health pickups, temporary damage boost
- [ ] **Dynamic Arena**: Obstacles, moving platforms
- [ ] **Multi-Team Battles**: 3+ teams competing simultaneously
- [ ] **Friendly Fire**: Toggle to enable damage to teammates
- [ ] **Stamina System**: Limited number of attacks per time period
- [ ] **Team Strategies**: Different unit types (attacker, defender, healer)
- [ ] **Replay System**: Record and playback notable battles

---

## Technical Specs

**Scene Requirements**:
- Minimum 2 spawn points per team
- SM64StaticTerrain for arena
- MeshColliders on all terrain
- SM64Context (auto-created on first spawn)

**Agent Requirements**:
- SM64Mario component
- MarioInputProvider component
- TeamBattleAgent component
- BehaviorParameters (configured automatically)
- DecisionRequester (configured automatically)

**Memory**: ~500MB for 10 agents training simultaneously

**GPU**: Optional (PPO can train on CPU)

---

## References

- **ML-Agents Documentation**: https://github.com/Unity-Technologies/ml-agents
- **libsm64-unity**: Uses native SM64 emulator for realistic Mario physics
- **Mario 64 Combat System**: Implemented via SM64 Button.Kick and Button.Stomp

---

## License

Part of libsm64-unity-dev project. Follows original project license.

---

**Created**: April 2026  
**Last Updated**: April 2026  
**Status**: ✅ Ready for training

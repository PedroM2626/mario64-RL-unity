# Team Battle Scene Setup Guide

## Quick Setup Option 1: Using the Scene Builder Script

1. In the Unity Editor, create a new empty GameObject in the Hierarchy
2. Add the `TeamBattleSceneBuilder` script component to it
3. In the Inspector, click "Build Team Battle Scene"
4. The scene will be automatically configured

## Manual Setup Option 2: Step-by-Step

### Step 1: Create New Scene
- File > New Scene > Basic (Empty)
- Save as: `Assets/ParkourRL/Scenes/TeamBattle.unity`

### Step 2: Create Arena Geometry
1. **Main Arena Platform** (large central platform):
   - GameObject > 3D Object > Cube
   - Name: `ArenaPlatform`
   - Position: (0, 0, 0)
   - Scale: (40, 1, 40)
   - Add Material with gray/brown color
   - Add Box Collider
   - Add SM64StaticTerrain component

2. **Arena Walls** (optional boundaries):
   - Create 4 cube walls around arena
   - Scale and position appropriately

### Step 3: Create Spawn Points

**Team A (Red) Spawn Zone:**
- Create empty GameObject: `TeamASpawns`
- Create 5 child empty GameObjects: `Spawn_A_0` through `Spawn_A_4`
- Position them in a semi-circle on left side of arena (e.g., x: -15 to -5)
- Positions:
  - Spawn_A_0: (-18, 2, -10)
  - Spawn_A_1: (-18, 2, -5)
  - Spawn_A_2: (-18, 2, 0)
  - Spawn_A_3: (-18, 2, 5)
  - Spawn_A_4: (-18, 2, 10)

**Team B (Blue) Spawn Zone:**
- Create empty GameObject: `TeamBSpawns`
- Create 5 child empty GameObjects: `Spawn_B_0` through `Spawn_B_4`
- Position them on right side (e.g., x: 5 to 15)
- Positions:
  - Spawn_B_0: (18, 2, -10)
  - Spawn_B_1: (18, 2, -5)
  - Spawn_B_2: (18, 2, 0)
  - Spawn_B_3: (18, 2, 5)
  - Spawn_B_4: (18, 2, 10)

### Step 4: Create Arena Center Marker
- Create empty GameObject: `ArenaCenter`
- Position: (0, 0, 0)
- This marks the center point and radius for boundary checking

### Step 5: Create Main Manager
- Create empty GameObject: `TeamBattleManager`
- Add TeamBattleEnvironment script
- Set the following in Inspector:
  - **Mario Setup**:
    - Mario Prefab: Assets/Mario.prefab
    - Base Mario Material: (find a suitable material, or leave empty)
  - **Arena Setup**:
    - Team A Spawn Points: Select the 5 Spawn_A_X transforms
    - Team B Spawn Points: Select the 5 Spawn_B_X transforms
    - Arena Center: ArenaCenter
    - Arena Radius: 35
  - **Battle Settings**:
    - Mario Per Team: 5
    - Max Battle Time: 120
    - Mario Health: 100
    - Kick Damage: 20
    - Stomp Damage: 15
    - Knockback Force: 5

### Step 6: Create Lighting
- GameObject > Light > Directional Light
- Position: (10, 10, 10)
- Rotation: (45, -30, 0)
- Intensity: 1.5

### Step 7: Create Camera (optional, for visualization)
- Main Camera should exist by default
- Position: (0, 15, -25)
- Rotation: (20, 0, 0)

### Step 8: Tag Setup
- Make sure "Goal" tag exists in Project Settings > Tags
- (Not strictly necessary for team battle, but good to have)

### Step 9: Configure SM64 Context
- Verify there's an SM64Context in the scene (should be created automatically on first Mario spawn)

## Expected Scene Structure

```
TeamBattle (Scene)
├── ArenaPlatform (Cube with SM64StaticTerrain)
├── TeamASpawns (Empty, with 5 children)
│   ├── Spawn_A_0 through Spawn_A_4
├── TeamBSpawns (Empty, with 5 children)
│   ├── Spawn_B_0 through Spawn_B_4
├── ArenaCenter (Empty)
├── TeamBattleManager (with TeamBattleEnvironment script)
├── DirectionalLight
├── Camera
└── [Generated at runtime: Mario_TeamA_0, Mario_TeamA_1, etc.]
```

## Testing

1. In Play Mode:
   - 10 Marios should spawn (5 red on left, 5 blue on right)
   - They should start moving and engaging in combat
   - Red vs Blue teams should be visible

2. Expected Behavior:
   - Marios use Kick (B) and Stomp (Z) to attack
   - When hit, enemies take damage and get knockback
   - Battle ends when one team is eliminated or time runs out
   - Scene resets automatically

## Troubleshooting

**Marios not visible:**
- Check SM64Mario material assignment
- Verify libsm64-unity is properly imported

**No combat happening:**
- Check TeamBattleEnvironment.ProcessCombatCollisions() is being called
- Verify raycast detection distance (2.5f for range)

**Scene crashes:**
- Ensure all spawn points are assigned
- Check for null references in Inspector

**Performance issues:**
- Reduce raycast count in TeamBattleAgent
- Reduce decision period (currently 5)

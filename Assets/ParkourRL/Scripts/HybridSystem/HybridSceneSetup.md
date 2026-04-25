# Setup da Cena HybridTraining

## Visão Geral

A cena **HybridTraining** permite treinar um agente ML-Agents (AI Mario - azul) lado a lado com um Mario controlado pelo player (verde). O sistema suporta dois modos:

- **Training Mode**: Treino RL normal com ML-Agents
- **Recording Mode**: Grava dados do player para Imitation Learning e Offline RL

## Método Automático (Recomendado)

### Usando o Hybrid Scene Builder

1. No Unity Editor, vá no menu: **`ParkourRL > Build Hybrid Training Scene`**
2. Aguarde a cena ser criada automaticamente
3. Quando aparecer a mensagem "Setup Complete", pressione **Play** para testar!

### Configurar Referências (se necessário)

Se alguma referência não foi atribuída automaticamente:

1. No menu: **`ParkourRL > Setup Hybrid Scene References`**
2. Verifique no Inspector do GameObject "HybridEnvironment":
   - `Goal` deve apontar para "GoalPlatform"
   - `Player Spawn Point` deve apontar para "PlayerSpawn"
   - `AI Spawn Point` deve apontar para "AISpawn"
   - `Data Recorder` e `Training Manager` devem estar conectados

## Método Manual (Alternativo)

Se preferir criar a cena manualmente, siga os passos abaixo:

### 1. Criar Nova Cena

1. No Unity, vá em `File > New Scene`
2. Salve como: `Assets/ParkourRL/Scenes/HybridTraining.unity`

### 2. Estrutura de GameObjects

Crie a seguinte hierarquia:

```
HybridTraining (Scene)
├── HybridEnvironment (Empty GameObject)
│   ├── HybridParkourEnvironment.cs
│   ├── HybridDataRecorder.cs
│   ├── HybridTrainingManager.cs
│   └── Spawn Points:
│       ├── PlayerSpawn (Transform - posição x: -7, y: 2, z: 0)
│       └── AISpawn (Transform - posição x: 7, y: 2, z: 0)
│
├── Parkour Level (copiar do ParkourTraining.unity)
│   ├── StartPlatform (SM64StaticTerrain)
│   ├── Platform_1 (SM64StaticTerrain)
│   ├── Platform_2 (SM64StaticTerrain)
│   ├── Platform_3 (SM64StaticTerrain)
│   └── GoalPlatform (com trigger collider + tag "Goal")
│
├── UI Canvas
│   ├── ModeText (Text UI - canto superior)
│   ├── InfoText (Text UI - centro superior)
│   ├── TrainingButton (Button)
│   ├── RecordingButton (Button)
│   └── StatsPanel (painel com estatísticas)
│
└── Cameras
    ├── PlayerCamera (Camera - viewport rect: 0,0,0.5,1)
    └── AICamera (Camera - viewport rect: 0.5,0,0.5,1)
```

### 3. Configuração dos Componentes

#### HybridParkourEnvironment

```csharp
// Inspector settings:
Mario Prefab: Assets/Mario.prefab
Player Material: (deixar null - será criado automaticamente)
AI Material: (deixar null - será criado automaticamente)
Goal: (arrastar GoalPlatform)
Player Spawn Point: (arrastar PlayerSpawn)
AI Spawn Point: (arrastar AISpawn)
Mario Spacing: 15
Sync Resets: true
Show Comparison UI: true
Data Recorder: (arrastar HybridDataRecorder da cena)
Training Manager: (arrastar HybridTrainingManager da cena)
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
UI References: (arrastar elementos da UI)
Environment: (arrastar HybridParkourEnvironment)
Data Recorder: (arrastar HybridDataRecorder)
```

### 4. Configuração do Parkour

Copie as plataformas da cena `ParkourTraining.unity`:

1. Abra `ParkourTraining.unity`
2. Selecione todas as plataformas (SM64StaticTerrain)
3. Copie (Ctrl+C)
4. Abra `HybridTraining.unity`
5. Cole (Ctrl+V)

Adicione **SM64StaticTerrain** em todas as plataformas se ainda não tiver.

### 5. Configuração do Goal

O Goal deve ter:
- **Transform**: Na última plataforma
- **Collider**: BoxCollider ou SphereCollider com `isTrigger = true`
- **Tag**: "Goal"

### 6. Configuração YAML

Crie `Assets/ParkourRL/Config/mario_parkour_hybrid.yaml`:

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

### 7. Script para Build/Play

Crie `train_hybrid.bat`:

```batch
@echo off
cd /d "c:\Users\pedro\Downloads\libsm64-unity-dev-master"
SET PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

"C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe" ^
    Assets/ParkourRL/Config/mario_parkour_hybrid.yaml ^
    --run-id=mario_hybrid_v1 ^
    --force

pause
```

## Como Usar

### Modo Training (Padrão)

1. Abra a cena `HybridTraining.unity`
2. Clique em **Play**
3. Execute o `train_hybrid.bat`
4. AI Mario (azul) treina com ML-Agents
5. Player Mario (verde) pode ser controlado com **WASD + Espaço**
6. Ambos competem para ver quem completa o parkour primeiro

### Modo Recording

1. Pressione **M** ou clique em "Recording Mode"
2. Controle o Mario verde com **WASD + Espaço**
3. Os dados são gravados automaticamente em:
   ```
   HybridTrainingData/hybrid_episodes_YYYYMMDD_HHMMSS_batch0.json
   HybridTrainingData/hybrid_episodes_YYYYMMDD_HHMMSS_batch0.csv
   ```
4. Complete o parkour (ou falhe) - o episódio será salvo
5. Os dados contêm: observações, ações, recompensas, flags de término

### Usar Dados Gravados

Os dados podem ser usados para:

1. **Imitation Learning (Behavior Cloning)**
   ```python
   # Treinar rede para prever ações dadas observações
   python train_behavior_cloning.py --data HybridTrainingData/
   ```

2. **Offline RL (CQL, IQL)**
   ```python
   # Treinar sem ambiente, apenas dos dados
   python train_offline_rl.py --data HybridTrainingData/ --algo CQL
   ```

3. **Warm Start para RL**
   ```python
   # Inicializar PPO com pesos do IL
   python train_hybrid.py --warm-start model_bc.pth
   ```

## Controles

| Tecla | Ação |
|-------|------|
| **WASD** / **Setas** | Mover Mario Player |
| **Espaço** | Pular |
| **M** | Alternar modo (Training/Recording) |
| **P** | Pausar/Resumar |
| **R** | Resetar ambos os Marios |

## Arquitetura dos Dados

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
          "observations": [0.1, 0.2, ...],  // 30 floats
          "actions": [0.5, 0.3, 0.0],       // [joystick_x, joystick_y, jump]
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

### Problema: AI Mario não se move

**Solução**: Verifique se:
1. ML-Agents está rodando (`train_hybrid.bat`)
2. Behavior Name no YAML corresponde ao do agente (`MarioHybrid`)
3. O modo está em "Training", não "Recording"

### Problema: Dados não estão sendo salvos

**Solução**: Verifique se:
1. Diretório `HybridTrainingData/` existe (criar manualmente se necessário)
2. Modo está em "Recording"
3. Episódio termina (success ou failure)
4. Verifique console por erros de permissão de arquivo

### Problema: Câmera dividida não funciona

**Solução**: 
1. Verifique se as câmeras têm Viewport Rect diferentes
2. PlayerCamera: Rect(0, 0, 0.5, 1)
3. AICamera: Rect(0.5, 0, 0.5, 1)

## Próximos Passos

1. Gravar 50+ episódios de demonstração humana
2. Treinar Behavior Cloning com os dados
3. Usar modelo BC como warm-start para PPO
4. Aplicar DAgger para melhorar com novos dados

---

**Nota**: Este sistema é uma implementação completa de arquitetura híbrida IL+RL. Os dados gerados são compatíveis com algoritmos padrão de Offline RL (CQL, IQL, BCQ).

# libsm64-unity-dev - Parkour RL + Hybrid Training System

Sistema completo para treinar agentes de IA em ambientes de parkour do Super Mario 64 usando **Reinforcement Learning**, **Imitation Learning** e **Offline RL**.

---

## 📋 Índice

- [Setup Inicial](#setup-inicial)
- [Arquitetura do Projeto](#arquitetura-do-projeto)
- [Sistemas de Treinamento](#sistemas-de-treinamento)
- [Cenas Disponíveis](#cenas-disponíveis)
- [Instruções de Uso](#instruções-de-uso)
- [API e Componentes](#api-e-componentes)
- [Troubleshooting](#troubleshooting)

---

## Setup Inicial

### Pré-requisitos

- **Unity**: 2019.3.10 ou superior
- **Python**: 3.8 ou 3.9 (recomendado 3.9)
- **ROM**: Super Mario 64 US (MD5: `20b854b239203baf6c961b850a4a51a2`)

### Instalação

1. Clone com submodulos:
    ```bash
    git clone https://github.com/libsm64/libsm64-unity-dev
    cd libsm64-unity-dev
    git submodule update --init --recursive
    ```

2. Coloque a ROM `baserom.us.z64` na raiz do projeto:
    ```
    libsm64-unity-dev/
    ├── baserom.us.z64    ← Coloque aqui
    ├── Assets/
    └── ...
    ```

3. Abra o projeto no Unity

4. Instale dependências Python:
    ```bash
    pip install -r requirements.txt
    ```

5. (Opcional) Configure virtual environment:
    ```powershell
    # PowerShell
    .\setup_python39.ps1  # ou setup_python38.ps1
    ```

---

## Arquitetura do Projeto

```
libsm64-unity-dev/
├── Assets/
│   ├── ParkourRL/
│   │   ├── Scripts/
│   │   │   ├── BackupSystem/
│   │   │   │   ├── BackupParkourEnvironment.cs     [Sistema de Recording]
│   │   │   │   ├── BackupMarioRLAgent.cs
│   │   │   │   └── BackupCheckpoint.cs
│   │   │   ├── HybridSystem/
│   │   │   │   ├── MarioHybridAgent.cs             [Agente Híbrido]
│   │   │   │   ├── HybridParkourEnvironment.cs
│   │   │   │   ├── HybridDataRecorder.cs           [Grava dados]
│   │   │   │   ├── HybridTrainingManager.cs        [Gerencia modos]
│   │   │   │   ├── README_HYBRID.md                [Docs]
│   │   │   │   ├── QUICKSTART.md
│   │   │   │   └── HybridSceneSetup.md
│   │   │   ├── ParkourEnvironment.cs               [RL tradicional]
│   │   │   ├── MarioRLAgent.cs
│   │   │   ├── MarioInputProvider.cs
│   │   │   ├── PlayerMarioSpawner.cs
│   │   │   ├── CompetitiveParkourEnvironment.cs    [Modo competitivo]
│   │   │   └── TeamBattleEnvironment.cs            [Modo time]
│   │   ├── Scenes/
│   │   │   ├── ParkourTraining.unity               [RL Padrão]
│   │   │   ├── ParkourTraining_OldSystem.unity
│   │   │   ├── HybridTraining.unity                [IL + RL]
│   │   │   ├── CompetitiveParkour.unity
│   │   │   └── TeamBattle.unity
│   │   ├── Config/
│   │   │   ├── mario_parkour.yaml                  [Config RL]
│   │   │   └── mario_parkour_hybrid.yaml           [Config Híbrida]
│   │   └── Prefabs/
│   │       └── Mario.prefab
│   └── libsm64-unity/                              [Wrapper nativo]
│
├── python_trainers/
│   ├── train_behavior_cloning.py                   [Imitation Learning]
│   ├── train_offline_rl.py                         [Offline RL (CQL/IQL)]
│   ├── analyze_dataset.py                          [Análise de dados]
│   └── requirements.txt
│
├── HybridTrainingData/                             [Dados gravados]
├── results/                                        [Modelos treinados]
├── train_hybrid.ps1                                [Script treinamento]
├── train_mario.ps1
├── trainer_mlflow.py                               [MLOps]
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Sistemas de Treinamento

### 1. **Parkour RL (Tradicional)**

Treinamento com **4 Marios paralelos** usando PPO.

**Componentes:**
- `ParkourEnvironment.cs` - Orquestrador
- `MarioRLAgent.cs` - Agente ML-Agents
- `MarioInputProvider.cs` - Controle
- `BackupMarioRLAgent.cs` - Versão simplificada

**Início:**
```powershell
# Terminal 1: Iniciar trainer
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2: Play na cena ParkourTraining.unity
```

**Observações (30-dim):**
| Campo | Índices | Valores |
|-------|---------|---------|
| Posição (x,y,z) | 0-2 | [-1, 1] |
| Direção ao goal (x,y,z,dist) | 3-6 | [-1, 1] |
| Velocidade (x,y,z) | 7-9 | [-1, 1] |
| No ar? | 10 | {0, 1} |
| Raycasts 8 direcções | 11-26 | [0, 1] |
| Raycast para baixo | 27 | [0, 1] |
| Pulando? | 28 | {0, 1} |
| Tempo / 30s | 29 | [0, 1] |

**Actions:**
- `[0]` Joystick X: [-1, 1] (esquerda/direita)
- `[1]` Joystick Y: [-1, 1] (frente/trás)
- `[2]` Pular: {0, 1} (discreto)

---

### 2. **Hybrid Training System** 

Combina **Imitation Learning (IL)**, **Offline RL** e **Online RL** em um único fluxo.

**Arquitetura:**
```
Fase 1: Recording (Grava demos do player)
  ↓
Fase 2: Behavior Cloning (Treina rede a imitar)
  ↓
Fase 3: Offline RL (Refina com CQL/IQL)
  ↓
Fase 4: Online RL com Warm-Start (PPO + pesos iniciais)
```

**Componentes:**
- `BackupParkourEnvironment.cs` - Controla modo Recording/Training
- `HybridDataRecorder.cs` - Salva dados em JSON/CSV
- `RecordingInputProvider` - Captura input humano
- `train_behavior_cloning.py` - BC training
- `train_offline_rl.py` - CQL/IQL training

**Arquivo de Dados:**
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
          "observations": [0.1, 0.2, 0.0, ...],  // 30 dims
          "actions": [0.5, 0.0, 1.0],           // [joy_x, joy_y, jump]
          "reward": 2.5,
          "nextObservations": [0.11, 0.21, ...],
          "done": false
        }
      ]
    }
  ]
}
```

**Workflow Completo:**

```powershell
# 1. Gerar dados em Recording mode
# Unity: HybridTraining.unity → Modo Recording (pressione M) → Jogar
# Dados salvos em: HybridTrainingData/hybrid_episodes_*.json

# 2. Analisar dataset
cd python_trainers
python analyze_dataset.py --data ../HybridTrainingData/ --plots

# 3. Treinar Behavior Cloning
python train_behavior_cloning.py \
    --data ../HybridTrainingData/ \
    --epochs 100 \
    --batch-size 32 \
    --output models/bc_mario.pth

# 4. Treinar Offline RL (CQL)
python train_offline_rl.py \
    --data ../HybridTrainingData/ \
    --algo CQL \
    --epochs 50 \
    --output models/cql_mario.pth

# 5. Treinar Online RL com warm-start (próxima versão)
mlagents-learn Assets/ParkourRL/Config/mario_parkour_hybrid.yaml \
    --run-id=mario_hybrid_bc \
    --initialize-from=models/bc_mario.pth
```

---

### 3. **Modo Competitivo**

Dois agentes compete para chegar ao goal primeiro.

- Cena: `CompetitiveParkour.unity`
- Script: `CompetitiveParkourEnvironment.cs`
- Agente: `MarioCompetitiveAgent.cs`

---

### 4. **Modo Team Battle**

Dois times de agentes cooperam/competem.

- Cena: `TeamBattle.unity`
- Script: `TeamBattleEnvironment.cs`
- Agente: `TeamBattleAgent.cs`

---

## Cenas Disponíveis

| Cena | Uso | Modo | Paralelo |
|------|-----|------|----------|
| **ParkourTraining.unity** | RL Padrão | 4 Marios paralelos | ✅ Sim |
| **ParkourTraining_OldSystem.unity** | Sistema antigo | Backup | ❌ Não |
| **HybridTraining.unity** | IL + Offline RL | Recording mode | ✅ Sim |
| **CompetitiveParkour.unity** | Competição 1v1 | 2 Agentes | ✅ Sim |
| **TeamBattle.unity** | Team 2v2 | 4 Agentes | ✅ Sim |
| **ChaseTraining.unity** | Perseguição 1v1 (SAC) | 2 Agentes (Pursuer + Fugitive) | ✅ Sim |

---

## Instruções de Uso

### Opção 1: Treino RL Tradicional (Parkour)

```powershell
# Terminal 1
cd libsm64-unity-dev
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2
# Abra ParkourTraining.unity no Unity → Play
# Aguarde mensagem: "Agent initialized. Ready to accept experiences."
```

**Progresso:**
- Step 0-100k: Mario aprende a se mover
- Step 100k-500k: Mario tenta pular plataformas
- Step 500k+: Mario consegue completar o parkour

---

### Opção 2: Gravação de Dados (Recording Mode)

```powershell
# No Unity
1. Abra Assets/ParkourRL/Scenes/HybridTraining.unity
2. Na Hierarchy, selecione: ParkourEnvironment
3. No Inspector, BackupParkourEnvironment: Startup Mode = Recording
4. Play (Ctrl+P)
5. Controle Mario com WASD + Espaço
6. Complete o parkour (ou falhe)
7. Episódio é salvo em: HybridTrainingData/hybrid_episodes_*.json
```

**Repetir:** Para gravar mais episódios, simplesmente jogue novamente.

---

### Opção 3: Treino com Dados Gravados

```powershell
cd python_trainers

# Analisar dados
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

**Saída:**
- `models/bc_mario.pth` - Modelo treinado (Behavior Cloning)
- `models/cql_mario.pth` - Modelo treinado (Offline RL)
- Gráficos de treinamento em `plots/`

---

### Opção 4: Modo Híbrido (Player + AI lado a lado)

```powershell
# Terminal 1
.\train_hybrid.ps1

# Terminal 2: No Unity
1. Abra HybridTraining.unity
2. Play
3. Player Mario (verde) = WASD + Espaço
4. AI Mario (azul) = ML-Agents (treina em tempo real)
5. Pressione M para alternar entre Training/Recording
```

---

### Opção 5: Treino Chase (Pursuer vs Fugitive - SAC)

Treinamento descentralizado com dois agentes SAC: um perseguidor e um fugitivo.

```powershell
# Terminal 1
.\train_chase.ps1

# Terminal 2: No Unity
1. Abra ChaseTraining.unity
2. Play
3. Aguarde o spawn automático dos agents
```

**Observações (31-dim):**
| Campo | Índices | Valores |
|-------|---------|---------|
| Posição local (x,y,z) | 0-2 | [-1, 1] |
| Velocidade (x,y,z) | 3-5 | [-1, 1] |
| Grounded | 6 | {0, 1} |
| Oponente relativo (x,y,z,dist) | 7-10 | [-1, 1] |
| Velocidade oponente (x,y,z) | 11-13 | [-1, 1] |
| Tempo normalizado | 14 | [0, 1] |
| Raycasts | 15-30 | [0, 1] |

**Actions (5 continuous):**
- `[0]` Joystick X: [-1, 1]
- `[1]` Joystick Y: [-1, 1]
- `[2]` Pular: [0, 1]
- `[3]` Chute/Soco: [0, 1]
- `[4]` Rasteira/Stomp: [0, 1]

**Nota de performance:** O algoritmo SAC faz updates da rede neural a cada passo. Para evitar congelamentos, a configuração usa `time_scale: 1.0`, `steps_per_update: 4` e `DecisionPeriod: 8`.

---

## API e Componentes

### BackupParkourEnvironment.cs

**Enum:**
```csharp
public enum StartupMode { Training = 0, Recording = 1 }
```

**Propriedades:**
```csharp
public Transform marioSpawnPointPublic { get; }  // Spawn do player
public Transform goalPublic { get; }              // Target objetivo
```

**Métodos:**
```csharp
public float[] CollectObservationVector()  // Retorna 30-dim obs
public void ResetEnvironment()              // Reset do episódio
public void SetStartupMode(StartupMode mode)  // Muda modo
```

### HybridDataRecorder.cs

**Métodos Públicos:**
```csharp
void StartEpisode()  // Inicia novo episódio
void RecordStep(float[] obs, float[] actions, float reward, 
                float[] nextObs, bool done)  // Grava um passo
void SaveEpisode(Transition[] transitions, bool success)  // Salva episódio
void FlushBatch()  // Força salvamento em disco
```

**Eventos:**
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
```python
python train_behavior_cloning.py \
    --data HybridTrainingData/ \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output models/bc.pth \
    --test-split 0.2
```

**train_offline_rl.py:**
```python
python train_offline_rl.py \
    --data HybridTrainingData/ \
    --algo CQL  # ou IQL
    --epochs 50 \
    --learning-rate 0.0003 \
    --output models/offline.pth
```

---

## Configuração Avançada

### YAML ML-Agents (mario_parkour_hybrid.yaml)

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

### MLOps (MLflow)

```powershell
# Rastrear metricas automaticamente
python trainer_mlflow.py --run-id mario_parkour_run1

# Visualizar painel
mlflow ui
# Acesse: http://localhost:5000
```

### Docker

```bash
# Build
docker build -t mariorl:latest .

# Run
docker run -it --rm -v ${PWD}/results:/app/results \
    mariorl:latest python trainer_mlflow.py --run-id dockerrun
```

---

## Troubleshooting

### Problema: "Nenhum arquivo JSON encontrado"

**Solução:**
1. Verifique se executou em Recording mode
2. Confirmepasta `HybridTrainingData/` existe
3. Rode pelo menos 1 episódio completo (até 30s ou queda)

### Problema: "Error: No agent found"

**Solução:**
```powershell
# Terminal 1: Iniciar trainer
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2: DEPOIS fazer Play no Unity (aguarde 3s)
```

### Problema: "ROM não encontrada"

**Solução:**
1. Verifique arquivo: `baserom.us.z64`
2. Confirme MD5: `20b854b239203baf6c961b850a4a51a2`
3. Coloque na raiz do projeto

### Problema: "Python module not found"

**Solução:**
```bash
pip install -r requirements.txt
# ou
pip install mlagents==0.28.0 torch numpy scikit-learn matplotlib
```

### Problema: "ModuleNotFoundError: No module named 'mlagents_envs'"

**Solução:**
```powershell
# Windows
pip install --upgrade ml-agents==0.28.0 ml-agents-envs==0.28.0

# ou use venv
.\setup_python39.ps1
```

### Problema: "Unity congela durante treino ChaseTraining (SAC)"

**Causa:** SAC com `steps_per_update: 1` faz updates da rede neural a cada step do ambiente. Com `time_scale` alto, o Python trainer não consegue acompanhar a Unity e fica bloqueado em `DecideAction`.

**Solução (já aplicada na config):**
- `steps_per_update: 4` — reduz frequência de updates da rede
- `time_scale: 1.0` — diminui velocidade de simulação
- `DecisionPeriod: 8` — reduz requisições de decisão por segundo
- `batch_size: 512` — aproveita melhor o hardware em cada update

Se ainda congelar, reduza ainda mais o `time_scale` ou aumente `steps_per_update`.

---

## Referências

- **ML-Agents**: https://github.com/Unity-Technologies/ml-agents
- **libsm64-unity**: https://github.com/libsm64/libsm64-unity
- **PyTorch**: https://pytorch.org/
- **MLflow**: https://mlflow.org/

---

## Licença

Este projeto utiliza:
- libsm64-unity (MIT)
- ML-Agents Toolkit (Apache 2.0)
- PyTorch (BSD)

Verifique os respectivos repositórios para detalhes de licença.

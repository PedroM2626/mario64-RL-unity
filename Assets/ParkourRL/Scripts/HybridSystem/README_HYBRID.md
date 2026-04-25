# Sistema Híbrido de Treinamento - Mario Parkour

## Visão Geral

Este sistema implementa uma arquitetura híbrida que combina:

- **Imitation Learning (IL)**: Aprendizado a partir de demonstrações humanas
- **Offline RL**: Treinamento de dados gravados sem ambiente
- **Online RL**: Treinamento tradicional com ML-Agents
- **DAgger**: Dataset Aggregation para melhorar políticas iterativamente

## Componentes

### Scripts Unity

| Script | Função |
|--------|--------|
| `MarioHybridAgent.cs` | Agente que suporta modo Treino e Gravação |
| `HybridPlayerMario.cs` | Mario controlado pelo player para gravar demos |
| `HybridParkourEnvironment.cs` | Ambiente que gerencia dois Marios lado a lado |
| `HybridDataRecorder.cs` | Grava e salva transições em JSON/CSV |
| `HybridTrainingManager.cs` | Interface para alternar entre modos |

### Scripts Python

| Script | Função |
|--------|--------|
| `train_behavior_cloning.py` | Treina rede neural para imitar demonstrações |
| `train_offline_rl.py` | Treina com CQL/IQL usando dados gravados |
| `train_dagger.py` | Implementa DAgger para melhorar iterativamente |

## Fluxo de Trabalho

### 1. Fase de Gravação (Recording)

```
Cena: HybridTraining.unity
Modo: Recording (pressione 'M')
Ação: Controle Mario verde com WASD+Space
Saída: Arquivos JSON/CSV em HybridTrainingData/
```

### 2. Fase de Imitation Learning

```bash
# Treinar Behavior Cloning
python train_behavior_cloning.py \
    --data HybridTrainingData/ \
    --epochs 100 \
    --output models/bc_model.pth
```

### 3. Fase de DAgger (Opcional)

```bash
# Iteração 1: Agente tenta, humano corrige
python train_dagger.py \
    --initial-model models/bc_model.pth \
    --iterations 5 \
    --output models/dagger_model.pth
```

### 4. Fase de RL com Warm-Start

```bash
# Treinar PPO inicializando com pesos do BC
mlagents-learn mario_parkour_hybrid.yaml \
    --run-id=mario_hybrid_bc \
    --initialize-from=models/bc_model.pth
```

### 5. Fase de Offline RL (Alternativa)

```bash
# Treinar CQL diretamente dos dados
python train_offline_rl.py \
    --data HybridTrainingData/ \
    --algo CQL \
    --output models/cql_model.pth
```

## Formato dos Dados

### Estrutura JSON

```json
{
  "metadata": {
    "createdAt": "2026-04-25T14:30:00",
    "episodeCount": 100,
    "successRate": 0.68,
    "averageReward": 25.3
  },
  "episodes": [
    {
      "episodeId": 0,
      "success": true,
      "startTime": "2026-04-25T14:30:00",
      "duration": 15.2,
      "stepCount": 304,
      "totalReward": 47.5,
      "transitions": [
        {
          "step": 0,
          "timestamp": 0.0,
          "observations": [0.1, 0.2, 0.0, ..., 0.5],  // 30 obs
          "actions": [0.5, 0.0, 1.0],  // [joy_x, joy_y, jump]
          "reward": -0.01,
          "nextObservations": [0.11, 0.21, ...],
          "done": false
        }
      ]
    }
  ]
}
```

## API dos Componentes

### MarioHybridAgent

```csharp
// Modos de operação
public enum HybridMode { Training, Recording }

// Métodos públicos
void SetMode(HybridMode mode)
void SetDataRecorder(HybridDataRecorder recorder)
void SetEnvironment(HybridParkourEnvironment env)
void SetTargetGoal(Transform goal)
```

### HybridDataRecorder

```csharp
// Eventos
event Action OnEpisodeStarted
event Action<bool> OnEpisodeEnded  // bool = success

// Métodos
void StartEpisode()
void RecordStep(float[] obs, float[] actions, float reward, float[] nextObs, bool done)
void SaveEpisode(Transition[] transitions, bool success)
void FlushBatch()  // Salvar imediatamente
```

### HybridTrainingManager

```csharp
// Modos
public enum HybridMode { Training, Recording }

// Propriedades
HybridMode CurrentMode { get; }

// Métodos
void SetMode(HybridMode mode)
void ToggleMode()
void TogglePause()
void ResetScene()
```

## Integração com ML-Agents

O sistema é totalmente compatível com ML-Agents:

1. **Behavior Name**: `MarioHybrid`
2. **Observations**: 30 dimensões (mesmo formato do MarioRLAgent padrão)
3. **Actions**: 2 contínuas (joystick) + 1 discreta (jump)
4. **Reward**: Mesma estrutura do parkour original

## Arquitetura de Comunicação

```
┌─────────────────────────────────────────────────────────┐
│                    HybridTraining                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │  Player Mario   │    │    AI Mario     │            │
│  │  (WASD+Space)   │    │  (ML-Agents)    │            │
│  │     [GREEN]     │    │     [BLUE]      │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           ▼                      ▼                      │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │ HybridPlayer    │    │ MarioHybridAgent│            │
│  │    Mario        │    │                 │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           └──────────┬───────────┘                      │
│                      ▼                                  │
│           ┌─────────────────────┐                       │
│           │ HybridParkour       │                       │
│           │ Environment         │                       │
│           └──────────┬──────────┘                       │
│                      │                                  │
│           ┌──────────┴──────────┐                       │
│           │                     │                        │
│           ▼                     ▼                        │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ HybridData      │  │ HybridTraining  │              │
│  │ Recorder        │  │ Manager         │              │
│  │                 │  │                 │              │
│  │ JSON/CSV Output │  │ UI / Controls   │              │
│  └─────────────────┘  └─────────────────┘              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Configurações YAML

### mario_parkour_hybrid.yaml

Ver arquivo em `Assets/ParkourRL/Config/mario_parkour_hybrid.yaml`

### Diferenças do parkour padrão:

- **Behavior Name**: `MarioHybrid` (não `MarioParkour`)
- **Curriculum**: Adaptado para treino com player
- **Time Horizon**: 128 (mesmo do padrão)
- **Batch Size**: 512 (menor para adaptação rápida)

## Extensões Futuras

### GAIL (Generative Adversarial IL)

Adicionar discriminador adversarial para distinguir humano vs agente:

```python
# Pseudocódigo
for iteration in range(1000):
    # Treinar discriminador
    loss_disc = distinguish(human_traj, agent_traj)
    
    # Treinar agente para enganar discriminador
    reward = log(discriminator(agent_traj))
    ppo_update(reward + env_reward)
```

### SQIL (Soft Q Imitation Learning)

Modificar recompensas:
- Demonstrações humanas: reward = 1 (máximo)
- Interações agente: reward normal do ambiente

### IQ-Learn

Aprender Q-function implicitamente dos dados:

```python
# Sem necessidade de actions no dataset
# Apenas (obs, next_obs, reward, done)
```

## Troubleshooting

### Dados não aparecem

```bash
# Verificar diretório
ls -la HybridTrainingData/

# Permissões
chmod 755 HybridTrainingData/
```

### Agente não aprende com BC

```python
# Verificar sobreposição de dados
python analyze_dataset.py --data HybridTrainingData/

# Aumentar capacidade da rede
python train_behavior_cloning.py --hidden-size 512 --layers 3
```

### Memória insuficiente

```python
# Processar em batches
python train_offline_rl.py --batch-size 256 --buffer-size 10000
```

## Referências

1. **Behavior Cloning**: Pomerleau (1989) - ALVINN
2. **DAgger**: Ross et al. (2011) - A Reduction of Imitation Learning
3. **GAIL**: Ho & Ermon (2016) - Generative Adversarial IL
4. **CQL**: Kumar et al. (2020) - Conservative Q-Learning
5. **IQL**: Kostrikov et al. (2021) - Implicit Q-Learning

---

**Autor**: Sistema criado para projeto Mario Parkour RL
**Data**: Abril 2026
**Versão**: 1.0

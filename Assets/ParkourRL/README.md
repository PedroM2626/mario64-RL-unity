# Parkour RL - Mario com Reinforcement Learning

Este projeto implementa um ambiente de parkour no Unity onde um agente Mario é treinado usando Reinforcement Learning (RL) para completar obstáculos.

## Estrutura

```
Assets/ParkourRL/
├── Scenes/
│   └── ParkourTraining.unity    # Cena de treinamento pronta
├── Scripts/
│   ├── MarioRLAgent.cs          # Agente ML-Agents que controla o Mario
│   ├── MarioInputProvider.cs    # Input provider para SM64
│   ├── ParkourEnvironment.cs    # Gerenciamento do ambiente
│   └── Checkpoint.cs            # Sistema de checkpoints
├── Config/
│   └── mario_parkour.yaml       # Configuração do treinamento ML-Agents
└── README.md                    # Este arquivo
```

## Requisitos

### Unity Packages necessários:
1. **ML Agents** (`com.unity.ml-agents`) - instale via Package Manager
2. **LibSM64 Unity** - já incluído em `Assets/libsm64-unity/`

### Python (para treinamento):
```bash
pip install mlagents
```

## Setup

### 1. Instalar ML-Agents no Unity

Abra o Unity Package Manager (`Window > Package Manager`) e instale:
- **ML Agents** (com.unity.ml-agents)

Ou adicione manualmente ao `Packages/manifest.json`:
```json
"com.unity.ml-agents": "2.3.0"
```

### 2. Abrir a Cena de Parkour

A cena já está criada em `Assets/ParkourRL/Scenes/ParkourTraining.unity`:

1. Abra a cena no Unity
2. Selecione o objeto `ParkourEnvironment` na hierarquia
3. Arraste o prefab `Assets/Mario.prefab` para o campo `Mario Prefab`
4. Arraste o material `Assets/libsm64-unity/Samples/SampleScene/spicy.mat` para o campo `Mario Material`

O Mario com RL será criado automaticamente quando clicar Play.

### 3. Configurar Tags e Layers

No Unity (`Edit > Project Settings > Tags and Layers`):
- Crie a tag **"Goal"**
- Crie a tag **"Checkpoint"**
- Verifique se a layer **"Default"** (0) está sendo usada para terreno

### 4. Behavior Parameters

O Behavior Parameters é adicionado automaticamente pelo ParkourEnvironment:
- **Behavior Name**: `MarioParkour`
- **Vector Observation**:
  - Space Size: `26` (posição 3 + objetivo 4 + raycasts 16 + ground 1 + tempo 1)
  - Stacked Vectors: `1`
- **Vector Action**:
  - Continuous: `2` (joystick X, joystick Y)
  - Discrete: `3` (Jump, Kick, Stomp - cada um com 2 valores: 0/1)

## Treinamento

### 1. Build para Treinamento (opcional)

Para treinar mais rápido, faça uma build:
1. `File > Build Settings`
2. Adicione a cena `ParkourTraining`
3. Clique em Build e salve como `build/ParkourRL.exe`

### 2. Iniciar Treinamento

#### Opção A: Treinamento na Editor (mais lento)

```bash
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1
```

Depois clique no Play no Unity Editor.

#### Opção B: Treinamento com Build (mais rápido)

```bash
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --env=build/ParkourRL.exe --run-id=mario_parkour_v1 --num-envs=4
```

Isso executa 4 ambientes em paralelo para treinamento mais rápido.

### 3. Monitorar Treinamento

```bash
tensorboard --logdir=results
```

Abra `http://localhost:6006` no navegador.

### 4. Salvar e Usar o Modelo

Após o treinamento, o modelo será salvo em `results/mario_parkour_v1/`. 

Para usar o modelo treinado:
1. Copie o arquivo `.nn` para `Assets/ParkourRL/Models/`
2. No prefab do Mario, adicione o componente **Model Overrider**
3. Arraste o modelo para o campo Model

## Configuração de Recompensas

O sistema de recompensas está configurado em `MarioRLAgent.cs`:

- **+0.1 * distância ganha**: Por se aproximar do objetivo
- **-0.001**: Penalidade por tempo (a cada frame)
- **+2.0**: Ao atingir um checkpoint
- **+10.0**: Ao alcançar o goal
- **-1.0**: Se cair ou timeout (60 segundos)

Para ajustar, edite os valores em `OnActionReceived()`.

## Curriculum Learning

O arquivo `mario_parkour.yaml` inclui curriculum learning:

1. **Fase Easy**: Plataformas estáveis
2. **Fase Medium**: Plataformas com variação moderada
3. **Fase Hard**: Plataformas com alta variação

O agente progride automaticamente quando atinge o threshold de reward.

## Observações do Agente

O agente observa:
- Posição atual (3 valores)
- Vetor para o objetivo (3 valores)
- Distância até o objetivo (1 valor)
- Raycasts em 8 direções (16 valores - distância e altura)
- Distância até o chão (1 valor)
- Tempo restante normalizado (1 valor)

Total: **26 observações**

## Ações do Agente

**Contínuas** (2):
- Joystick X: -1 a 1 (esquerda/direita)
- Joystick Y: -1 a 1 (trás/frente)

**Discretas** (3):
- Jump: 0 ou 1
- Kick: 0 ou 1
- Stomp: 0 ou 1

## Dicas de Treinamento

1. **Comece simples**: Primeiro treine com poucas plataformas
2. **Aumente gradualmente**: Adicione complexidade após o agente aprender o básico
3. **Curriculum é essencial**: Use o curriculum learning para treinamento mais estável
4. **Múltiplos ambientes**: Use `--num-envs=4` ou mais para treinamento paralelo
5. **Time Scale**: Aumente `time_scale` no YAML para treinamento mais rápido
6. **Visualização**: Use `--no-graphics` para treinamento headless mais rápido

## Troubleshooting

### Mario não se move
- Verifique se `SM64Mario` está configurado corretamente
- Confirme que `MarioRLAgent` implementa `SM64InputProvider`
- Verifique se o `marioMaterial` não está null

### Agente não aprende
- Verifique se as recompensas estão sendo dadas (adicione Debug.Log)
- Confirme que Behavior Parameters está configurado corretamente
- Aumente `max_steps` no YAML
- Verifique as observações no Tensorboard

### Colisões não funcionam
- Verifique se as plataformas têm `SM64StaticTerrain`
- Execute `SM64Context.RefreshStaticTerrain()` após mover plataformas
- Confirme que as tags "Goal" e "Checkpoint" estão criadas

### Erro "Could not find file baserom.us.z64"
- Certifique-se de que o arquivo `baserom.us.z64` está na raiz do projeto
- Ou renomeie sua ROM para esse nome

## Referências

- [ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Readme.md)
- [LibSM64 Unity](https://github.com/libsm64/libsm64-unity)
- [LibSM64](https://github.com/libsm64/libsm64)

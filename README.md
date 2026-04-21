# libsm64-unity-dev

Este repositorio contem um projeto Unity que wrapa o [libsm64-unity](https://github.com/libsm64/libsm64-unity) para treinar agentes de IA usando Reinforcement Learning em um ambiente de parkour do Super Mario 64.

## Setup Inicial

1. Clone este repositorio com submodulos:
    ```
    git clone https://github.com/libsm64/libsm64-unity-dev
    cd libsm64-unity-dev
    git submodule update --init --recursive
    ```
2. Obtenha uma copia da ROM US do Super Mario 64 (MD5 `20b854b239203baf6c961b850a4a51a2`)
3. Nomeie como `baserom.us.z64` e coloque na raiz do projeto
4. Abra o projeto no Unity 2019.3.10+
5. Instale dependencias Python: `pip install -r requirements.txt`

## Parkour RL

Ambiente de Reinforcement Learning para treinar Mario a completar percursos de parkour.

### Arquitetura

- **ParkourEnvironment.cs**: Orquestrador do ambiente. Instancia o Mario, gerencia resets de episodio e cria ambientes paralelos para treinamento acelerado.
- **MarioRLAgent.cs**: Agente ML-Agents (PPO). Coleta 30 observacoes (posicao, velocidade, raycasts, etc.) e controla o Mario via joystick + botoes.
- **MarioInputProvider.cs**: Ponte entre as acoes do agente RL e o motor SM64.
- **SM64Mario.cs**: Wrapper nativo que comunica com a DLL do SM64.

### Treinamento Multi-Mario Paralelo

O sistema treina **4 Marios simultaneamente** na mesma cena, cada um em sua copia do percurso. Todos compartilham o mesmo cerebro neural (`MarioParkour`), acelerando o treinamento sem precisar de `TimeScale` alto.

Para treinar:
```powershell
# Terminal 1: Iniciar o trainer
mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id parkour_v1

# Terminal 2: Abrir a cena ParkourRL/Scenes/ParkourTraining.unity e dar Play
```

### Configuracao do nivel

O nivel atual consiste em 4 plataformas com gaps de 1.5-2 unidades, todas no mesmo nivel Y (acessiveis pelo pulo padrao do SM64 Mario). O Goal fica na ultima plataforma com um trigger collider grande para facilitar deteccao.

| Plataforma    | Posicao       | Escala    | Gap anterior |
|---------------|---------------|-----------|--------------|
| StartPlatform | (0, -0.5, 0)  | 6x1x6     | -            |
| Platform_1    | (5, -0.5, 0)  | 3x1x3     | ~2 unidades  |
| Platform_2    | (9.5, -0.5, 0)| 3x1x3     | ~1.5 unidades|
| Platform_3    | (14, 0, 0)    | 3x1x3     | ~2 unidades  |
| GoalPlatform  | (19, 0, 0)    | 5x1x5     | ~2 unidades  |

### MLOps (MLflow)

Para rastrear metricas, logs e modelos automaticamente:
```powershell
python trainer_mlflow.py --run-id mario_parkour_run1
```
Visualize o painel: `mlflow ui`

### Docker

```bash
docker build -t mariorl:latest .
docker run -it --rm -v ${PWD}/results:/app/results mariorl:latest python trainer_mlflow.py --run-id dockerrun
```

### Observacoes do Agente (30 dims)

| Observacao                 | Dims | Range       |
|----------------------------|------|-------------|
| Posicao normalizada        | 3    | [-1, 1]     |
| Direcao ao goal            | 4    | [-1, 1]     |
| Velocidade                 | 3    | [-1, 1]     |
| No ar?                     | 1    | {0, 1}      |
| Raycasts (8 direcoes x 2)  | 16   | [0, 1]      |
| Altura do chao             | 1    | [0, 1]      |
| Pulando?                   | 1    | {0, 1}      |
| Tempo restante             | 1    | [0, 1]      |

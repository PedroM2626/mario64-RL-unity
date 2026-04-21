# libsm64-unity-dev

This repo contains a Unity project that wraps the [libsm64-unity](https://github.com/libsm64/libsm64-unity) package so it can be easily worked on directly inside Unity without going through the package manager.

## Features

- **LibSM64- **Vector Observation**:
  - Space Size: `25` (posição 3 + objetivo 4 + raycasts 16 + ground 1 + tempo 1)**Parkour RL**: Ambiente de Reinforcement Learning para treinar IA no parkour (veja `Assets/ParkourRL/`)

To get started:
- Clone this repository, and recursively clone submodules:
    ```
    git clone https://github.com/libsm64/libsm64-unity-dev
    cd libsm64-unity
    git submodule update --init --recursive
    ```
- Get a copy of the Super Mario 64 \[US\] z64 ROM (MD5 20b854b239203baf6c961b850a4a51a2)
- Name the ROM `baserom.us.z64` and place it in the root folder of this repo/project
- Open the project in Unity 2019.3.10+
- Make sure you have a controller attached
- Open the test scene `Assets/pipescene.unity`
- Run the scene, make sure it's working, stop it, and start poking around.

## Parkour RL

Para treinar um agente Mario usando Reinforcement Learning e seguindo príncipios de MLOps:

1. Veja `Assets/ParkourRL/README.md` para instruções completas
2. Instale o pacote ML-Agents no Unity e as dependências: `pip install -r requirements.txt`
3. Use o menu `Parkour RL > Setup Parkour Scene` para criar a cena
4. O mario é iniciado via script `ParkourEnvironment.cs` e possui um `DecisionRequester`

### Treinamento MLOps (Recomendado)
Para rastrear métricas, logs e modelos automaticamente no [MLflow](https://mlflow.org/):
```powershell
python trainer_mlflow.py --run-id mario_parkour_run1
```
Isso aciona o `mlagents-learn` por trás dos panos e registra o experimento no MLflow. Após treinar, visualize o painel usando `mlflow ui`.

### Como rodar em Docker 
Para utilizar o ambiente do ML-Agents em qualquer lugar (com Python 3.9) sem poluir sua máquina local, o repositório acompanha um `Dockerfile`.
1. Faça build da imagem: `docker build -t mariorl:latest .`
2. Rode o container para chamar o MLOps Tracker: `docker run -it --rm -v ${PWD}/results:/app/results mariorl:latest python trainer_mlflow.py --run-id dockerrun`
(Use os devidos bindings no comando se houver o editor linkado).

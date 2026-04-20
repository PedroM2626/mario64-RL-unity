# libsm64-unity-dev

This repo contains a Unity project that wraps the [libsm64-unity](https://github.com/libsm64/libsm64-unity) package so it can be easily worked on directly inside Unity without going through the package manager.

## Features

- **LibSM64 Unity Integration**: Mario 64 controller and physics in Unity
- **Parkour RL**: Ambiente de Reinforcement Learning para treinar IA no parkour (veja `Assets/ParkourRL/`)

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

Para treinar um agente Mario usando Reinforcement Learning:

1. Veja `Assets/ParkourRL/README.md` para instruções completas
2. Instale o pacote ML-Agents no Unity
3. Instale as dependências Python: `pip install -r requirements.txt`
4. Use o menu `Parkour RL > Setup Parkour Scene` para criar a cena automaticamente
5. Configure o Behavior Parameters no prefab do Mario
6. Inicie o treinamento: `mlagents-learn Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1`

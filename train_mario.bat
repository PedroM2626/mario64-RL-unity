@echo off
REM Script para treinar o Mario no Windows

cd /d "c:\Users\pedro\Downloads\libsm64-unity-dev-master"

REM Definir variável de ambiente para contornar problema do protobuf
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Executar treinamento
"C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe" Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

pause

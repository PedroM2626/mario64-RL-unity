@echo off
REM Script para treinar o Mario no modo hibrido (com ML-Agents)

cd /d "c:\Users\pedro\Downloads\libsm64-unity-dev-master"

REM Definir variável de ambiente para contornar problema do protobuf
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Executar treinamento com config hibrida
"C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe" Assets/ParkourRL/Config/mario_parkour_hybrid.yaml --run-id=mario_hybrid_v1 --force

pause

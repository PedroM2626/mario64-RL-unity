@echo off
REM Script to train Mario in hybrid mode (with ML-Agents)

cd /d "c:\Users\pedro\Downloads\libsm64-unity-dev-master"

REM Set environment variable to work around protobuf issue
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Run training with hybrid config
"C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe" Assets/ParkourRL/Config/mario_parkour_hybrid.yaml --run-id=mario_hybrid_v1 --force

pause

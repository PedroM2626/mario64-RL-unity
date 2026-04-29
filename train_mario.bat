@echo off
REM Script to train Mario on Windows

cd /d "c:\Users\pedro\Downloads\libsm64-unity-dev-master"

REM Set environment variable to work around protobuf issue
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

REM Run training
"C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe" Assets/ParkourRL/Config/mario_parkour.yaml --run-id=mario_parkour_v1 --force

pause

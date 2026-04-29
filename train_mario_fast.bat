@echo off
REM Faster training - without modifying physics or agent
REM Optimizations: no_graphics, larger batch_size, quality_level=0

SET PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe Assets/ParkourRL/Config/mario_parkour_fast.yaml --run-id=marioV2_fast --force

@echo off
REM Treino mais rapido - sem alterar fisica ou agente
REM Otimizacoes: no_graphics, maior batch_size, quality_level=0

SET PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
C:\Users\pedro\AppData\Roaming\Python\Python314\Scripts\mlagents-learn.exe Assets/ParkourRL/Config/mario_parkour_fast.yaml --run-id=marioV2_fast --force

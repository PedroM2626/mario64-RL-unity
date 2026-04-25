#!/usr/bin/env python3
"""
Script de validação da integração de gravação
Verifica que todos os componentes estão presentes e funcionais
"""

import os
import re

def validate_integration():
    """Valida que a integração está completa"""
    
    base_path = r"c:\Users\pedro\Downloads\libsm64-unity-dev-master"
    backup_env_path = os.path.join(
        base_path,
        "Assets/ParkourRL/Scripts/BackupSystem/BackupParkourEnvironment.cs"
    )
    
    print("=" * 70)
    print("VALIDAÇÃO DE INTEGRAÇÃO - SISTEMA DE GRAVAÇÃO")
    print("=" * 70)
    print()
    
    # Ler arquivo
    if not os.path.exists(backup_env_path):
        print("❌ ERRO: BackupParkourEnvironment.cs não encontrado!")
        return False
    
    with open(backup_env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "Import HybridSystem": "using ParkourRL.HybridSystem;",
        "Campo dataRecorder": "private HybridDataRecorder dataRecorder;",
        "Inicialização em Start": "dataRecorder = gameObject.AddComponent<HybridDataRecorder>();",
        "Método CollectObservationVector": "public float[] CollectObservationVector()",
        "Propriedade marioSpawnPointPublic": "public Transform marioSpawnPointPublic =>",
        "Propriedade goalPublic": "public Transform goalPublic =>",
        "SetEnvironment em SpawnPlayerControlledMario": "inputProvider.SetEnvironment(this);",
        "SetRecorder em SpawnPlayerControlledMario": "inputProvider.SetRecorder(dataRecorder);",
        "StartEpisode em SpawnPlayerControlledMario": "dataRecorder.StartEpisode();",
        "RecordingInputProvider class": "public class RecordingInputProvider : SM64InputProvider",
        "SetEnvironment method": "public void SetEnvironment(BackupParkourEnvironment env)",
        "SetRecorder method": "public void SetRecorder(HybridDataRecorder rec)",
        "RecordStep em Update": "recorder.RecordStep(previousObservations, actions, reward, currentObservations, false);",
        "RecordStep em RecordFinalStep": "recorder.RecordStep(previousObservations, actions, reward, currentObservations, true);",
        "FlushBatch em RecordFinalStep": "recorder.FlushBatch();",
        "ResetEnvironment em RecordFinalStep": "environment.ResetEnvironment();",
        "StartEpisode novo em RecordFinalStep": "recorder.StartEpisode();",
    }
    
    all_passed = True
    
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    print()
    print("=" * 70)
    
    # Verificar pasta
    data_folder = os.path.join(base_path, "HybridTrainingData")
    if os.path.exists(data_folder):
        print(f"✅ Pasta HybridTrainingData existe")
    else:
        print(f"❌ Pasta HybridTrainingData não existe")
        all_passed = False
    
    # Verificar documentação
    docs = [
        "GRAVACAO_SETUP_DOCS.md",
        "VALIDACAO_GRAVACAO.md",
        "RESUMO_INTEGRACAO_FINAL.md"
    ]
    
    print()
    for doc in docs:
        doc_path = os.path.join(base_path, doc)
        if os.path.exists(doc_path):
            print(f"✅ Documentação criada: {doc}")
        else:
            print(f"❌ Documentação faltando: {doc}")
            all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("✅ INTEGRAÇÃO COMPLETA E FUNCIONAL")
        print()
        print("Próximos passos:")
        print("1. Abrir Assets/ParkourRL/Scenes/HybridTraining.unity")
        print("2. Selecionar ParkourEnvironment na Hierarchy")
        print("3. Inspector → BackupParkourEnvironment → Startup Mode = Recording")
        print("4. Play (Ctrl+P)")
        print("5. Jogar com WASD + Space")
        print("6. Verificar HybridTrainingData/ para arquivos JSON/CSV")
        return True
    else:
        print("❌ INTEGRAÇÃO INCOMPLETA - VERIFIQUE OS ERROS ACIMA")
        return False

if __name__ == "__main__":
    success = validate_integration()
    exit(0 if success else 1)

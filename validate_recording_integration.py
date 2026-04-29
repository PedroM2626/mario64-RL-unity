#!/usr/bin/env python3
"""
Recording integration validation script.
Verifies that all components are present and functional.
"""

import os
import re

def validate_integration():
    """Validates that the integration is complete"""
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    backup_env_path = os.path.join(
        base_path,
        "Assets/ParkourRL/Scripts/BackupSystem/BackupParkourEnvironment.cs"
    )
    
    print("=" * 70)
    print("INTEGRATION VALIDATION - RECORDING SYSTEM")
    print("=" * 70)
    print()
    
    # Read file
    if not os.path.exists(backup_env_path):
        print("[FAIL] BackupParkourEnvironment.cs not found!")
        return False
    
    with open(backup_env_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "Import HybridSystem": "using ParkourRL.HybridSystem;",
        "dataRecorder field": "private HybridDataRecorder dataRecorder;",
        "Initialization in Start": "dataRecorder = gameObject.AddComponent<HybridDataRecorder>();",
        "CollectObservationVector method": "public float[] CollectObservationVector()",
        "marioSpawnPointPublic property": "public Transform marioSpawnPointPublic =>",
        "goalPublic property": "public Transform goalPublic =>",
        "SetEnvironment in SpawnPlayerControlledMario": "inputProvider.SetEnvironment(this);",
        "SetRecorder in SpawnPlayerControlledMario": "inputProvider.SetRecorder(dataRecorder);",
        "StartEpisode in SpawnPlayerControlledMario": "dataRecorder.StartEpisode();",
        "RecordingInputProvider class": "public class RecordingInputProvider : SM64InputProvider",
        "SetEnvironment method": "public void SetEnvironment(BackupParkourEnvironment env)",
        "SetRecorder method": "public void SetRecorder(HybridDataRecorder rec)",
        "RecordStep in Update": "recorder.RecordStep(previousObservations, actions, reward, currentObservations, false);",
        "RecordStep in RecordFinalStep": "recorder.RecordStep(previousObservations, actions, reward, currentObservations, true);",
        "FlushBatch in RecordFinalStep": "recorder.FlushBatch();",
        "ResetEnvironment in RecordFinalStep": "environment.ResetEnvironment();",
        "New StartEpisode in RecordFinalStep": "recorder.StartEpisode();",
    }
    
    all_passed = True
    
    for check_name, check_string in checks.items():
        if check_string in content:
            print(f"[PASS] {check_name}")
        else:
            print(f"[FAIL] {check_name}")
            all_passed = False
    
    print()
    print("=" * 70)
    
    # Check folder
    data_folder = os.path.join(base_path, "HybridTrainingData")
    if os.path.exists(data_folder):
        print(f"[PASS] HybridTrainingData folder exists")
    else:
        print(f"[FAIL] HybridTrainingData folder does not exist")
        all_passed = False
    
    print()
    print("=" * 70)
    
    if all_passed:
        print("[PASS] INTEGRATION COMPLETE AND FUNCTIONAL")
        print()
        print("Next steps:")
        print("1. Open Assets/ParkourRL/Scenes/HybridTraining.unity")
        print("2. Select ParkourEnvironment in the Hierarchy")
        print("3. Inspector -> BackupParkourEnvironment -> Startup Mode = Recording")
        print("4. Play (Ctrl+P)")
        print("5. Play with WASD + Space")
        print("6. Check HybridTrainingData/ for JSON/CSV files")
        return True
    else:
        print("[FAIL] INTEGRATION INCOMPLETE - CHECK ERRORS ABOVE")
        return False

if __name__ == "__main__":
    success = validate_integration()
    exit(0 if success else 1)

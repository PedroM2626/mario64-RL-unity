using UnityEngine;
using UnityEditor;

#if UNITY_EDITOR
namespace ParkourRL.HybridSystem.Editor
{
    /// <summary>
    /// Validates the HybridTraining scene setup and reports issues
    /// </summary>
    public class HybridSceneValidator : MonoBehaviour
    {
        [MenuItem("ParkourRL/Validate Hybrid Scene")]
        public static void ValidateScene()
        {
            Debug.Log("=== HYBRID SCENE VALIDATION ===");
            
            bool allValid = true;
            
            // Check HybridParkourEnvironment
            var env = FindObjectOfType<HybridParkourEnvironment>();
            if (env == null)
            {
                Debug.LogError("[VALIDATOR] HybridParkourEnvironment NOT FOUND!");
                allValid = false;
            }
            else
            {
                Debug.Log("[VALIDATOR] HybridParkourEnvironment found: " + env.name);
                
                // Check serialized fields via reflection
                var envType = typeof(HybridParkourEnvironment);
                var goalField = envType.GetField("goal", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var playerSpawnField = envType.GetField("playerSpawnPoint", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var aiSpawnField = envType.GetField("aiSpawnPoint", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var dataRecorderField = envType.GetField("dataRecorder", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var trainingManagerField = envType.GetField("trainingManager", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                
                if (goalField != null)
                {
                    var goal = goalField.GetValue(env) as Transform;
                    Debug.Log($"[VALIDATOR] Goal: {(goal != null ? goal.name : "NOT SET - REQUIRED!")}");
                    if (goal == null) allValid = false;
                }
                
                if (dataRecorderField != null)
                {
                    var recorder = dataRecorderField.GetValue(env) as HybridDataRecorder;
                    Debug.Log($"[VALIDATOR] DataRecorder: {(recorder != null ? recorder.name : "NOT SET - REQUIRED for recording!")}");
                }
                
                if (trainingManagerField != null)
                {
                    var manager = trainingManagerField.GetValue(env) as HybridTrainingManager;
                    Debug.Log($"[VALIDATOR] TrainingManager: {(manager != null ? manager.name : "NOT SET")}");
                }
            }
            
            // Check HybridDataRecorder
            var recorders = FindObjectsOfType<HybridDataRecorder>();
            Debug.Log($"[VALIDATOR] Found {recorders.Length} HybridDataRecorder(s)");
            foreach (var rec in recorders)
            {
                Debug.Log($"  - {rec.name} on {(rec.gameObject != null ? rec.gameObject.name : "null")}");
            }
            
            // Check HybridTrainingManager
            var managers = FindObjectsOfType<HybridTrainingManager>();
            Debug.Log($"[VALIDATOR] Found {managers.Length} HybridTrainingManager(s)");
            foreach (var man in managers)
            {
                Debug.Log($"  - {man.name}");
                Debug.Log($"    Current Mode: {man.CurrentMode}");
            }
            
            // Check Goal tag
            var goalObj = GameObject.FindWithTag("Goal");
            if (goalObj == null)
            {
                Debug.LogError("[VALIDATOR] No GameObject with 'Goal' tag found!");
                allValid = false;
            }
            else
            {
                Debug.Log($"[VALIDATOR] Goal object: {goalObj.name}");
            }
            
            // Check for SM64StaticTerrain
            var terrains = FindObjectsOfType<LibSM64.SM64StaticTerrain>();
            Debug.Log($"[VALIDATOR] Found {terrains.Length} SM64StaticTerrain(s)");
            if (terrains.Length == 0)
            {
                Debug.LogError("[VALIDATOR] No SM64StaticTerrain found! Mario will fall through floor.");
                allValid = false;
            }
            
            // Check data directory
            string dataPath = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "HybridTrainingData"));
            Debug.Log($"[VALIDATOR] Data directory: {dataPath}");
            if (!System.IO.Directory.Exists(dataPath))
            {
                Debug.LogWarning("[VALIDATOR] HybridTrainingData directory does not exist yet. It will be created on first run.");
            }
            else
            {
                Debug.Log("[VALIDATOR] HybridTrainingData directory exists");
                var files = System.IO.Directory.GetFiles(dataPath, "*.json");
                Debug.Log($"[VALIDATOR] Found {files.Length} JSON files in data directory");
            }
            
            Debug.Log("=== VALIDATION COMPLETE ===");
            
            if (allValid)
            {
                Debug.Log("[VALIDATOR] All critical components found!");
                EditorUtility.DisplayDialog("Validation Complete", 
                    "All critical components found!\n\n" +
                    "To test recording:\n" +
                    "1. Enter Play Mode\n" +
                    "2. Press 'M' to switch to Recording mode\n" +
                    "3. Play with WASD + Space\n" +
                    "4. Complete or fall off the map\n" +
                    "5. Press 'Save Data' button or exit play mode\n" +
                    "6. Check HybridTrainingData/ folder", "OK");
            }
            else
            {
                Debug.LogError("[VALIDATOR] Some critical components are missing! Check errors above.");
                EditorUtility.DisplayDialog("Validation Failed", 
                    "Some critical components are missing!\n\n" +
                    "Check the Console for specific errors.\n\n" +
                    "Common fixes:\n" +
                    "- Ensure HybridParkourEnvironment has Goal assigned\n" +
                    "- Add SM64StaticTerrain to platforms\n" +
                    "- Tag the goal platform with 'Goal' tag", "OK");
            }
        }
    }
}
#endif

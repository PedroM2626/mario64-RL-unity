using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

#if UNITY_EDITOR
namespace ParkourRL.HybridSystem.Editor
{
    /// <summary>
    /// Sets up HybridTraining components in the current scene
    /// </summary>
    public class HybridSceneSetup : MonoBehaviour
    {
        [MenuItem("ParkourRL/Setup Hybrid Scene Components")]
        public static void SetupHybridScene()
        {
            Debug.Log("=== SETTING UP HYBRID SCENE ===");
            
            // Find or create HybridEnvironment GameObject
            GameObject hybridEnv = GameObject.Find("HybridEnvironment");
            if (hybridEnv == null)
            {
                hybridEnv = new GameObject("HybridEnvironment");
                Debug.Log("[SETUP] Created HybridEnvironment GameObject");
            }
            else
            {
                Debug.Log("[SETUP] Found existing HybridEnvironment");
            }
            
            // Add HybridParkourEnvironment if not present
            var parkourEnv = hybridEnv.GetComponent<HybridParkourEnvironment>();
            if (parkourEnv == null)
            {
                parkourEnv = hybridEnv.AddComponent<HybridParkourEnvironment>();
                Debug.Log("[SETUP] Added HybridParkourEnvironment component");
            }
            
            // Add HybridDataRecorder if not present
            var dataRecorder = hybridEnv.GetComponent<HybridDataRecorder>();
            if (dataRecorder == null)
            {
                dataRecorder = hybridEnv.AddComponent<HybridDataRecorder>();
                Debug.Log("[SETUP] Added HybridDataRecorder component");
            }
            
            // Add HybridTrainingManager if not present
            var trainingManager = hybridEnv.GetComponent<HybridTrainingManager>();
            if (trainingManager == null)
            {
                trainingManager = hybridEnv.AddComponent<HybridTrainingManager>();
                Debug.Log("[SETUP] Added HybridTrainingManager component");
            }
            
            // Find references in scene
            var goal = GameObject.FindWithTag("Goal");
            if (goal == null)
            {
                // Try to find by name
                goal = GameObject.Find("Goal");
            }
            
            var playerSpawn = GameObject.Find("PlayerSpawn");
            var aiSpawn = GameObject.Find("AISpawn");
            
            // Find shared spawn points parent
            var sharedSpawnsParent = GameObject.Find("SharedSpawnPoints");
            
            // Assign via SerializedObject
            SerializedObject envSO = new SerializedObject(parkourEnv);
            
            if (goal != null)
            {
                envSO.FindProperty("goal").objectReferenceValue = goal.transform;
                Debug.Log($"[SETUP] Assigned Goal: {goal.name}");
            }
            else
            {
                Debug.LogWarning("[SETUP] Goal not found! Please tag a GameObject with 'Goal' tag.");
            }
            
            if (playerSpawn != null)
            {
                envSO.FindProperty("playerSpawnPoint").objectReferenceValue = playerSpawn.transform;
                Debug.Log($"[SETUP] Assigned PlayerSpawn: {playerSpawn.name}");
            }
            else
            {
                Debug.LogWarning("[SETUP] PlayerSpawn not found! Please create a GameObject named 'PlayerSpawn'.");
            }
            
            if (aiSpawn != null)
            {
                envSO.FindProperty("aiSpawnPoint").objectReferenceValue = aiSpawn.transform;
                Debug.Log($"[SETUP] Assigned AISpawn: {aiSpawn.name}");
            }
            else
            {
                Debug.LogWarning("[SETUP] AISpawn not found! Please create a GameObject named 'AISpawn'.");
            }
            
            // Connect components to each other
            envSO.FindProperty("dataRecorder").objectReferenceValue = dataRecorder;
            envSO.FindProperty("trainingManager").objectReferenceValue = trainingManager;
            
            // Set some default values
            envSO.FindProperty("marioSpacing").floatValue = 15f;
            envSO.FindProperty("syncResets").boolValue = true;
            envSO.FindProperty("showComparisonUI").boolValue = true;
            
            envSO.ApplyModifiedProperties();
            
            // Setup TrainingManager references
            SerializedObject managerSO = new SerializedObject(trainingManager);
            managerSO.FindProperty("environment").objectReferenceValue = parkourEnv;
            managerSO.FindProperty("dataRecorder").objectReferenceValue = dataRecorder;
            managerSO.ApplyModifiedProperties();
            
            Debug.Log("[SETUP] Connected component references");
            
            // Mark scene as dirty
            EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
            
            Debug.Log("=== SETUP COMPLETE ===");
            
            EditorUtility.DisplayDialog("Hybrid Scene Setup Complete", 
                "Components added and configured!\n\n" +
                "Next steps:\n" +
                "1. Save the scene (Ctrl+S)\n" +
                "2. Enter Play Mode to test\n" +
                "3. Check ParkourRL > Validate Hybrid Scene to verify", "OK");
            
            // Select the HybridEnvironment object
            Selection.activeGameObject = hybridEnv;
        }
        
        [MenuItem("ParkourRL/Create Missing Spawn Points")]
        public static void CreateMissingSpawnPoints()
        {
            Debug.Log("=== CREATING MISSING SPAWN POINTS ===");
            
            bool createdAny = false;
            
            // Find or create PlayerSpawn
            var playerSpawn = GameObject.Find("PlayerSpawn");
            if (playerSpawn == null)
            {
                playerSpawn = new GameObject("PlayerSpawn");
                playerSpawn.transform.position = new Vector3(-7, 2, 0);
                Debug.Log("[SETUP] Created PlayerSpawn at (-7, 2, 0)");
                createdAny = true;
            }
            
            // Find or create AISpawn
            var aiSpawn = GameObject.Find("AISpawn");
            if (aiSpawn == null)
            {
                aiSpawn = new GameObject("AISpawn");
                aiSpawn.transform.position = new Vector3(7, 2, 0);
                Debug.Log("[SETUP] Created AISpawn at (7, 2, 0)");
                createdAny = true;
            }
            
            // Find or create SharedSpawnPoints parent
            var sharedSpawns = GameObject.Find("SharedSpawnPoints");
            if (sharedSpawns == null)
            {
                sharedSpawns = new GameObject("SharedSpawnPoints");
                
                // Create child spawn points
                var shared1 = new GameObject("Shared_0");
                shared1.transform.position = new Vector3(-7, 2, 0);
                shared1.transform.parent = sharedSpawns.transform;
                
                var shared2 = new GameObject("Shared_1");
                shared2.transform.position = new Vector3(7, 2, 0);
                shared2.transform.parent = sharedSpawns.transform;
                
                Debug.Log("[SETUP] Created SharedSpawnPoints with 2 children");
                createdAny = true;
            }
            
            if (createdAny)
            {
                EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
                Debug.Log("=== SPAWN POINTS CREATED ===");
                EditorUtility.DisplayDialog("Spawn Points Created", 
                    "Missing spawn points have been created.\n\n" +
                    "Run 'Setup Hybrid Scene Components' again to connect them.", "OK");
            }
            else
            {
                Debug.Log("[SETUP] All spawn points already exist");
            }
        }
    }
}
#endif

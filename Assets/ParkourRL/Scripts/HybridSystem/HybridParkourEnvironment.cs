using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Hybrid environment: manages two Marios side by side.
    /// One is player-controlled (for recording/IL), the other is AI (for training).
    /// Ambos tentam completar o mesmo parkour sem competir entre si.
    /// </summary>
    public class HybridParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material playerMarioMaterial;
        [SerializeField] private Material aiMarioMaterial;

        [Header("Level Elements")]
        [SerializeField] private Transform goal;
        [SerializeField] private Transform playerSpawnPoint;
        [SerializeField] private Transform aiSpawnPoint;
        [SerializeField] private Transform[] sharedSpawnPoints;  // Fallback if specific ones are not defined

        [Header("Hybrid Settings")]
        [Tooltip("Distance between the two Marios (X axis)")]
        [SerializeField] private float marioSpacing = 15f;
        
        [Tooltip("Sincronizar resets (quando um morre, ambos reiniciam)")]
        [SerializeField] private bool syncResets = true;
        
        [Tooltip("Show comparison UI")]
        [SerializeField] private bool showComparisonUI = true;

        [Header("Recording Components")]
        [SerializeField] private HybridDataRecorder dataRecorder;
        [SerializeField] private HybridTrainingManager trainingManager;
        
        [Header("Mode Settings")]
        [Tooltip("Initial mode - can be changed at runtime via HybridTrainingManager")]
        [SerializeField] private HybridMode initialMode = HybridMode.Recording;
        
        public enum HybridMode
        {
            Recording,    // Apenas 1 Mario (Player controlado)
            Training    // 2 Marios lado a lado: RL (Azul) vs Offline RL/IL (Verde)
        }
        
        public HybridMode CurrentMode { get; private set; }

        // References to Marios
        private GameObject playerMario;
        private GameObject aiMario;
        private HybridPlayerMario playerController;
        private MarioHybridAgent aiAgent;
        
        // Estado
        private Vector3 currentPlayerSpawn;
        private Vector3 currentAISpawn;
        private float playerBestTime = float.MaxValue;
        private float aiBestTime = float.MaxValue;
        private int playerCompletedCount = 0;
        private int aiCompletedCount = 0;

        void Start()
        {
            // Set initial mode from inspector
            CurrentMode = initialMode;
            
            EnsureAllMeshColliders();
            InitializeSpawnPoints();
            
            StartCoroutine(RefreshTerrainAndSpawnMarios());
        }
        
        public void SetMode(HybridMode mode)
        {
            if (CurrentMode == mode) return;
            
            CurrentMode = mode;
            Debug.Log($"[HybridEnv] Mode changed to: {mode}");
            
            // Recreate Marios for new mode
            RecreateMariosForMode();
        }
        
        private void RecreateMariosForMode()
        {
            // Destroy existing
            if (playerMario != null)
            {
                Destroy(playerMario);
                playerMario = null;
                playerController = null;
            }
            if (aiMario != null)
            {
                Destroy(aiMario);
                aiMario = null;
                aiAgent = null;
            }
            
            // Destroy old cameras
            GameObject oldPlayerCam = GameObject.Find("PlayerCamera");
            if (oldPlayerCam != null) Destroy(oldPlayerCam);
            GameObject oldAICam = GameObject.Find("AICamera");
            if (oldAICam != null) Destroy(oldAICam);
            
            // Respawn based on mode
            StartCoroutine(RespawnAfterModeChange(CurrentMode == HybridMode.Recording));
        }

        private IEnumerator RefreshTerrainAndSpawnMarios()
        {
            // Wait multiple physics cycles to ensure colliders are ready
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            
            // Now reload terrain in SM64 with ALL platforms
            Debug.Log("[HybridEnv] Refreshing static terrain...");
            SM64Context.RefreshStaticTerrain();
            
            // Wait for terrain to register in SM64
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            
            Debug.Log("[HybridEnv] Terrain loaded, spawning Marios...");
            
            // Spawn Marios based on current mode
            SpawnMariosBasedOnMode();
        }

        private void InitializeSpawnPoints()
        {
            // Usar spawn points specifics ou calcular a partir do shared
            if (playerSpawnPoint != null)
            {
                currentPlayerSpawn = playerSpawnPoint.position;
                Debug.Log($"[HybridEnv] Using playerSpawnPoint: {currentPlayerSpawn}");
            }
            else if (sharedSpawnPoints != null && sharedSpawnPoints.Length > 0 && sharedSpawnPoints[0] != null)
            {
                currentPlayerSpawn = sharedSpawnPoints[0].position;
                Debug.Log($"[HybridEnv] Using sharedSpawnPoints[0]: {currentPlayerSpawn}");
            }
            else
            {
                currentPlayerSpawn = transform.position + Vector3.left * (marioSpacing / 2f);
                Debug.LogWarning($"[HybridEnv] No player spawn point set! Using fallback: {currentPlayerSpawn}");
            }
            
            if (aiSpawnPoint != null)
            {
                currentAISpawn = aiSpawnPoint.position;
                Debug.Log($"[HybridEnv] Using aiSpawnPoint: {currentAISpawn}");
            }
            else if (sharedSpawnPoints != null && sharedSpawnPoints.Length > 1 && sharedSpawnPoints[1] != null)
            {
                currentAISpawn = sharedSpawnPoints[1].position;
                Debug.Log($"[HybridEnv] Using sharedSpawnPoints[1]: {currentAISpawn}");
            }
            else
            {
                currentAISpawn = transform.position + Vector3.right * (marioSpacing / 2f);
                Debug.LogWarning($"[HybridEnv] No AI spawn point set! Using fallback: {currentAISpawn}");
            }
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            foreach (var terrain in terrains)
            {
                if (terrain.GetComponent<MeshCollider>() == null)
                {
                    MeshFilter mf = terrain.GetComponent<MeshFilter>();
                    if (mf != null && mf.sharedMesh != null)
                    {
                        MeshCollider mc = terrain.gameObject.AddComponent<MeshCollider>();
                        mc.sharedMesh = mf.sharedMesh;
                        mc.convex = false;
                    }
                }
            }
        }

        private void SpawnMariosBasedOnMode()
        {
            // Use the mode set in the inspector (CurrentMode) - it has priority
            bool isRecording = CurrentMode == HybridMode.Recording;
            
            // Sync trainingManager TO this environment (not the other way around)
            // This ensures the inspector choice in HybridParkourEnvironment is respected
            if (trainingManager != null)
            {
                bool envIsRecording = CurrentMode == HybridMode.Recording;
                bool managerIsRecording = trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording;
                
                if (envIsRecording != managerIsRecording)
                {
                    // Force trainingManager to match this environment's mode
                    Debug.Log($"[HybridEnv] Syncing TrainingManager to Environment mode: {CurrentMode}");
                    trainingManager.SetMode(envIsRecording ? HybridTrainingManager.HybridMode.Recording : HybridTrainingManager.HybridMode.Training);
                }
            }
            
            if (isRecording)
            {
                // Recording mode: Only Player Mario
                SpawnPlayerMario();
                Debug.Log("[HybridEnv] Recording mode: Only Player Mario spawned (Player-controlled)");
            }
            else
            {
                // Training mode: Both Marios for comparison
                // Mario 1: RL (Blue) - Training with PPO/SAC
                // Mario 2: Offline RL/IL (Green) - Training with recorded data
                SpawnPlayerMario();  // Green = Offline RL/IL
                SpawnAIMario();    // Blue = Online RL
                Debug.Log("[HybridEnv] Training mode: Both Marios spawned - Green (Offline RL/IL) vs Blue (Online RL)");
            }
        }

        private void SpawnPlayerMario()
        {
            // Destruir anterior se existir
            if (playerMario != null)
            {
                Destroy(playerMario);
                playerMario = null;
                playerController = null;
            }
            
            // Destruir câmera antiga do player se existir
            GameObject oldCam = GameObject.Find("PlayerCamera");
            if (oldCam != null)
            {
                Destroy(oldCam);
            }
            
            // Criar Mario player
            playerMario = new GameObject("Mario_Player_Controlled");
            playerMario.SetActive(false);
            playerMario.transform.position = currentPlayerSpawn + Vector3.up * 2f;
            
            // Adicionar controller
            playerController = playerMario.AddComponent<HybridPlayerMario>();
            playerController.SetEnvironment(this);
            
            // Input provider
            playerMario.AddComponent<MarioInputProvider>();
            
            // SM64Mario
            SM64Mario sm64Mario = playerMario.AddComponent<SM64Mario>();
            Material matToUse = playerMarioMaterial;
            if (matToUse == null && marioPrefab != null)
            {
                var prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    var matField = typeof(SM64Mario).GetField("material",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matToUse = matField.GetValue(prefabMario) as Material;
                }
            }
            
            if (matToUse != null)
            {
                Material playerMat = new Material(matToUse);
                playerMat.color = Color.green;  // Verde = Player
                
                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, playerMat);
            }
            
            // Camera for the player
            GameObject camObj = new GameObject("PlayerCamera");
            Camera cam = camObj.AddComponent<Camera>();
            cam.rect = new Rect(0, 0, 0.5f, 1);  // Metade esquerda da tela
            // Determine camera mode
            bool isRecording = false;
            if (trainingManager != null)
            {
                isRecording = trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording;
            }
            
            // Recording = full screen, Training = left half
            cam.rect = isRecording ? new Rect(0, 0, 1, 1) : new Rect(0, 0, 0.5f, 1);
            playerController.SetCamera(cam);
            
            // Ensure terrain is loaded before activating Mario
            SM64Context.RefreshStaticTerrain();
            
            // Wait a frame for terrain to register
            StartCoroutine(ActivateMarioAfterTerrain(playerMario, sm64Mario, currentPlayerSpawn + Vector3.up * 2f));
            
            // Always connect dataRecorder to playerController (used when mode switches to Recording)
            if (dataRecorder != null)
            {
                playerController.SetDataRecorder(dataRecorder);
                Debug.Log("[HybridEnv] DataRecorder connected to Player Mario");
            }
            else
            {
                Debug.LogWarning("[HybridEnv] DataRecorder is NULL! Recording will not work.");
            }
            
            Debug.Log($"[HybridEnv] Player Mario spawned at {playerMario.transform.position}");
        }

        private void SpawnAIMario()
        {
            // Destruir anterior se existir
            if (aiMario != null)
            {
                Destroy(aiMario);
                aiMario = null;
                aiAgent = null;
            }
            
            // Destruir câmera antiga do AI se existir
            GameObject oldCam = GameObject.Find("AICamera");
            if (oldCam != null)
            {
                Destroy(oldCam);
            }
            
            // Criar Mario AI
            aiMario = new GameObject("Mario_AI_Agent");
            aiMario.SetActive(false);
            aiMario.transform.position = currentAISpawn + Vector3.up * 2f;
            
            // Add hybrid agent
            aiAgent = aiMario.AddComponent<MarioHybridAgent>();
            aiAgent.SetEnvironment(this);
            if (goal != null)
                aiAgent.SetTargetGoal(goal);
            if (dataRecorder != null)
                aiAgent.SetDataRecorder(dataRecorder);
            
            // Input provider
            aiMario.AddComponent<MarioInputProvider>();
            
            // SM64Mario
            SM64Mario sm64Mario = aiMario.AddComponent<SM64Mario>();
            Material matToUse = aiMarioMaterial;
            if (matToUse == null && marioPrefab != null)
            {
                var prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    var matField = typeof(SM64Mario).GetField("material",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matToUse = matField.GetValue(prefabMario) as Material;
                }
            }
            
            if (matToUse != null)
            {
                Material aiMat = new Material(matToUse);
                aiMat.color = Color.blue;  // Azul = AI
                
                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, aiMat);
            }
            
            // Behavior Parameters
            var behaviorParams = aiMario.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            behaviorParams.BehaviorName = "MarioHybrid";
            
            if (trainingManager != null && trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording)
            {
                behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.HeuristicOnly;
            }
            else
            {
                behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            }
            
            behaviorParams.BrainParameters.VectorObservationSize = 30;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2 });
            
            // Decision Requester
            var decisionRequester = aiMario.AddComponent<Unity.MLAgents.DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            
            // Camera for the AI (view-only) - only in Training mode
            GameObject camObj = new GameObject("AICamera");
            Camera cam = camObj.AddComponent<Camera>();
            cam.rect = new Rect(0.5f, 0, 0.5f, 1);  // Metade direita da tela
            
            // Ensure terrain is loaded before activating Mario
            SM64Context.RefreshStaticTerrain();
            
            // Wait a frame for terrain to register
            StartCoroutine(ActivateMarioAfterTerrain(aiMario, sm64Mario, currentAISpawn + Vector3.up * 2f));
            
            // Sincronizar modo
            if (trainingManager != null)
            {
                aiAgent.SetMode(trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording 
                    ? MarioHybridAgent.HybridMode.Recording 
                    : MarioHybridAgent.HybridMode.Training);
            }
            
            Debug.Log($"[HybridEnv] AI Mario spawned at {aiMario.transform.position}");
        }

        private IEnumerator ActivateMarioAfterTerrain(GameObject mario, SM64Mario sm64Component, Vector3 spawnPosition)
        {
            Debug.Log($"[HybridEnv] Activating Mario at spawn position: {spawnPosition}");
            
            // Wait multiple physics updates so terrain is fully registered
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            
            if (mario == null) 
            {
                Debug.LogError("[HybridEnv] Mario GameObject is null before activation!");
                yield break;
            }
            
            // Reset position one more time before activation
            mario.transform.position = spawnPosition;
            
            // Now activate
            mario.SetActive(true);
            
            // Wait for SM64 to initialize
            yield return new WaitForFixedUpdate();
            yield return null;
            
            if (mario == null) yield break;
            
            // Teleport to ensure correct position in SM64 physics
            if (sm64Component != null && sm64Component.isActiveAndEnabled)
            {
                sm64Component.Teleport(spawnPosition);
                Debug.Log($"[HybridEnv] Teleported Mario to {spawnPosition}");
            }
            
            // Validate position after spawn
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            
            if (mario == null) yield break;
            
            Vector3 currentPos = mario.transform.position;
            float heightDiff = currentPos.y - spawnPosition.y;
            
            if (heightDiff < -1.0f || currentPos.y < -5f)
            {
                Debug.LogWarning($"[HybridEnv] Mario spawned below expected position! Expected Y={spawnPosition.y}, Actual Y={currentPos.y}. Repositioning...");
                
                // Hard reset - deactivate, reposition, reactivate
                mario.SetActive(false);
                yield return null;
                mario.transform.position = spawnPosition;
                yield return new WaitForFixedUpdate();
                mario.SetActive(true);
                yield return new WaitForFixedUpdate();
                
                if (sm64Component != null && sm64Component.isActiveAndEnabled)
                {
                    sm64Component.Teleport(spawnPosition);
                }
                
                // Check again
                yield return new WaitForFixedUpdate();
                currentPos = mario.transform.position;
                if (currentPos.y < spawnPosition.y - 1.0f)
                {
                    Debug.LogError($"[HybridEnv] Failed to properly spawn Mario! Final position: {currentPos}");
                }
            }
            else
            {
                Debug.Log($"[HybridEnv] Mario activated successfully at {currentPos}");
            }
        }

        public void ResetEnvironment()
        {
            // Determine if we're in recording mode
            bool isRecording = false;
            if (trainingManager != null)
            {
                isRecording = trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording;
            }
            
            // In recording mode, only respawn player
            // In training mode, respawn both
            if (playerMario != null && playerController != null)
            {
                playerController.ResetPlayer();
            }
            
            if (!isRecording && aiMario != null && aiAgent != null)
            {
                // Reset AI Mario with proper teleport (only in training mode)
                Vector3 respawnPosition = GetCurrentSpawnPoint() + Vector3.up * 2f;
                
                aiMario.transform.position = respawnPosition;
                
                // Reset rigidbody velocity
                Rigidbody rb = aiMario.GetComponent<Rigidbody>();
                if (rb != null)
                {
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;
                }
                
                // Use Teleport if available
                SM64Mario sm64Mario = aiMario.GetComponent<SM64Mario>();
                if (sm64Mario != null && sm64Mario.isActiveAndEnabled)
                {
                    sm64Mario.Teleport(respawnPosition);
                }
                
                // Validate position after respawn
                StartCoroutine(ValidateRespawnPosition(aiMario, respawnPosition));
                
                // DO NOT call EndEpisode() here - causes infinite recursion!
            }
            else if (isRecording && aiMario != null)
            {
                // In recording mode, destroy AI mario if it exists
                Destroy(aiMario);
                aiMario = null;
                aiAgent = null;
            }
        }

        private IEnumerator ValidateRespawnPosition(GameObject mario, Vector3 expectedPosition)
        {
            yield return new WaitForFixedUpdate();

            if (mario == null)
                yield break;

            Vector3 marioPos = mario.transform.position;
            float horizontalDistance = Vector3.Distance(
                new Vector3(marioPos.x, 0f, marioPos.z),
                new Vector3(expectedPosition.x, 0f, expectedPosition.z)
            );

            bool invalidRespawn = horizontalDistance > 1.5f || marioPos.y < (expectedPosition.y - 1.0f);
            if (!invalidRespawn)
                yield break;

            Debug.LogWarning($"[HybridEnv] Inconsistent AI respawn detected. Forcing repositioning. Esperado={expectedPosition} Atual={marioPos}");

            mario.transform.position = expectedPosition;
            SM64Mario sm64Mario = mario.GetComponent<SM64Mario>();
            if (sm64Mario != null && sm64Mario.isActiveAndEnabled)
            {
                sm64Mario.Teleport(expectedPosition);
            }
        }

        public void OnPlayerReachedGoal(float time)
        {
            playerCompletedCount++;
            if (time < playerBestTime)
            {
                playerBestTime = time;
            }
            
            Debug.Log($"[HybridEnv] Player completou em {time:F2}s! Melhor: {playerBestTime:F2}s");
            
            if (syncResets)
            {
                // Aguardar um pouco e resetar ambos
                Invoke(nameof(ResetEnvironment), 2f);
            }
        }

        public void OnAIReachedGoal(float time)
        {
            aiCompletedCount++;
            if (time < aiBestTime)
            {
                aiBestTime = time;
            }
            
            Debug.Log($"[HybridEnv] AI completou em {time:F2}s! Melhor: {aiBestTime:F2}s");
        }

        public Vector3 GetCurrentSpawnPoint()
        {
            return currentAISpawn;  // Para o agente AI
        }

        public void SetMode(HybridTrainingManager.HybridMode mode)
        {
            Debug.Log($"[HybridEnv] SetMode called: {mode}");
            
            // When switching modes, we need to recreate the Marios
            // because the number of Marios differs between modes
            bool isRecording = mode == HybridTrainingManager.HybridMode.Recording;
            
            if (aiAgent != null)
            {
                aiAgent.SetMode(isRecording 
                    ? MarioHybridAgent.HybridMode.Recording 
                    : MarioHybridAgent.HybridMode.Training);
            }
            
            // Ensure dataRecorder is connected to player when in Recording mode
            if (playerController != null && isRecording)
            {
                if (dataRecorder != null)
                {
                    playerController.SetDataRecorder(dataRecorder);
                    Debug.Log("[HybridEnv] DataRecorder connected to Player in Recording mode");
                }
            }
            
            // Recreate Marios to apply mode change (1 mario for Recording, 2 for Training)
            // Destroy existing first
            if (playerMario != null)
            {
                Destroy(playerMario);
                playerMario = null;
                playerController = null;
            }
            if (aiMario != null)
            {
                Destroy(aiMario);
                aiMario = null;
                aiAgent = null;
            }
            
            // Respawn based on new mode
            StartCoroutine(RespawnAfterModeChange(isRecording));
        }
        
        private IEnumerator RespawnAfterModeChange(bool isRecording)
        {
            // Wait for destruction to complete
            yield return null;
            
            // Refresh terrain
            SM64Context.RefreshStaticTerrain();
            yield return new WaitForFixedUpdate();
            
            if (isRecording)
            {
                SpawnPlayerMario();
                Debug.Log("[HybridEnv] Mode changed to Recording: 1 Mario spawned");
            }
            else
            {
                SpawnPlayerMario();
                yield return new WaitForSeconds(0.1f); // Small delay between spawns
                SpawnAIMario();
                Debug.Log("[HybridEnv] Mode changed to Training: 2 Marios spawned");
            }
        }

        void OnGUI()
        {
            if (!showComparisonUI) return;
            
            // Simple comparison UI
            GUI.Box(new Rect(10, 10, 250, 120), "Hybrid Training Stats");
            
            GUI.Label(new Rect(20, 35, 230, 20), $"Player (Green): {playerCompletedCount} completes");
            GUI.Label(new Rect(20, 55, 230, 20), $"Player Best: {(playerBestTime < 999 ? playerBestTime.ToString("F2") + "s" : "--")}");
            
            GUI.Label(new Rect(20, 80, 230, 20), $"AI (Blue): {aiCompletedCount} completes");
            GUI.Label(new Rect(20, 100, 230, 20), $"AI Best: {(aiBestTime < 999 ? aiBestTime.ToString("F2") + "s" : "--")}");
            
            // Current mode
            string modeStr = trainingManager != null ? trainingManager.CurrentMode.ToString() : "Unknown";
            GUI.Label(new Rect(Screen.width - 150, 10, 140, 20), $"Mode: {modeStr}");
        }
    }
}

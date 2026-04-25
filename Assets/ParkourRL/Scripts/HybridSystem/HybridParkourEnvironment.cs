using UnityEngine;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Ambiente híbrido: gerencia dois Marios lado a lado.
    /// Um é player-controlled (para gravação/IL), outro é AI (para treino).
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
        [SerializeField] private Transform[] sharedSpawnPoints;  // Fallback se os específicos não estiverem definidos

        [Header("Hybrid Settings")]
        [Tooltip("Distância entre os dois Marios (eixo X)")]
        [SerializeField] private float marioSpacing = 15f;
        
        [Tooltip("Sincronizar resets (quando um morre, ambos reiniciam)")]
        [SerializeField] private bool syncResets = true;
        
        [Tooltip("Mostrar UI de comparação")]
        [SerializeField] private bool showComparisonUI = true;

        [Header("Recording Components")]
        [SerializeField] private HybridDataRecorder dataRecorder;
        [SerializeField] private HybridTrainingManager trainingManager;

        // Referências aos Marios
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
            EnsureAllMeshColliders();
            SM64Context.RefreshStaticTerrain();
            
            InitializeSpawnPoints();
            
            // Aguardar inicialização
            Invoke(nameof(SpawnBothMarios), 0.5f);
        }

        private void InitializeSpawnPoints()
        {
            // Usar spawn points específicos ou calcular a partir do shared
            if (playerSpawnPoint != null)
            {
                currentPlayerSpawn = playerSpawnPoint.position;
            }
            else if (sharedSpawnPoints != null && sharedSpawnPoints.Length > 0 && sharedSpawnPoints[0] != null)
            {
                currentPlayerSpawn = sharedSpawnPoints[0].position;
            }
            else
            {
                currentPlayerSpawn = transform.position + Vector3.left * (marioSpacing / 2f);
            }
            
            if (aiSpawnPoint != null)
            {
                currentAISpawn = aiSpawnPoint.position;
            }
            else if (sharedSpawnPoints != null && sharedSpawnPoints.Length > 1 && sharedSpawnPoints[1] != null)
            {
                currentAISpawn = sharedSpawnPoints[1].position;
            }
            else
            {
                currentAISpawn = transform.position + Vector3.right * (marioSpacing / 2f);
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

        private void SpawnBothMarios()
        {
            SpawnPlayerMario();
            SpawnAIMario();
            
            Debug.Log("[HybridEnv] Ambos os Marios spawnados!");
        }

        private void SpawnPlayerMario()
        {
            // Destruir anterior se existir
            if (playerMario != null)
            {
                Destroy(playerMario);
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
            
            // Câmera para o player
            GameObject camObj = new GameObject("PlayerCamera");
            Camera cam = camObj.AddComponent<Camera>();
            cam.rect = new Rect(0, 0, 0.5f, 1);  // Metade esquerda da tela
            playerController.SetCamera(cam);
            
            playerMario.SetActive(true);
            
            // Conectar ao recorder se estiver em modo gravação
            if (dataRecorder != null && trainingManager != null && 
                trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording)
            {
                playerController.SetDataRecorder(dataRecorder);
            }
        }

        private void SpawnAIMario()
        {
            // Destruir anterior se existir
            if (aiMario != null)
            {
                Destroy(aiMario);
            }
            
            // Criar Mario AI
            aiMario = new GameObject("Mario_AI_Agent");
            aiMario.SetActive(false);
            aiMario.transform.position = currentAISpawn + Vector3.up * 2f;
            
            // Adicionar agente híbrido
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
            decisionRequester.TakeActionsBetweenDecisions = true;
            
            // Câmera para o AI (view-only)
            GameObject camObj = new GameObject("AICamera");
            Camera cam = camObj.AddComponent<Camera>();
            cam.rect = new Rect(0.5f, 0, 0.5f, 1);  // Metade direita da tela
            
            aiMario.SetActive(true);
            
            // Sincronizar modo
            if (trainingManager != null)
            {
                aiAgent.SetMode(trainingManager.CurrentMode == HybridTrainingManager.HybridMode.Recording 
                    ? MarioHybridAgent.HybridMode.Recording 
                    : MarioHybridAgent.HybridMode.Training);
            }
        }

        public void ResetEnvironment()
        {
            // Respawn de ambos - APENAS resetar posições, NÃO chamar EndEpisode/OnEpisodeBegin
            if (playerMario != null && playerController != null)
            {
                playerController.ResetPlayer();
            }
            
            if (aiMario != null && aiAgent != null)
            {
                // Apenas resetar a posição do agente IA
                aiAgent.transform.position = GetCurrentSpawnPoint();
                // NÃO chamar EndEpisode() aqui - causa recursão infinita!
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
            if (aiAgent != null)
            {
                aiAgent.SetMode(mode == HybridTrainingManager.HybridMode.Recording 
                    ? MarioHybridAgent.HybridMode.Recording 
                    : MarioHybridAgent.HybridMode.Training);
            }
            
            if (playerController != null && mode == HybridTrainingManager.HybridMode.Recording)
            {
                playerController.SetDataRecorder(dataRecorder);
            }
            
            // Recriar Marios para aplicar mudanças
            ResetEnvironment();
        }

        void OnGUI()
        {
            if (!showComparisonUI) return;
            
            // UI simples de comparação
            GUI.Box(new Rect(10, 10, 250, 120), "Hybrid Training Stats");
            
            GUI.Label(new Rect(20, 35, 230, 20), $"Player (Green): {playerCompletedCount} completes");
            GUI.Label(new Rect(20, 55, 230, 20), $"Player Best: {(playerBestTime < 999 ? playerBestTime.ToString("F2") + "s" : "--")}");
            
            GUI.Label(new Rect(20, 80, 230, 20), $"AI (Blue): {aiCompletedCount} completes");
            GUI.Label(new Rect(20, 100, 230, 20), $"AI Best: {(aiBestTime < 999 ? aiBestTime.ToString("F2") + "s" : "--")}");
            
            // Modo atual
            string modeStr = trainingManager != null ? trainingManager.CurrentMode.ToString() : "Unknown";
            GUI.Label(new Rect(Screen.width - 150, 10, 140, 20), $"Mode: {modeStr}");
        }
    }
}

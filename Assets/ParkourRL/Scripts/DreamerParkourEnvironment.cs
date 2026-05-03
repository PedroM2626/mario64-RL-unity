using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Environment dedicated for Dreamer algorithm training.
    /// Optimized for world-model based RL with multi-agent support.
    /// </summary>
    public class DreamerParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Material")]
        [Tooltip("Optional base material.")]
        [SerializeField] private Material baseMarioMaterial;
        [Tooltip("Optional prefab to obtain fallback material.")]
        [SerializeField] private GameObject marioPrefab;

        [Header("Dreamer Agent Colors")]
        [Tooltip("Dreamer Mario primary color.")]
        [SerializeField] private Color dreamerColor = new Color(0.8f, 0.2f, 0.8f, 1f);
        [Tooltip("Dreamer secondary color for comparison training.")]
        [SerializeField] private Color dreamerSecondaryColor = new Color(0.2f, 0.8f, 0.8f, 1f);

        [Header("Level Configuration")]
        [SerializeField] private Transform[] spawnPoints;
        [SerializeField] private Transform goal;
        [SerializeField] private Transform[] checkpointPoints;

        [Header("Dreamer Training Settings")]
        [SerializeField] private int agentCount = 4;
        [Tooltip("Enable for competitive training between multiple Dreamer agents")]
        [SerializeField] private bool competitiveMode = false;
        [SerializeField] private string behaviorName = "MarioDreamer";
        
        [Header("Parallel Environments")]
        [Tooltip("Spacing between parallel environments in Z axis")]
        [SerializeField] private float environmentSpacing = 50f;
        [Tooltip("Enable parallel environments (each agent gets its own platform copy)")]
        [SerializeField] private bool useParallelEnvironments = true;
        
        [Header("Visual")]
        [SerializeField] private bool useVisualObservations = false;
        [SerializeField] private Vector2Int cameraResolution = new Vector2Int(64, 64);

        private Color[] teamColors;

        private List<MarioDreamerAgent> agents = new List<MarioDreamerAgent>();
        private int finishOrder = 0;
        
        // Parallel environment instances
        private class ParallelEnvInstance
        {
            public GameObject root;
            public Transform goalTransform;
            public Transform[] spawnPoints;
            public Vector3 offset;
        }
        private List<ParallelEnvInstance> parallelInstances = new List<ParallelEnvInstance>();
        
        // Scoreboard
        private Dictionary<string, int> scores = new Dictionary<string, int>();
        private Text scoreText;

        // Runtime materials - dinâmico para múltiplos agentes
        private List<Material> runtimeMaterials = new List<Material>();
        private Color[] predefinedColors = new Color[]
        {
            new Color(1.0f, 0.0f, 0.0f, 1.0f),      // Vermelho (Mario original)
            new Color(0.0f, 0.5f, 1.0f, 1.0f),      // Azul
            new Color(0.0f, 1.0f, 0.0f, 1.0f),      // Verde
            new Color(1.0f, 0.0f, 1.0f, 1.0f),      // Magenta
            new Color(1.0f, 1.0f, 0.0f, 1.0f),      // Amarelo
            new Color(0.0f, 1.0f, 1.0f, 1.0f),      // Ciano
            new Color(1.0f, 0.5f, 0.0f, 1.0f),      // Laranja
            new Color(0.5f, 0.0f, 1.0f, 1.0f),      // Roxo
        };

        // Camera for visual observations (if enabled)
        private Camera observationCamera;
        private RenderTexture observationRenderTexture;

        void Awake()
        {
            teamColors = new Color[] { dreamerColor, dreamerSecondaryColor };
            SetupScoreboard();
            
            if (useVisualObservations)
            {
                SetupVisualObservationCamera();
            }
        }

        void Start()
        {
            Debug.Log("[DreamerEnv] Start() called - Initializing environment...");
            
            EnsureAllMeshColliders();
            
            // Criar ambientes paralelos (cópias da plataforma para cada agente)
            if (useParallelEnvironments && agentCount > 1 && transform.parent == null)
            {
                SpawnParallelEnvironments();
            }
            
            // Forçar criação do SM64Context se não existir
            var existingContext = FindObjectOfType<SM64Context>();
            if (existingContext == null)
            {
                Debug.Log("[DreamerEnv] Creating SM64Context...");
                GameObject contextObj = new GameObject("SM64Context");
                contextObj.AddComponent<SM64Context>();
            }
            
            SM64Context.RefreshStaticTerrain();
            Debug.Log("[DreamerEnv] Static terrain refreshed");
            
            // Aguardar SM64Context estar pronto
            StartCoroutine(SpawnAllAgentsDelayed());
        }

        private IEnumerator SpawnAllAgentsDelayed()
        {
            Debug.Log("[DreamerEnv] Waiting for SM64Context to be ready...");
            
            // Aguardar SM64Context existir e estar inicializado
            int attempts = 0;
            SM64Context context = null;
            
            while (context == null && attempts < 50)
            {
                context = FindObjectOfType<SM64Context>();
                if (context == null)
                {
                    yield return new WaitForSeconds(0.1f);
                    attempts++;
                }
            }
            
            if (context == null)
            {
                Debug.LogError("[DreamerEnv] CRITICAL: SM64Context not found after 50 attempts!");
            }
            else
            {
                Debug.Log($"[DreamerEnv] SM64Context found after {attempts} attempts");
            }
            
            // Aguardar mais tempo para garantir que o SM64Context processou o terreno
            yield return new WaitForSeconds(1.0f);
            
            Debug.Log("[DreamerEnv] Spawning agents now...");
            SpawnAllAgents();
        }

        void OnDestroy()
        {
            if (observationRenderTexture != null)
            {
                observationRenderTexture.Release();
                Destroy(observationRenderTexture);
            }
        }

        private void SetupVisualObservationCamera()
        {
            GameObject camObj = new GameObject("DreamerObservationCamera");
            observationCamera = camObj.AddComponent<Camera>();
            observationCamera.enabled = false;
            
            observationRenderTexture = new RenderTexture(cameraResolution.x, cameraResolution.y, 24, RenderTextureFormat.RGB565);
            observationRenderTexture.Create();
            observationCamera.targetTexture = observationRenderTexture;
            
            camObj.transform.SetParent(transform);
        }

        public RenderTexture GetObservationRenderTexture()
        {
            return observationRenderTexture;
        }

        public void UpdateObservationCamera(Vector3 position, Vector3 lookDirection)
        {
            if (observationCamera != null)
            {
                observationCamera.transform.position = position + Vector3.up * 1.5f;
                observationCamera.transform.rotation = Quaternion.LookRotation(lookDirection);
            }
        }

        private void SetupScoreboard()
        {
            scores["Dreamer"] = 0;
            if (competitiveMode)
            {
                scores["Dreamer2"] = 0;
            }

            GameObject canvasObj = new GameObject("DreamerScoreCanvas");
            Canvas canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasObj.AddComponent<CanvasScaler>();
            canvasObj.AddComponent<GraphicRaycaster>();

            GameObject textObj = new GameObject("ScoreText");
            textObj.transform.SetParent(canvasObj.transform);
            scoreText = textObj.AddComponent<Text>();
            try
            {
                scoreText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            }
            catch
            {
                scoreText.font = Font.CreateDynamicFontFromOSFont("Arial", 16);
            }
            scoreText.fontSize = 24;
            scoreText.color = Color.white;
            scoreText.alignment = TextAnchor.UpperLeft;
            
            RectTransform rt = scoreText.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = new Vector2(0, 1);
            rt.pivot = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(20, -20);
            rt.sizeDelta = new Vector2(350, 150);

            UpdateScoreboardUI();
        }

        private void UpdateScoreboardUI()
        {
            if (scoreText != null)
            {
                string dreamerHex = ColorUtility.ToHtmlStringRGB(dreamerColor);
                
                string text = $"<b>Dreamer Training</b>\n" +
                              $"<color=#{dreamerHex}>Wins: {scores["Dreamer"]}</color>\n";
                
                if (competitiveMode)
                {
                    string dreamer2Hex = ColorUtility.ToHtmlStringRGB(dreamerSecondaryColor);
                    text += $"<color=#{dreamer2Hex}>Wins (Agent 2): {scores["Dreamer2"]}</color>\n";
                }
                
                text += $"Total: {scores["Dreamer"] + (competitiveMode ? scores["Dreamer2"] : 0)}";
                
                scoreText.text = text;
            }
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            Debug.Log($"[DreamerEnv] Found {terrains.Length} SM64 terrains");
            
            int collidersAdded = 0;
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
                        collidersAdded++;
                    }
                }
            }
            Debug.Log($"[DreamerEnv] {collidersAdded} MeshColliders added");
        }

        private Material ResolveBaseMaterial()
        {
            Material matBase = baseMarioMaterial;
            if (matBase == null && marioPrefab != null)
            {
                SM64Mario prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    var matField = typeof(SM64Mario).GetField("material",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matBase = matField.GetValue(prefabMario) as Material;
                }
            }
            return matBase;
        }

        private Material CreateRuntimeMaterial(Material source, Color color, string label)
        {
            if (source == null) return null;

            Material mat = new Material(source);
            mat.name = $"DreamerMario_{label}";

            Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            tex.name = $"DreamerTex_{label}";
            tex.SetPixel(0, 0, color);
            tex.SetPixel(0, 1, color);
            tex.SetPixel(1, 0, color);
            tex.SetPixel(1, 1, color);
            tex.Apply();

            mat.mainTexture = tex;
            mat.SetTexture("_MainTex", tex);
            mat.color = Color.white;
            return mat;
        }

        private void SpawnParallelEnvironments()
        {
            Debug.Log($"[DreamerEnv] Spawning {agentCount} parallel environments...");
            
            // Coletar todas as plataformas da cena original (que não são filhas de outras)
            List<GameObject> scenePlatforms = new List<GameObject>();
            foreach (var terrain in FindObjectsOfType<SM64StaticTerrain>())
            {
                // Só plataformas da raiz (não filhas de outros envs paralelos)
                if (terrain.transform.parent == null)
                    scenePlatforms.Add(terrain.gameObject);
            }
            
            Debug.Log($"[DreamerEnv] Found {scenePlatforms.Count} platforms to clone");
            
            // Criar um ambiente paralelo para cada agente
            for (int i = 1; i < agentCount; i++)
            {
                Vector3 offset = new Vector3(0, 0, environmentSpacing * i);
                
                // Criar root do ambiente paralelo
                GameObject envRoot = new GameObject($"DreamerParallelEnv_{i}");
                envRoot.transform.position = offset;
                
                // Clonar cada plataforma
                foreach (var platform in scenePlatforms)
                {
                    GameObject clone = Instantiate(platform, 
                        platform.transform.position + offset, 
                        platform.transform.rotation, 
                        envRoot.transform);
                    clone.name = platform.name + $"_Env{i}";
                    clone.transform.localScale = platform.transform.localScale;
                    
                    // Garantir MeshCollider no clone
                    if (clone.GetComponent<MeshCollider>() == null)
                    {
                        MeshFilter mf = clone.GetComponent<MeshFilter>();
                        if (mf != null && mf.sharedMesh != null)
                        {
                            MeshCollider mc = clone.AddComponent<MeshCollider>();
                            mc.sharedMesh = mf.sharedMesh;
                            mc.convex = false;
                        }
                    }
                }
                
                // Clonar Goal
                GameObject goalClone = null;
                if (goal != null)
                {
                    goalClone = Instantiate(goal.gameObject, 
                        goal.position + offset, 
                        goal.rotation, 
                        envRoot.transform);
                    goalClone.name = $"Goal_Env{i}";
                    goalClone.tag = "Goal";
                }
                
                // Clonar spawnPoints
                List<Transform> clonedSpawnPoints = new List<Transform>();
                if (spawnPoints != null)
                {
                    foreach (Transform sourceSpawn in spawnPoints)
                    {
                        if (sourceSpawn == null) continue;
                        
                        GameObject spawnClone = new GameObject($"{sourceSpawn.name}_Env{i}");
                        spawnClone.transform.position = sourceSpawn.position + offset;
                        spawnClone.transform.rotation = sourceSpawn.rotation;
                        spawnClone.transform.parent = envRoot.transform;
                        clonedSpawnPoints.Add(spawnClone.transform);
                    }
                }
                
                // Fallback spawn se não tiver spawnPoints
                if (clonedSpawnPoints.Count == 0)
                {
                    GameObject fallbackSpawn = new GameObject($"SpawnPoint_Env{i}");
                    fallbackSpawn.transform.position = transform.position + offset;
                    fallbackSpawn.transform.parent = envRoot.transform;
                    clonedSpawnPoints.Add(fallbackSpawn.transform);
                }
                
                // Criar DreamerParkourEnvironment para este ambiente paralelo
                GameObject envControllerObj = new GameObject($"DreamerEnvController_Env{i}");
                envControllerObj.transform.position = offset;
                envControllerObj.transform.parent = envRoot.transform;
                
                DreamerParkourEnvironment envScript = envControllerObj.AddComponent<DreamerParkourEnvironment>();
                envScript.agentCount = 1; // Cada ambiente paralelo tem 1 agente
                envScript.useParallelEnvironments = false; // Evitar recursão
                envScript.goal = goalClone != null ? goalClone.transform : null;
                envScript.spawnPoints = clonedSpawnPoints.ToArray();
                envScript.checkpointPoints = this.checkpointPoints; // Compartilhar checkpoints ou clonar se necessário
                envScript.behaviorName = this.behaviorName;
                envScript.useVisualObservations = this.useVisualObservations;
                envScript.cameraResolution = this.cameraResolution;
                envScript.marioPrefab = this.marioPrefab;
                envScript.baseMarioMaterial = this.baseMarioMaterial;
                
                // Registrar instância paralela
                parallelInstances.Add(new ParallelEnvInstance
                {
                    root = envRoot,
                    goalTransform = goalClone != null ? goalClone.transform : null,
                    spawnPoints = clonedSpawnPoints.ToArray(),
                    offset = offset
                });
                
                Debug.Log($"[DreamerEnv] Created parallel environment {i} at offset {offset}");
            }
            
            Debug.Log($"[DreamerEnv] Created {parallelInstances.Count} parallel environments");
        }

        private void SpawnAllAgents()
        {
            Material matBase = ResolveBaseMaterial();
            
            if (competitiveMode && agentCount < 2)
            {
                agentCount = 2;
            }

            // Se for ambiente paralelo (não é root), spawnar apenas 1 agente
            // O root spawna agentCount agentes, cada um em seu ambiente
            int agentsToSpawn = agentCount;
            bool isParallelEnv = transform.parent != null;
            if (isParallelEnv)
            {
                agentsToSpawn = 1;
                Debug.Log($"[DreamerEnv] This is a parallel environment, spawning only 1 agent");
            }
            else if (useParallelEnvironments && agentCount > 1)
            {
                // Ambiente root com ambientes paralelos: spawnar apenas 1 agente
                // Os outros agentCount-1 serão spawnados nos ambientes paralelos
                agentsToSpawn = 1;
                Debug.Log($"[DreamerEnv] Root environment with parallel envs, spawning only 1 agent here");
            }

            // Create runtime materials para cada agente com cor única
            runtimeMaterials.Clear();
            for (int i = 0; i < agentsToSpawn; i++)
            {
                Color agentColor = predefinedColors[i % predefinedColors.Length];
                Material runtimeMat = CreateRuntimeMaterial(matBase, agentColor, $"Dreamer{i}");
                runtimeMaterials.Add(runtimeMat);
                Debug.Log($"[DreamerEnv] Created material for Agent {i} with color {agentColor}");
            }

            Debug.Log($"[DreamerEnv] Starting spawn of {agentsToSpawn} Dreamer agents.");

            // Calcular o offset do agent ID baseado no índice do ambiente paralelo
            int envIndex = 0;
            if (transform.parent != null)
            {
                // Extrair o índice do nome do parent (DreamerParallelEnv_X)
                string parentName = transform.parent.name;
                if (parentName.StartsWith("DreamerParallelEnv_"))
                {
                    int.TryParse(parentName.Substring("DreamerParallelEnv_".Length), out envIndex);
                }
            }

            for (int i = 0; i < agentsToSpawn; i++)
            {
                // ID global do agente (considerando ambiente paralelo)
                int globalAgentId = envIndex * 1 + i; // Cada ambiente tem 1 agente
                
                Vector3 spawnPos = GetSpawnPositionForIndex(i);
                Material runtimeMat = runtimeMaterials[i];
                Color teamColor = predefinedColors[globalAgentId % predefinedColors.Length];
                string agentBehaviorName = competitiveMode ? $"{behaviorName}_{globalAgentId}" : behaviorName;
                SpawnAgent(globalAgentId, spawnPos, teamColor, runtimeMat, agentBehaviorName);
            }

            Debug.Log($"[DreamerEnv] Total agents created: {agents.Count}");
            
            if (competitiveMode)
            {
                foreach (var agent in agents)
                {
                    agent.SetRivals(agents);
                }
            }

            Debug.Log("[DreamerEnv] Dreamer agents spawned successfully!");
        }

        private Vector3 GetSpawnPositionForIndex(int index)
        {
            // Distância segura entre agentes (1.0m) - não muito perto, não muito longe
            float spacing = 1.0f;
            
            // Primeiro tentar usar spawnPoints definidos no editor
            if (spawnPoints != null && spawnPoints.Length > 0)
            {
                // Se temos spawnPoints suficientes, use diretamente
                if (index < spawnPoints.Length && spawnPoints[index] != null)
                {
                    Debug.Log($"[DreamerEnv] Agent {index} spawning at spawnPoints[{index}]: {spawnPoints[index].position}");
                    return spawnPoints[index].position;
                }
                
                // Caso contrário, distribuir a partir do primeiro spawnPoint
                if (spawnPoints[0] != null)
                {
                    Vector3 basePos = spawnPoints[0].position;
                    // Distribuir em linha no eixo X, mantendo Y e Z do spawnPoint
                    Vector3 spawnPos = basePos + new Vector3(index * spacing, 0, 0);
                    Debug.Log($"[DreamerEnv] Agent {index} spawning at calculated pos: {spawnPos} (from base: {basePos})");
                    return spawnPos;
                }
            }

            // Fallback final: usar posição do próprio transform do environment
            // Isso garante que spawnem onde o ambiente está posicionado
            Vector3 envPos = transform.position;
            Vector3 fallbackPos = envPos + new Vector3(index * spacing, 2f, 0f);
            Debug.LogWarning($"[DreamerEnv] Agent {index} using fallback spawn: {fallbackPos} (env at {envPos})");
            return fallbackPos;
        }

        private void SpawnAgent(int index, Vector3 spawnPos, Color teamColor, Material runtimeMat, string agentBehaviorName)
        {
            Debug.Log($"[DreamerEnv] Spawning Agent {index} at {spawnPos} with behavior '{agentBehaviorName}'");
            
            GameObject agentObj = new GameObject($"Mario_{agentBehaviorName}_{index}");
            agentObj.SetActive(false);
            agentObj.transform.position = spawnPos;

            // ORDEM CRÍTICA: DecisionRequester tem [RequireComponent(typeof(Agent))]
            // Se adicionarmos DecisionRequester antes do MarioDreamerAgent, o Unity
            // cria um Agent fantasma que rouba o ID 0 e não processa ações!
            
            // 1. MarioDreamerAgent PRIMEIRO (herda de Agent)
            MarioDreamerAgent agent = agentObj.AddComponent<MarioDreamerAgent>();
            
            // 2. Behavior Parameters - verificar se já existe
            var bp = agentObj.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp == null)
            {
                bp = agentObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            }
            bp.BehaviorName = agentBehaviorName;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = useVisualObservations ? 0 : 42;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            // IMPORTANTE: 2 continuous (joystick x, y), 3 discrete (jump, kick, stomp)
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });
            
            Debug.Log($"[DreamerEnv] Agent {index} - Behavior Name: {agentBehaviorName}");
            Debug.Log($"[DreamerEnv] Agent {index} - ActionSpec: Continuous={bp.BrainParameters.ActionSpec.NumContinuousActions}, DiscreteBranches=[{string.Join(",", bp.BrainParameters.ActionSpec.BranchSizes)}]");

            // 3. DecisionRequester - DEPOIS do Agent (para não criar fantasma)
            var dr = agentObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // 4. Input provider
            agentObj.AddComponent<MarioInputProvider>();
            
            // 5. SM64Mario
            SM64Mario sm64Mario = agentObj.AddComponent<SM64Mario>();
            
            // Configurar SM64Mario para spawn correto
            var spawnPosField = typeof(SM64Mario).GetField("spawnPosition",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (spawnPosField != null)
                spawnPosField.SetValue(sm64Mario, spawnPos);
            
            // 6. Apply custom runtime material
            if (runtimeMat != null)
            {
                sm64Mario.useCustomTexture = true;
                sm64Mario.tintColor = Color.white;

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, runtimeMat);
            }

            // Configure agent properties
            agent.teamId = index;
            agent.teamColor = teamColor;
            agent.SetGoal(goal);
            agent.SetDreamerEnv(this);
            agent.useVisualObservations = useVisualObservations;
            agent.cameraResolution = cameraResolution;
            
            agents.Add(agent);
            
            // Ativar objeto APÓS todas as configurações
            agentObj.SetActive(true);
            
            Debug.Log($"[DreamerEnv] Agent {index} spawned successfully at {agentObj.transform.position}");
        }

        public int RegisterFinish(MarioDreamerAgent agent)
        {
            finishOrder++;
            int position = finishOrder;
            
            if (position == 1)
            {
                string agentKey = agent.teamId == 0 ? "Dreamer" : "Dreamer2";
                if (scores.ContainsKey(agentKey))
                {
                    scores[agentKey]++;
                }
                
                UpdateScoreboardUI();
            }

            if (competitiveMode)
            {
                foreach (var a in agents)
                {
                    if (a != agent && !a.HasFinished)
                    {
                        a.OnRivalFinished(agent);
                    }
                }
            }

            bool allDone = true;
            foreach (var a in agents)
            {
                if (!a.HasFinished && a.gameObject.activeInHierarchy)
                {
                    allDone = false;
                    break;
                }
            }

            if (allDone)
            {
                finishOrder = 0;
                Debug.Log("[DreamerEnv] All agents finished! Resetting order.");
            }

            return position;
        }

        public void RespawnAgent(MarioDreamerAgent agent)
        {
            int idx = agents.IndexOf(agent);
            Vector3 spawnPos;

            if (spawnPoints != null && idx < spawnPoints.Length && idx >= 0 && spawnPoints[idx] != null)
            {
                spawnPos = spawnPoints[idx].position + Vector3.up * 1f;
            }
            else if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                spawnPos = spawnPoints[0].position + Vector3.up * 1f + Vector3.right * (idx * 2f);
            }
            else
            {
                spawnPos = Vector3.up * 2f + Vector3.right * (idx * 2f);
            }

            SM64Mario sm64Mario = agent.GetComponent<SM64Mario>();
            if (sm64Mario != null)
            {
                sm64Mario.Teleport(spawnPos + Vector3.up * 1f);
            }
            else
            {
                agent.transform.position = spawnPos + Vector3.up * 1f;
            }
        }

        public Vector3 GetCurrentSpawnPoint(MarioDreamerAgent agent)
        {
            int idx = agents.IndexOf(agent);
            return GetSpawnPositionForIndex(idx >= 0 ? idx : 0);
        }

        public Transform[] GetCheckpointPoints()
        {
            return checkpointPoints;
        }

        void OnDrawGizmos()
        {
            if (spawnPoints != null)
            {
                for (int i = 0; i < spawnPoints.Length; i++)
                {
                    if (spawnPoints[i] != null)
                    {
                        Gizmos.color = (i < teamColors.Length) ? teamColors[i] : Color.white;
                        Gizmos.DrawWireSphere(spawnPoints[i].position, 0.8f);
                    }
                }
            }

            if (goal != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireCube(goal.position, Vector3.one * 3f);
            }

            if (checkpointPoints != null)
            {
                Gizmos.color = Color.cyan;
                for (int i = 0; i < checkpointPoints.Length; i++)
                {
                    if (checkpointPoints[i] != null)
                    {
                        Gizmos.DrawWireSphere(checkpointPoints[i].position, 1.0f);
                    }
                }
            }
        }
    }
}

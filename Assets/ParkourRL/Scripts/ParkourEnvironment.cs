using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using LibSM64;
using Unity.Barracuda;

namespace ParkourRL
{
    public class ParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material marioMaterial;
        [Tooltip("Usa um modelo base para warm-start do Mario no BehaviorParameters.")]
        [SerializeField] private bool useWarmStartModel = true;
        [SerializeField] private NNModel warmStartModel;
        [Tooltip("Caminho do asset para auto-carregar o modelo no editor quando o campo acima estiver vazio.")]
        [SerializeField] private string warmStartModelAssetPath = "Assets/ParkourRL/Models/mario_parkour_baseV1.onnx";

        [Header("Level Elements")]
        [SerializeField] private Transform goal;
        [Tooltip("Defina manualmente os locais de spawn no Inspector.")]
        [SerializeField] private List<Transform> spawnPoints = new List<Transform>();
        [Tooltip("Indice do spawnpoint utilizado para treino e respawn.")]
        [SerializeField] private int selectedSpawnPointIndex = 0;

        [Header("Curriculum")]
        [Tooltip("Seleciona automaticamente o spawnpoint com base na licao atual do curriculum.")]
        [SerializeField] private bool useCurriculumLessonForSpawn = true;
        [Tooltip("Nome do Environment Parameter que guarda a licao atual.")]
        [SerializeField] private string curriculumLessonParameter = "spawn_lesson";
        [Tooltip("Se ligado, licao 0 usa o primeiro item da lista; se desligado, usa o ultimo.")]
        [SerializeField] private bool lessonZeroUsesFirstSpawnPoint = true;
        [Tooltip("Forca manualmente um spawnpoint fixo, ignorando o curriculum.")]
        [SerializeField] private bool useManualSpawnPointOverride = false;
        [Tooltip("Spawnpoint manual usado quando o override esta ativo.")]
        [SerializeField] private int manualSpawnPointIndex = 0;

        [Header("Randomization")]
        [SerializeField] private bool randomizePlatforms = false;
        [Tooltip("Usa a licao do curriculum para escalar a randomizacao.")]
        [SerializeField] private bool useCurriculumForRandomization = true;
        [Tooltip("Range maximo usado quando a randomizacao estiver no valor final.")]
        [SerializeField] private float platformRandomizationRange = 0.5f;
        [Tooltip("Range manual usado quando o curriculum esta desligado.")]
        [SerializeField] private float manualPlatformRandomizationRange = 0.15f;
        [Tooltip("A randomizacao so entra quando a licao do curriculum atingir este valor. Aumentado para 10 para manter licoes 0-4 fixas.")]
        [SerializeField] private int randomizationStartsAtLesson = 10;
        [Tooltip("Licao na qual a randomizacao atinge o valor maximo configurado.")]
        [SerializeField] private int randomizationMaxesAtLesson = 4;

        [Header("Advanced Phases (5-8)")]
        [Tooltip("Variação máxima de spawn/goal em metros para fases 8+.")]
        [SerializeField] private float fullMapSpawnVariationRange = 1.0f;

        [Header("Win-Rate Cycle (Infinite)")]
        [Tooltip("Quando ativo, ignora avanço por reward e alterna entre mapa fixo e mapa random usando win-rate.")]
        [SerializeField] private bool useWinRateCycle = true;
        [Tooltip("Quantidade de episodios usados para calcular win-rate.")]
        [SerializeField] private int winRateWindowSize = 10;
        [Tooltip("Taxa minima de vitoria para alternar entre mapa fixo e random.")]
        [Range(0.0f, 1.0f)]
        [SerializeField] private float winRateThreshold = 0.7f;

        [Header("Multi-Agent Parallel Training")]
        [SerializeField] private int parallelEnvironments = 4;
        [SerializeField] private float environmentSpacing = 30f;

        [Header("Per-Env Diversification")]
        [Tooltip("Aplica um offset fixo por env (spawn e goal) para evitar decoracao entre envs paralelos.")]
        [SerializeField] private bool diversifyEachEnvironment = true;
        [Tooltip("Range em metros do offset por env no plano XZ.")]
        [SerializeField] private float perEnvironmentVariationRange = 1.0f;

        private Vector3 currentSpawnPoint;
        private GameObject currentMario;
        private Vector3 originalGoalPosition;
        private bool warnedMissingSpawnPoints = false;
        private int lastLoggedCurriculumLesson = int.MinValue;
        private int lastLoggedSpawnIndex = int.MinValue;
        private Vector3 envSpawnOffset = Vector3.zero;
        private Vector3 envGoalOffset = Vector3.zero;
        private bool envOffsetsInitialized = false;
        private const float SPAWN_RANDOMIZATION_BOUNDS_FACTOR = 0.45f;
        private const float MIN_RANDOMIZATION_RANGE = 0.05f;
        private Queue<bool> recentEpisodeResults = new Queue<bool>();
        private bool randomModeEnabled = false;
        
        private List<ParallelEnvInstance> parallelInstances = new List<ParallelEnvInstance>();

        private class ParallelEnvInstance
        {
            public GameObject root;
            public Transform goalTransform;
            public ParkourEnvironment envScript;
        }

        private bool justSpawned = false;

        void Awake()
        {
            TryResolveWarmStartModel();
            InitializeOriginalPositions();
        }

        void OnValidate()
        {
            TryResolveWarmStartModel();
        }

        private void TryResolveWarmStartModel()
        {
            if (!useWarmStartModel || warmStartModel != null)
                return;

            #if UNITY_EDITOR
            if (!string.IsNullOrWhiteSpace(warmStartModelAssetPath))
            {
                warmStartModel = UnityEditor.AssetDatabase.LoadAssetAtPath<NNModel>(warmStartModelAssetPath);
            }
            #endif
        }

        public void InitializeOriginalPositions()
        {
            currentSpawnPoint = GetSelectedSpawnPosition();
            if (goal != null)
                originalGoalPosition = goal.position;

            EnsurePerEnvironmentOffsets();
        }

        private Vector3 GetSelectedSpawnPosition()
        {
            Vector3 baseSpawn = Vector3.zero;
            
            if (useManualSpawnPointOverride && spawnPoints != null && spawnPoints.Count > 0)
            {
                manualSpawnPointIndex = Mathf.Clamp(manualSpawnPointIndex, 0, spawnPoints.Count - 1);
                Transform manualSpawn = spawnPoints[manualSpawnPointIndex];
                if (manualSpawn != null)
                    baseSpawn = manualSpawn.position;
            }
            else if (spawnPoints != null && spawnPoints.Count > 0)
            {
                selectedSpawnPointIndex = Mathf.Clamp(selectedSpawnPointIndex, 0, spawnPoints.Count - 1);
                Transform selectedSpawn = spawnPoints[selectedSpawnPointIndex];
                if (selectedSpawn != null)
                    baseSpawn = selectedSpawn.position;
            }
            else
            {
                if (!warnedMissingSpawnPoints)
                {
                    Debug.LogWarning("[ParkourEnv] Nenhum spawnpoint manual configurado. Usando posicao do ParkourEnvironment.");
                    warnedMissingSpawnPoints = true;
                }
                baseSpawn = transform.position;
            }

            return baseSpawn;
        }

        private void EnsurePerEnvironmentOffsets()
        {
            if (envOffsetsInitialized)
                return;

            envOffsetsInitialized = true;

            if (!diversifyEachEnvironment || perEnvironmentVariationRange <= 0f)
            {
                envSpawnOffset = Vector3.zero;
                envGoalOffset = Vector3.zero;
                return;
            }

            float spawnX = Random.Range(-perEnvironmentVariationRange, perEnvironmentVariationRange);
            float spawnZ = Random.Range(-perEnvironmentVariationRange, perEnvironmentVariationRange);
            float goalX = Random.Range(-perEnvironmentVariationRange, perEnvironmentVariationRange);
            float goalZ = Random.Range(-perEnvironmentVariationRange, perEnvironmentVariationRange);

            envSpawnOffset = new Vector3(spawnX, 0f, spawnZ);
            envGoalOffset = new Vector3(goalX, 0f, goalZ);
        }

        private void ApplyPerEnvironmentOffsets()
        {
            EnsurePerEnvironmentOffsets();
            currentSpawnPoint += envSpawnOffset;
            if (goal != null)
                goal.position += envGoalOffset;
        }

        void Start()
        {
            // PASSO 1: Garantir MeshColliders em TODAS as plataformas SM64StaticTerrain
            // DEVE rodar ANTES de qualquer RefreshStaticTerrain!
            EnsureAllMeshColliders();

            // PASSO 2: Instanciar ambientes paralelos (se for o ambiente principal)
            if (parallelEnvironments > 1 && transform.parent == null)
            {
                SpawnParallelEnvironments();
            }

            StartCoroutine(RefreshTerrainAndSpawnMario());
        }

        private IEnumerator RefreshTerrainAndSpawnMario()
        {
            // Espera o ciclo de fisica para garantir que os colliders recem-criados estejam prontos.
            yield return new WaitForFixedUpdate();

            // PASSO 3: Agora sim, recarregar terreno no SM64 com TODAS as plataformas
            SM64Context.RefreshStaticTerrain();
            LogTerrainInfo();

            // PASSO 4: Aplicar selecao por curriculum antes do primeiro spawn
            UpdateSpawnSelectionFromCurriculum();
            currentSpawnPoint = GetSelectedSpawnPosition();
            ApplyPerEnvironmentOffsets();

            // PASSO 5: Spawnar Mario
            SpawnMario();
        }

        /// <summary>
        /// Garante MeshCollider em todas as plataformas SM64StaticTerrain.
        /// Nesta versao, os cubos sao GIGANTES e afundados (padrao pipescene),
        /// as paredes ficam distantes e nao bloqueiam o Mario.
        /// </summary>
        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            int fixed_count = 0;
            
            foreach (var terrain in terrains)
            {
                MeshCollider mc = terrain.GetComponent<MeshCollider>();
                if (mc == null)
                {
                    MeshFilter meshFilter = terrain.GetComponent<MeshFilter>();
                    if (meshFilter != null && meshFilter.sharedMesh != null)
                    {
                        mc = terrain.gameObject.AddComponent<MeshCollider>();
                        mc.sharedMesh = meshFilter.sharedMesh;
                        mc.convex = false;
                        fixed_count++;
                    }
                }
                
                Debug.Log($"[ParkourEnv] Terreno '{terrain.gameObject.name}' pos={terrain.transform.position} scale={terrain.transform.lossyScale} MC={(mc != null)}");
            }
            
            Debug.Log($"[ParkourEnv] {terrains.Length} plataformas verificadas, {fixed_count} MeshColliders adicionados");
        }

        /// <summary>
        /// Log diagnostico para verificar quantas superficies o SM64 carregou
        /// </summary>
        private void LogTerrainInfo()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            Debug.Log($"[ParkourEnv] === DIAGNOSTICO DE TERRENO SM64 ===");
            Debug.Log($"[ParkourEnv] Total de plataformas SM64StaticTerrain: {terrains.Length}");
            
            foreach (var t in terrains)
            {
                MeshCollider mc = t.GetComponent<MeshCollider>();
                bool hasMC = mc != null;
                bool hasMesh = hasMC && mc.sharedMesh != null;
                int triCount = hasMesh ? mc.sharedMesh.triangles.Length / 3 : 0;
                
                Debug.Log($"  [{t.gameObject.name}] Pos={t.transform.position}, " +
                          $"Scale={t.transform.lossyScale}, " +
                          $"MeshCollider={hasMC}, Mesh={hasMesh}, Tris={triCount}");
            }
            
            // Usar metodo publico do SM64Context para contar superficies
            int surfaceCount = SM64Context.GetStaticSurfaceCount();
            Debug.Log($"[ParkourEnv] Total de superficies SM64 carregadas: {surfaceCount}");
            Debug.Log($"[ParkourEnv] === FIM DIAGNOSTICO ===");
        }

        private void SpawnParallelEnvironments()
        {
            Debug.Log($"[ParkourEnv] Instanciando {parallelEnvironments - 1} ambientes paralelos...");
            
            List<GameObject> scenePlatforms = new List<GameObject>();
            foreach (var terrain in FindObjectsOfType<SM64StaticTerrain>())
            {
                if (terrain.transform.parent == null)
                    scenePlatforms.Add(terrain.gameObject);
            }
            
            for (int i = 1; i < parallelEnvironments; i++)
            {
                Vector3 offset = new Vector3(0, 0, environmentSpacing * i);
                
                GameObject envRoot = new GameObject($"ParallelEnv_{i}");
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
                
                // Clonar os spawnpoints manuais
                List<Transform> clonedSpawnPoints = new List<Transform>();
                if (spawnPoints != null)
                {
                    foreach (Transform sourceSpawn in spawnPoints)
                    {
                        if (sourceSpawn == null)
                            continue;

                        GameObject spawnClone = new GameObject($"{sourceSpawn.name}_Env{i}");
                        spawnClone.transform.position = sourceSpawn.position + offset;
                        spawnClone.transform.rotation = sourceSpawn.rotation;
                        spawnClone.transform.parent = envRoot.transform;
                        clonedSpawnPoints.Add(spawnClone.transform);
                    }
                }

                if (clonedSpawnPoints.Count == 0)
                {
                    GameObject fallbackSpawn = new GameObject($"SpawnPoint_Env{i}");
                    fallbackSpawn.transform.position = transform.position + offset;
                    fallbackSpawn.transform.parent = envRoot.transform;
                    clonedSpawnPoints.Add(fallbackSpawn.transform);
                }
                
                // Criar ParkourEnvironment para esta copia
                GameObject envControllerObj = new GameObject($"ParkourController_Env{i}");
                envControllerObj.transform.position = offset;
                envControllerObj.transform.parent = envRoot.transform;
                
                ParkourEnvironment envScript = envControllerObj.AddComponent<ParkourEnvironment>();
                envScript.parallelEnvironments = 0; // Impede recursao
                envScript.goal = goalClone != null ? goalClone.transform : null;
                envScript.marioPrefab = this.marioPrefab;
                envScript.marioMaterial = this.marioMaterial;
                envScript.useWarmStartModel = this.useWarmStartModel;
                envScript.warmStartModel = this.warmStartModel;
                envScript.warmStartModelAssetPath = this.warmStartModelAssetPath;
                envScript.randomizePlatforms = this.randomizePlatforms;
                envScript.useCurriculumForRandomization = this.useCurriculumForRandomization;
                envScript.platformRandomizationRange = this.platformRandomizationRange;
                envScript.manualPlatformRandomizationRange = this.manualPlatformRandomizationRange;
                envScript.randomizationStartsAtLesson = this.randomizationStartsAtLesson;
                envScript.randomizationMaxesAtLesson = this.randomizationMaxesAtLesson;
                envScript.spawnPoints = clonedSpawnPoints;
                envScript.selectedSpawnPointIndex = Mathf.Clamp(this.selectedSpawnPointIndex, 0, clonedSpawnPoints.Count - 1);
                envScript.useCurriculumLessonForSpawn = this.useCurriculumLessonForSpawn;
                envScript.curriculumLessonParameter = this.curriculumLessonParameter;
                envScript.lessonZeroUsesFirstSpawnPoint = this.lessonZeroUsesFirstSpawnPoint;
                envScript.useManualSpawnPointOverride = this.useManualSpawnPointOverride;
                envScript.manualSpawnPointIndex = Mathf.Clamp(this.manualSpawnPointIndex, 0, clonedSpawnPoints.Count - 1);
                envScript.diversifyEachEnvironment = this.diversifyEachEnvironment;
                envScript.perEnvironmentVariationRange = this.perEnvironmentVariationRange;
                
                // Forca re-inicializacao das posicoes originais APOS os valores (goal, spawn) terem sido copiados!
                envScript.InitializeOriginalPositions();
                
                parallelInstances.Add(new ParallelEnvInstance
                {
                    root = envRoot,
                    goalTransform = goalClone != null ? goalClone.transform : null,
                    envScript = envScript
                });
            }
            
            Debug.Log($"[ParkourEnv] {parallelEnvironments - 1} ambientes paralelos criados.");
        }

        public void ResetEnvironment()
        {
            EnsurePerEnvironmentOffsets();
            UpdateSpawnSelectionFromCurriculum();
            currentSpawnPoint = GetSelectedSpawnPosition();
            if (goal != null)
                goal.position = originalGoalPosition;

            float randomizationRange = GetCurriculumRandomizationRange();
            if (randomizationRange > 0f)
            {
                RandomizeSpawnAndGoal(randomizationRange);
            }

            // No modo random do ciclo infinito, aplica variacao adicional no spawn e no goal.
            if (useWinRateCycle && randomModeEnabled)
            {
                float spawnVariationX = Random.Range(-fullMapSpawnVariationRange, fullMapSpawnVariationRange);
                float spawnVariationZ = Random.Range(-fullMapSpawnVariationRange, fullMapSpawnVariationRange);
                currentSpawnPoint += new Vector3(spawnVariationX, 0f, spawnVariationZ);

                if (goal != null)
                {
                    float goalVariationX = Random.Range(-fullMapSpawnVariationRange, fullMapSpawnVariationRange);
                    float goalVariationZ = Random.Range(-fullMapSpawnVariationRange, fullMapSpawnVariationRange);
                    goal.position = new Vector3(
                        goal.position.x + goalVariationX,
                        goal.position.y,
                        goal.position.z + goalVariationZ
                    );
                    Debug.Log($"[ParkourEnv] RandomMode: Spawn +({spawnVariationX:F2}, {spawnVariationZ:F2})m | Goal +({goalVariationX:F2}, {goalVariationZ:F2})m");
                }
            }

            ApplyPerEnvironmentOffsets();

            if (!justSpawned)
            {
                RespawnMario();
            }
            justSpawned = false;
        }

        public Vector3 GetCurrentSpawnPoint()
        {
            return currentSpawnPoint;
        }

        private void RandomizeSpawnAndGoal(float range)
        {
            if (spawnPoints != null && spawnPoints.Count > 0)
            {
                float rangeX = range;
                float rangeZ = range;

                if (TryGetGroundBounds(currentSpawnPoint, out Bounds groundBounds))
                {
                    rangeX = Mathf.Min(range, Mathf.Max(MIN_RANDOMIZATION_RANGE, groundBounds.extents.x * SPAWN_RANDOMIZATION_BOUNDS_FACTOR));
                    rangeZ = Mathf.Min(range, Mathf.Max(MIN_RANDOMIZATION_RANGE, groundBounds.extents.z * SPAWN_RANDOMIZATION_BOUNDS_FACTOR));
                }

                float offsetX = UnityEngine.Random.Range(-rangeX, rangeX);
                float offsetZ = UnityEngine.Random.Range(-rangeZ, rangeZ);
                currentSpawnPoint += new Vector3(offsetX, 0, offsetZ);
            }

            if (goal != null)
            {
                float offsetX = UnityEngine.Random.Range(-range, range);
                float offsetZ = UnityEngine.Random.Range(-range, range);
                goal.position = originalGoalPosition + new Vector3(offsetX, 0, offsetZ);
            }
        }

        private float GetCurriculumLessonValue()
        {
            if (useWinRateCycle)
                return 0f;

            if (!useCurriculumLessonForSpawn)
                return -1f;

            var academy = Unity.MLAgents.Academy.Instance;
            if (academy == null)
                return -1f;

            return academy.EnvironmentParameters.GetWithDefault(curriculumLessonParameter, -1f);
        }

        private float GetCurriculumRandomizationRange()
        {
            if (!randomizePlatforms)
                return 0f;

            if (!useCurriculumForRandomization)
                return Mathf.Max(0f, manualPlatformRandomizationRange);

            float lessonValue = GetCurriculumLessonValue();
            if (lessonValue < 0f)
                return 0f;

            int lesson = Mathf.Max(0, Mathf.RoundToInt(lessonValue));
            if (lesson < randomizationStartsAtLesson)
                return 0f;

            float maxRange = platformRandomizationRange > 0f ? platformRandomizationRange : 1.5f;
            if (lesson >= randomizationMaxesAtLesson)
                return maxRange;

            float t = Mathf.InverseLerp(randomizationStartsAtLesson, Mathf.Max(randomizationMaxesAtLesson, randomizationStartsAtLesson + 1), lesson);
            return Mathf.Lerp(0f, maxRange, t);
        }

        private bool TryGetGroundBounds(Vector3 position, out Bounds bounds)
        {
            Vector3 rayOrigin = position + Vector3.up * 10f;

            if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, 50f, Physics.DefaultRaycastLayers, QueryTriggerInteraction.Ignore))
            {
                bounds = hit.collider.bounds;
                return true;
            }

            bounds = new Bounds();
            return false;
        }

        private void UpdateSpawnSelectionFromCurriculum()
        {
            // Com win-rate cycle, o spawn point eh controlado pelo ReportEpisodeResult (avanço de lição)
            // Mas ainda aplicamos o clamping para garantir que nao saia dos limites
            if (useWinRateCycle)
            {
                selectedSpawnPointIndex = Mathf.Clamp(selectedSpawnPointIndex, 0, spawnPoints.Count - 1);
                return;
            }

            if (!useCurriculumLessonForSpawn || spawnPoints == null || spawnPoints.Count == 0)
                return;

            var academy = Unity.MLAgents.Academy.Instance;
            if (academy == null)
                return;

            float lessonValue = academy.EnvironmentParameters.GetWithDefault(curriculumLessonParameter, -1f);
            if (lessonValue < 0f)
                return;

            int lesson = Mathf.Max(0, Mathf.RoundToInt(lessonValue));
            int mappedIndex = lessonZeroUsesFirstSpawnPoint
                ? lesson
                : (spawnPoints.Count - 1 - lesson);

            selectedSpawnPointIndex = Mathf.Clamp(mappedIndex, 0, spawnPoints.Count - 1);

            if (lesson != lastLoggedCurriculumLesson || selectedSpawnPointIndex != lastLoggedSpawnIndex)
            {
                Debug.Log($"[ParkourEnv] Curriculum '{curriculumLessonParameter}'={lesson} => spawnPoints[{selectedSpawnPointIndex}]");
                lastLoggedCurriculumLesson = lesson;
                lastLoggedSpawnIndex = selectedSpawnPointIndex;
            }
        }

        public void ReportEpisodeResult(bool success)
        {
            if (!useWinRateCycle)
                return;

            int targetWindow = Mathf.Max(1, winRateWindowSize);
            recentEpisodeResults.Enqueue(success);
            while (recentEpisodeResults.Count > targetWindow)
            {
                recentEpisodeResults.Dequeue();
            }

            if (recentEpisodeResults.Count < targetWindow)
                return;

            int wins = 0;
            foreach (bool result in recentEpisodeResults)
            {
                if (result)
                    wins++;
            }

            float winRate = (float)wins / targetWindow;
            if (winRate >= winRateThreshold)
            {
                // Avanca para o proximo spawn point (proxima lição)
                int oldSpawnIndex = selectedSpawnPointIndex;
                selectedSpawnPointIndex = Mathf.Min(selectedSpawnPointIndex + 1, spawnPoints.Count - 1);
                recentEpisodeResults.Clear();

                if (selectedSpawnPointIndex != oldSpawnIndex)
                {
                    Debug.Log($"[ParkourEnv] Win-rate {winRate:P0} atingiu meta ({winRateThreshold:P0}). Avançando: spawnPoints[{oldSpawnIndex}] -> spawnPoints[{selectedSpawnPointIndex}]");
                }
                else
                {
                    Debug.Log($"[ParkourEnv] Win-rate {winRate:P0} atingiu meta ({winRateThreshold:P0}). Já está no spawn final [spawnPoints[{selectedSpawnPointIndex}]]. Alternando modo random.");
                    randomModeEnabled = !randomModeEnabled;
                }
            }
            else
            {
                Debug.Log($"[ParkourEnv] Win-rate atual: {winRate:P0} ({wins}/{targetWindow}) | Spawn: [{selectedSpawnPointIndex}]");
            }
        }

        private void SpawnMario()
        {
            if (marioPrefab == null)
            {
                #if UNITY_EDITOR
                string marioPath = "Assets/Mario.prefab";
                marioPrefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>(marioPath);
                #endif
                
                if (marioPrefab == null)
                {
                    Debug.LogError("[ParkourEnv] Mario prefab nao atribuido!");
                    return;
                }
            }

            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 2f;
            
            currentMario = new GameObject("MarioRL");
            currentMario.SetActive(false);
            currentMario.transform.position = spawnPos;
            
            // Adicionar agente RL PRIMEIRO
            MarioRLAgent agent = currentMario.AddComponent<MarioRLAgent>();
            
            // Input provider
            currentMario.AddComponent<MarioInputProvider>();
            
            // SM64Mario
            SM64Mario sm64Mario = currentMario.AddComponent<SM64Mario>();
            
            // Configurar material
            Material matToUse = marioMaterial;
            if (matToUse == null)
            {
                SM64Mario prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    System.Reflection.FieldInfo matField = typeof(SM64Mario).GetField("material", 
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matToUse = matField.GetValue(prefabMario) as Material;
                }
            }
            if (matToUse != null)
            {
                System.Reflection.FieldInfo materialField = typeof(SM64Mario).GetField("material", 
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, matToUse);
            }

            // Configurar Behavior Parameters
            var behaviorParams = currentMario.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams == null) 
                behaviorParams = currentMario.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            
            behaviorParams.BehaviorName = "MarioParkour";
            behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            TryResolveWarmStartModel();
            if (useWarmStartModel && warmStartModel != null)
            {
                behaviorParams.Model = warmStartModel;
            }
            behaviorParams.BrainParameters.VectorObservationSize = 30;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
            // 2 acoes continuas (joystick X/Y) + 1 discreta (Jump com 2 opcoes: 0=nao, 1=sim)
            behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2 });

            // Decision Requester (estilo old: 2 para respostas mais rapidas em plataforma/salto)
            var decisionRequester = currentMario.GetComponent<Unity.MLAgents.DecisionRequester>();
            if (decisionRequester == null)
                decisionRequester = currentMario.AddComponent<Unity.MLAgents.DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.TakeActionsBetweenDecisions = true;

            agent.SetEnvironment(this);
            if (goal != null)
                agent.SetTargetGoal(goal);
                
            currentMario.SetActive(true);
            justSpawned = true;
            Debug.Log($"[ParkourEnv] Mario spawned at {spawnPos} | Goal at {(goal != null ? goal.position.ToString() : "null")}");
        }

        private void RespawnMario()
        {
            if (currentMario != null)
            {
                Vector3 respawnPosition = currentSpawnPoint + Vector3.up * 2f;
                SM64Mario sm64Mario = currentMario.GetComponent<SM64Mario>();
                if (sm64Mario != null)
                {
                    currentMario.transform.position = respawnPosition;

                    Rigidbody rb = currentMario.GetComponent<Rigidbody>();
                    if (rb != null)
                    {
                        rb.velocity = Vector3.zero;
                        rb.angularVelocity = Vector3.zero;
                    }

                    if (sm64Mario.isActiveAndEnabled)
                    {
                        sm64Mario.Teleport(respawnPosition);
                    }
                    else
                    {
                        currentMario.transform.position = respawnPosition;
                    }
                }
                else
                {
                    currentMario.transform.position = respawnPosition;
                }

                StartCoroutine(ValidateRespawnPosition(respawnPosition));
            }
            else
            {
                SpawnMario();
            }
        }

        private IEnumerator ValidateRespawnPosition(Vector3 expectedPosition)
        {
            yield return new WaitForFixedUpdate();

            if (currentMario == null)
                yield break;

            Vector3 marioPos = currentMario.transform.position;
            float horizontalDistance = Vector3.Distance(
                new Vector3(marioPos.x, 0f, marioPos.z),
                new Vector3(expectedPosition.x, 0f, expectedPosition.z)
            );

            bool invalidRespawn = horizontalDistance > 1.5f || marioPos.y < (expectedPosition.y - 1.0f);
            if (!invalidRespawn)
                yield break;

            Debug.LogWarning($"[ParkourEnv] Respawn inconsistente detectado. Forcando reposicionamento. Esperado={expectedPosition} Atual={marioPos}");

            SM64Mario sm64Mario = currentMario.GetComponent<SM64Mario>();
            currentMario.transform.position = expectedPosition;
            if (sm64Mario != null && sm64Mario.isActiveAndEnabled)
            {
                sm64Mario.Teleport(expectedPosition);
            }
        }

        void OnDrawGizmos()
        {
            if (spawnPoints != null)
            {
                for (int i = 0; i < spawnPoints.Count; i++)
                {
                    if (spawnPoints[i] == null)
                        continue;

                    Gizmos.color = i == selectedSpawnPointIndex ? Color.green : Color.cyan;
                    Gizmos.DrawWireSphere(spawnPoints[i].position, 1f);
                }
            }

            if (goal != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(goal.position, 1f);
            }
        }
    }
}

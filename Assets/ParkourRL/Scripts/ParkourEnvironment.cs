using UnityEngine;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    public class ParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Transform marioSpawnPoint;
        [SerializeField] private Material marioMaterial;

        [Header("Level Elements")]
        [SerializeField] private Transform goal;
        [SerializeField] private List<Checkpoint> checkpoints = new List<Checkpoint>();
        [SerializeField] private List<Transform> platformSpawnPoints = new List<Transform>();

        [Header("Randomization")]
        [SerializeField] private bool randomizePlatforms = false;
        [SerializeField] private float platformRandomizationRange = 0.5f;
        [SerializeField] private List<GameObject> platformPrefabs;

        [Header("Multi-Agent Parallel Training")]
        [SerializeField] private int parallelEnvironments = 4;
        [SerializeField] private float environmentSpacing = 30f;

        private Vector3 currentSpawnPoint;
        private GameObject currentMario;
        private Vector3[] originalPlatformPositions;
        
        private List<ParallelEnvInstance> parallelInstances = new List<ParallelEnvInstance>();

        private class ParallelEnvInstance
        {
            public GameObject root;
            public GameObject mario;
            public Transform spawnPoint;
            public Transform goalTransform;
            public ParkourEnvironment envScript;
        }

        private bool justSpawned = false;

        void Awake()
        {
            currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;

            // Salvar posicoes originais
            if (platformSpawnPoints != null && platformSpawnPoints.Count > 0)
            {
                originalPlatformPositions = new Vector3[platformSpawnPoints.Count];
                for (int i = 0; i < platformSpawnPoints.Count; i++)
                {
                    if (platformSpawnPoints[i] != null)
                        originalPlatformPositions[i] = platformSpawnPoints[i].position;
                }
            }

            // Coletar checkpoints
            if (checkpoints.Count == 0)
            {
                Checkpoint[] foundCheckpoints = FindObjectsOfType<Checkpoint>();
                checkpoints.AddRange(foundCheckpoints);
            }
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

            // PASSO 3: Agora sim, recarregar terreno no SM64 com TODAS as plataformas
            SM64Context.RefreshStaticTerrain();
            LogTerrainInfo();
            
            // PASSO 4: Spawnar Mario
            Invoke(nameof(SpawnMario), 0.5f);
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
                
                // Criar SpawnPoint
                GameObject spawnClone = new GameObject($"SpawnPoint_Env{i}");
                spawnClone.transform.position = (marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero) + offset;
                spawnClone.transform.parent = envRoot.transform;
                
                // Criar ParkourEnvironment para esta copia
                GameObject envControllerObj = new GameObject($"ParkourController_Env{i}");
                envControllerObj.transform.position = offset;
                envControllerObj.transform.parent = envRoot.transform;
                
                ParkourEnvironment envScript = envControllerObj.AddComponent<ParkourEnvironment>();
                envScript.parallelEnvironments = 0; // Impede recursao
                envScript.marioSpawnPoint = spawnClone.transform;
                envScript.goal = goalClone != null ? goalClone.transform : null;
                envScript.marioPrefab = this.marioPrefab;
                envScript.marioMaterial = this.marioMaterial;
                envScript.randomizePlatforms = false;
                
                parallelInstances.Add(new ParallelEnvInstance
                {
                    root = envRoot,
                    spawnPoint = spawnClone.transform,
                    goalTransform = goalClone != null ? goalClone.transform : null,
                    envScript = envScript
                });
            }
            
            Debug.Log($"[ParkourEnv] {parallelEnvironments - 1} ambientes paralelos criados.");
        }

        public void ResetEnvironment()
        {
            ResetCheckpoints();
            if (!justSpawned)
            {
                RespawnMario();
            }
            justSpawned = false;
        }

        public void SetCheckpoint(Vector3 position)
        {
            currentSpawnPoint = position;
        }

        private void ResetCheckpoints()
        {
            foreach (var checkpoint in checkpoints)
            {
                if (checkpoint != null)
                    checkpoint.Reset();
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

            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 1f;
            
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
            behaviorParams.BrainParameters.VectorObservationSize = 30;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
            // 2 acoes continuas (joystick X/Y) + 1 discreta (Jump com 2 opcoes: 0=nao, 1=sim)
            behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2 });

            // Decision Requester
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
                currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;
                
                SM64Mario sm64Mario = currentMario.GetComponent<SM64Mario>();
                if (sm64Mario != null)
                {
                    sm64Mario.Teleport(currentSpawnPoint + Vector3.up * 1f);
                }
                else
                {
                    currentMario.transform.position = currentSpawnPoint + Vector3.up * 1f;
                }
            }
            else
            {
                currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;
                SpawnMario();
            }
        }

        void OnDrawGizmos()
        {
            if (marioSpawnPoint != null)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawWireSphere(marioSpawnPoint.position, 1f);
            }

            if (goal != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(goal.position, 1f);
            }
        }
    }
}

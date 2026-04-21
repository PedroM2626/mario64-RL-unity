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
        [SerializeField] private bool randomizePlatforms = true;
        [SerializeField] private float platformRandomizationRange = 2f;
        [SerializeField] private List<GameObject> platformPrefabs;

        private Vector3 currentSpawnPoint;
        private GameObject currentMario;
        private Vector3[] originalPlatformPositions;

        void Awake()
        {
            // PRIMEIRO: Garantir que todas as plataformas com SM64StaticTerrain tenham MeshCollider
            // Isso deve ser feito antes de qualquer coisa relacionada ao SM64
            EnsureMeshColliders();
            
            currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;

            // Salvar posições originais das plataformas
            if (platformSpawnPoints != null && platformSpawnPoints.Count > 0)
            {
                originalPlatformPositions = new Vector3[platformSpawnPoints.Count];
                for (int i = 0; i < platformSpawnPoints.Count; i++)
                {
                    if (platformSpawnPoints[i] != null)
                        originalPlatformPositions[i] = platformSpawnPoints[i].position;
                }
            }

            // Coletar checkpoints automaticamente se não estiverem atribuídos
            if (checkpoints.Count == 0)
            {
                Checkpoint[] foundCheckpoints = FindObjectsOfType<Checkpoint>();
                checkpoints.AddRange(foundCheckpoints);
            }
        }

        void EnsureMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            int addedCount = 0;
            foreach (var terrain in terrains)
            {
                if (terrain.GetComponent<MeshCollider>() == null)
                {
                    MeshFilter meshFilter = terrain.GetComponent<MeshFilter>();
                    if (meshFilter != null && meshFilter.sharedMesh != null)
                    {
                        MeshCollider mc = terrain.gameObject.AddComponent<MeshCollider>();
                        mc.sharedMesh = meshFilter.sharedMesh;
                        mc.convex = false;
                        addedCount++;
                    }
                    else
                    {
                        // Se não tem MeshFilter, adicionar BoxCollider como fallback
                        if (terrain.GetComponent<BoxCollider>() == null)
                        {
                            terrain.gameObject.AddComponent<BoxCollider>();
                            addedCount++;
                        }
                    }
                }
            }
            // Debug.Log($"[ParkourEnvironment] Added {addedCount} colliders to terrain objects");
        }

        void Start()
        {
            // Atualizar terreno após adicionar MeshColliders
            SM64Context.RefreshStaticTerrain();
            // Debug.Log("[ParkourEnvironment] Static terrain refreshed");
            
            // Delay maior para garantir que o SM64Context processou o terreno completamente
            Invoke(nameof(SpawnMario), 0.5f);
        }

        private bool justSpawned = false;
        
        public void ResetEnvironment()
        {
            ResetCheckpoints();
            RandomizeLevel();
            // Evitar recriar Mario se acabou de ser criado (previne loop no inicio)
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

        private void RandomizeLevel()
        {
            if (!randomizePlatforms || platformSpawnPoints == null || platformSpawnPoints.Count == 0)
                return;

            for (int i = 0; i < platformSpawnPoints.Count; i++)
            {
                if (platformSpawnPoints[i] == null) continue;

                Vector3 originalPos = originalPlatformPositions[i];
                Vector3 randomizedPos = originalPos + new Vector3(
                    Random.Range(-platformRandomizationRange, platformRandomizationRange),
                    Random.Range(-platformRandomizationRange * 0.5f, platformRandomizationRange * 0.5f),
                    Random.Range(-platformRandomizationRange, platformRandomizationRange)
                );

                platformSpawnPoints[i].position = randomizedPos;

                // Randomizar rotação sutil
                platformSpawnPoints[i].rotation = Quaternion.Euler(
                    Random.Range(-10f, 10f),
                    Random.Range(-30f, 30f),
                    Random.Range(-10f, 10f)
                );
            }

            // Atualizar terreno estático no SM64
            SM64Context.RefreshStaticTerrain();
        }

        private void SpawnMario()
        {
            // Carregar prefab padrão se não atribuído
            if (marioPrefab == null)
            {
                #if UNITY_EDITOR
                // Usar o Mario.prefab original (não o MarioRL.prefab que pode ter GUIDs quebrados)
                string marioPath = "Assets/Mario.prefab";
                marioPrefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>(marioPath);
                #endif
                
                if (marioPrefab == null)
                {
                    Debug.LogError("Mario prefab não atribuído! Arraste Assets/Mario.prefab para o campo Mario Prefab no ParkourEnvironment.");
                    return;
                }
            }

            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 2f;
            
            // Criar objeto vazio - IMPORTANTE: ordem dos componentes é crítica!
            currentMario = new GameObject("MarioRL");
            currentMario.SetActive(false); // Inativa para não rodar OnEnable do SM64Mario prematuramente
            currentMario.transform.position = spawnPos;
            
            // ETAPA 1: Adicionar agente RL PRIMEIRO (MarioInputProvider vai precisar dele)
            MarioRLAgent agent = currentMario.AddComponent<MarioRLAgent>();
            
            // ETAPA 2: Adicionar input provider (vai encontrar MarioRLAgent lazy)
            currentMario.AddComponent<MarioInputProvider>();
            
            // ETAPA 3: Adicionar SM64Mario (vai encontrar MarioInputProvider no OnEnable)
            SM64Mario sm64Mario = currentMario.AddComponent<SM64Mario>();
            
            // ETAPA 4: Configurar material (copiar do prefab ou usar o assignado)
            Material matToUse = marioMaterial;
            if (matToUse == null)
            {
                SM64Mario prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    System.Reflection.FieldInfo matField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matToUse = matField.GetValue(prefabMario) as Material;
                }
            }
            Debug.Log($"[ParkourEnvironment] Material obtido: {matToUse?.name ?? "NULL"}");
            if (matToUse != null)
            {
                System.Reflection.FieldInfo materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                {
                    materialField.SetValue(sm64Mario, matToUse);
                    Debug.Log($"[ParkourEnvironment] Material aplicado via reflection: {matToUse.name}");
                }
                else
                {
                    Debug.LogError("[ParkourEnvironment] Field 'material' não encontrado no SM64Mario!");
                }
            }
            else
            {
                Debug.LogError("[ParkourEnvironment] Material é NULL! Mario vai aparecer rosa.");
            }

            // ETAPA 5: Configurar Behavior Parameters (Usar GetComponent pois Agent já adiciona via RequireComponent)
            var behaviorParams = currentMario.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams == null) 
            {
                behaviorParams = currentMario.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            }
            
            behaviorParams.BehaviorName = "MarioParkour";
            behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            
            // DEBUG: Verificar valor antes e depois
            Debug.Log($"[ParkourEnvironment] Configurando BehaviorParameters...");
            behaviorParams.BrainParameters.VectorObservationSize = 30;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });
            Debug.Log($"[ParkourEnvironment] BehaviorParameters configurados: VectorObservationSize={behaviorParams.BrainParameters.VectorObservationSize}");

            // ETAPA 6: Adicionar Decision Requester (SE ISSO FALTAR, O AGENTE TRAVA NOS MESMOS MOVIMENTOS)
            var decisionRequester = currentMario.GetComponent<Unity.MLAgents.DecisionRequester>();
            if (decisionRequester == null)
            {
                decisionRequester = currentMario.AddComponent<Unity.MLAgents.DecisionRequester>();
            }
            decisionRequester.DecisionPeriod = 2; // Reduzido de 5 para permitir timing de pulo
            decisionRequester.TakeActionsBetweenDecisions = true;

            agent.SetEnvironment(this);
            if (goal != null)
                agent.SetTargetGoal(goal);
                
            // Ativa o objeto, permitindo que Awake/OnEnable executem com as refs corretas (material preenchido)
            currentMario.SetActive(true);
            justSpawned = true;
            Debug.Log("[ParkourEnvironment] SpawnMario completo. justSpawned=true");
        }

        private void RespawnMario()
        {
            if (currentMario != null)
            {
                currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;
                
                // Reinicia a física do SM64 para forçar o respawn no novo local
                SM64Mario sm64Mario = currentMario.GetComponent<SM64Mario>();
                if (sm64Mario != null)
                {
                    sm64Mario.Teleport(currentSpawnPoint + Vector3.up * 2f); // Usa a rotina nova que NÃO destrói Meshes e previne leaks
                }
                else
                {
                    currentMario.transform.position = currentSpawnPoint + Vector3.up * 2f;
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
            // Desenhar spawn point
            if (marioSpawnPoint != null)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawWireSphere(marioSpawnPoint.position, 1f);
                Gizmos.DrawLine(marioSpawnPoint.position, marioSpawnPoint.position + Vector3.up * 2f);
            }

            // Desenhar goal
            if (goal != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(goal.position, 1f);
                Gizmos.DrawLine(goal.position, goal.position + Vector3.up * 3f);
            }

            // Desenhar conexões entre checkpoints
            if (checkpoints != null && checkpoints.Count > 0)
            {
                Gizmos.color = Color.cyan;
                for (int i = 0; i < checkpoints.Count - 1; i++)
                {
                    if (checkpoints[i] != null && checkpoints[i + 1] != null)
                    {
                        Gizmos.DrawLine(checkpoints[i].transform.position, checkpoints[i + 1].transform.position);
                    }
                }
            }
        }
    }
}

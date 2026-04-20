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
            Debug.Log($"[ParkourEnvironment] Added {addedCount} colliders to terrain objects");
        }

        void Start()
        {
            // Atualizar terreno após adicionar MeshColliders
            SM64Context.RefreshStaticTerrain();
            
            // Delay para garantir que o SM64Context processou o terreno
            Invoke(nameof(SpawnMario), 0.1f);
        }

        public void ResetEnvironment()
        {
            ResetCheckpoints();
            RandomizeLevel();
            RespawnMario();
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

            Vector3 spawnPos = currentSpawnPoint;
            currentMario = Instantiate(marioPrefab, spawnPos, Quaternion.identity);
            currentMario.name = "MarioRL";

            SM64Mario sm64Mario = currentMario.GetComponent<SM64Mario>();

            // ETAPA 1: Desabilitar SM64Mario (se estiver habilitado) para parar de usar o input antigo
            if (sm64Mario != null && sm64Mario.enabled)
            {
                sm64Mario.enabled = false;
            }

            // ETAPA 2: Configurar material
            if (marioMaterial != null && sm64Mario != null)
            {
                System.Reflection.FieldInfo materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, marioMaterial);
            }

            // ETAPA 3: Remover input provider manual IMEDIATAMENTE
            var oldInput = currentMario.GetComponent<ExampleInputProvider>();
            if (oldInput != null)
            {
                DestroyImmediate(oldInput);
                Debug.Log("[ParkourEnvironment] ExampleInputProvider removido");
            }

            // ETAPA 4: Adicionar input provider RL
            MarioInputProvider inputProvider = currentMario.GetComponent<MarioInputProvider>();
            if (inputProvider == null)
            {
                inputProvider = currentMario.AddComponent<MarioInputProvider>();
                Debug.Log("[ParkourEnvironment] MarioInputProvider adicionado");
            }

            // ETAPA 5: Adicionar agente RL
            MarioRLAgent agent = currentMario.GetComponent<MarioRLAgent>();
            if (agent == null)
                agent = currentMario.AddComponent<MarioRLAgent>();

            // ETAPA 6: Configurar Behavior Parameters para ML-Agents
            Unity.MLAgents.Policies.BehaviorParameters behaviorParams = currentMario.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams == null)
            {
                behaviorParams = currentMario.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
                behaviorParams.BehaviorName = "MarioParkour";
                
                // Configurar observações (26 valores)
                behaviorParams.BrainParameters.VectorObservationSize = 26;
                behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
                
                // Configurar ações: 2 contínuas (joystick X, Y) + 3 discretas (Jump, Kick, Stomp) com 2 opções cada
                behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });
            }

            agent.SetEnvironment(this);
            if (goal != null)
                agent.SetTargetGoal(goal);

            // ETAPA 7: SÓ AGORA reabilitar SM64Mario (ele vai encontrar o MarioInputProvider)
            if (sm64Mario != null)
            {
                sm64Mario.enabled = true;
                Debug.Log("[ParkourEnvironment] SM64Mario habilitado com novo input provider");
            }
        }

        private void RespawnMario()
        {
            if (currentMario != null)
            {
                Destroy(currentMario);
            }
            currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;
            SpawnMario();
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

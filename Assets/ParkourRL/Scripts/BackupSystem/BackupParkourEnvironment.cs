using UnityEngine;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL.BackupSystem
{
    public class BackupParkourEnvironment : MonoBehaviour
    {

        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Transform marioSpawnPoint;
        [SerializeField] private Material marioMaterial;

        [Header("Level Elements")]
        [SerializeField] private Transform goal;
        [SerializeField] private List<BackupCheckpoint> checkpoints = new List<BackupCheckpoint>();
        [SerializeField] private List<Transform> platformSpawnPoints = new List<Transform>();

        [Header("Randomization")]
        [SerializeField] private List<GameObject> platformPrefabs = new List<GameObject>();

        [Header("Multi-Agent Parallel Training")]
        [SerializeField] private int parallelEnvironments = 4;
        [SerializeField] private float environmentSpacing = 30f;

        private Vector3 currentSpawnPoint;
        private GameObject currentMario;
        private readonly List<ParallelEnvInstance> parallelInstances = new List<ParallelEnvInstance>();

        private class ParallelEnvInstance
        {
            public GameObject root;
            public Transform spawnPoint;
            public Transform goalTransform;
            public BackupParkourEnvironment envScript;
        }

        private bool justSpawned = false;

        void Awake()
        {
            currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;

            if (checkpoints.Count == 0)
            {
                BackupCheckpoint[] foundCheckpoints = FindObjectsOfType<BackupCheckpoint>();
                checkpoints.AddRange(foundCheckpoints);
            }
        }

        void Start()
        {
            EnsureAllMeshColliders();

            if (parallelEnvironments > 1 && transform.parent == null)
            {
                SpawnParallelEnvironments();
            }

            SM64Context.RefreshStaticTerrain();
            Invoke(nameof(SpawnInitialMario), 0.5f);
        }

        private void SpawnInitialMario()
        {

            SpawnMario();
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();

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
                    }
                }
            }
        }

        private void SpawnParallelEnvironments()
        {
            List<GameObject> scenePlatforms = new List<GameObject>();
            foreach (var terrain in FindObjectsOfType<SM64StaticTerrain>())
            {
                if (terrain.transform.parent == null)
                    scenePlatforms.Add(terrain.gameObject);
            }

            for (int i = 1; i < parallelEnvironments; i++)
            {
                Vector3 offset = new Vector3(0, 0, environmentSpacing * i);

                GameObject envRoot = new GameObject($"BackupParallelEnv_{i}");
                envRoot.transform.position = offset;

                foreach (var platform in scenePlatforms)
                {
                    GameObject clone = Instantiate(platform, platform.transform.position + offset, platform.transform.rotation, envRoot.transform);
                    clone.name = platform.name + $"_BackupEnv{i}";
                    clone.transform.localScale = platform.transform.localScale;
                }

                GameObject goalClone = null;
                if (goal != null)
                {
                    goalClone = Instantiate(goal.gameObject, goal.position + offset, goal.rotation, envRoot.transform);
                    goalClone.name = $"BackupGoal_Env{i}";
                    goalClone.tag = "Goal";
                }

                GameObject spawnClone = new GameObject($"BackupSpawnPoint_Env{i}");
                spawnClone.transform.position = (marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero) + offset;
                spawnClone.transform.parent = envRoot.transform;

                GameObject envControllerObj = new GameObject($"BackupParkourController_Env{i}");
                envControllerObj.transform.position = offset;
                envControllerObj.transform.parent = envRoot.transform;

                BackupParkourEnvironment envScript = envControllerObj.AddComponent<BackupParkourEnvironment>();
                envScript.parallelEnvironments = 0;
                envScript.marioSpawnPoint = spawnClone.transform;
                envScript.goal = goalClone != null ? goalClone.transform : null;
                envScript.marioPrefab = this.marioPrefab;
                envScript.marioMaterial = this.marioMaterial;

                parallelInstances.Add(new ParallelEnvInstance
                {
                    root = envRoot,
                    spawnPoint = spawnClone.transform,
                    goalTransform = goalClone != null ? goalClone.transform : null,
                    envScript = envScript
                });
            }
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
                    checkpoint.ResetCheckpoint();
            }
        }

        private void SpawnMario()
        {
            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 1f;

            currentMario = new GameObject("BackupMarioRL");
            currentMario.SetActive(false);
            currentMario.transform.position = spawnPos;

            BackupMarioRLAgent agent = currentMario.AddComponent<BackupMarioRLAgent>();
            currentMario.AddComponent<ParkourRL.MarioInputProvider>();

            SM64Mario sm64Mario = currentMario.AddComponent<SM64Mario>();
            if (marioMaterial != null)
            {
                System.Reflection.FieldInfo materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, marioMaterial);
            }

            var behaviorParams = currentMario.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams == null)
                behaviorParams = currentMario.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();

            behaviorParams.BehaviorName = "MarioParkour_Old";
            behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            behaviorParams.BrainParameters.VectorObservationSize = 30;
            behaviorParams.BrainParameters.NumStackedVectorObservations = 1;
            behaviorParams.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2 });

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
        }
    }
}

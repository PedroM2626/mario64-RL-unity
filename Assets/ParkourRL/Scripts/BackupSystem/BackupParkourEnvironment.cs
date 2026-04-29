using UnityEngine;
using System.Collections.Generic;
using LibSM64;
using ParkourRL.HybridSystem;

namespace ParkourRL.BackupSystem
{
    public class BackupParkourEnvironment : MonoBehaviour
    {
        public enum StartupMode
        {
            Training,
            Recording
        }

        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Transform marioSpawnPoint;
        [SerializeField] private Material marioMaterial;

        [Header("Mode")]
        [Tooltip("Modo inicial selecionado no Inspector antes de iniciar a cena")]
        [SerializeField] private StartupMode startupMode = StartupMode.Training;

        [Header("Level Elements")]
        [SerializeField] private Transform goal;
        [SerializeField] private List<BackupCheckpoint> checkpoints = new List<BackupCheckpoint>();
        [SerializeField] private List<Transform> platformSpawnPoints = new List<Transform>();

        [Header("Randomization")]
        [SerializeField] private List<GameObject> platformPrefabs = new List<GameObject>();

        [Header("Multi-Agent Parallel Training")]
        [SerializeField] private int parallelEnvironments = 4;
        [SerializeField] private float environmentSpacing = 30f;

        // Public properties for access by inner class RecordingInputProvider
        public Transform marioSpawnPointPublic => marioSpawnPoint;
        public Transform goalPublic => goal;

        private Vector3 currentSpawnPoint;
        private GameObject currentMario;
        private GameObject playerControlledMario;
        private Camera playerFollowCamera;
        private HybridDataRecorder dataRecorder;
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

            // Initialize the recording system for Recording mode
            if (startupMode == StartupMode.Recording)
            {
                dataRecorder = gameObject.AddComponent<HybridDataRecorder>();
                Debug.Log("[BackupEnv] HybridDataRecorder inicializado para modo Recording");
            }

            if (startupMode == StartupMode.Training && parallelEnvironments > 1 && transform.parent == null)
            {
                SpawnParallelEnvironments();
            }

            SM64Context.RefreshStaticTerrain();
            Invoke(nameof(SpawnInitialMario), 0.5f);
        }

        private void SpawnInitialMario()
        {
            if (startupMode == StartupMode.Recording)
            {
                SpawnPlayerControlledMario();
                return;
            }

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
                envScript.startupMode = this.startupMode;
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
                if (startupMode == StartupMode.Recording)
                {
                    RespawnPlayerControlledMario();
                }
                else
                {
                    RespawnMario();
                }
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

        private void SpawnPlayerControlledMario()
        {
            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 1f;

            if (playerControlledMario != null)
            {
                Destroy(playerControlledMario);
            }

            playerControlledMario = new GameObject("BackupMarioPlayer");
            playerControlledMario.SetActive(false);
            playerControlledMario.transform.position = spawnPos;

            var inputProvider = playerControlledMario.AddComponent<RecordingInputProvider>();
            inputProvider.SetEnvironment(this);
            inputProvider.SetRecorder(dataRecorder);
            if (Camera.main != null)
            {
                inputProvider.cameraTransform = Camera.main.transform;
            }

            SM64Mario sm64Mario = playerControlledMario.AddComponent<SM64Mario>();
            if (marioMaterial != null)
            {
                var materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, marioMaterial);
            }

            playerControlledMario.SetActive(true);
            justSpawned = true;

            // Start recording if dataRecorder was created
            if (dataRecorder != null)
            {
                dataRecorder.StartEpisode();
                Debug.Log("[BackupEnv] Recording episode started");
            }

            SetupFollowCamera();
            Debug.Log("[BackupEnv] Recording mode ativo. Mario controlavel spawnado (WASD/Setas + Espaco).");
        }

        private void SetupFollowCamera()
        {
            if (playerControlledMario == null)
                return;

            if (playerFollowCamera == null)
            {
                var camObj = new GameObject("RecordingPlayerCamera");
                playerFollowCamera = camObj.AddComponent<Camera>();
                playerFollowCamera.tag = "Untagged";
            }

            Vector3 targetPos = playerControlledMario.transform.position + new Vector3(0f, 5f, -8f);
            playerFollowCamera.transform.position = targetPos;
            playerFollowCamera.transform.LookAt(playerControlledMario.transform.position + Vector3.up);
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

        private void RespawnPlayerControlledMario()
        {
            currentSpawnPoint = marioSpawnPoint != null ? marioSpawnPoint.position : Vector3.zero;

            if (playerControlledMario == null)
            {
                SpawnPlayerControlledMario();
                return;
            }

            SM64Mario sm64Mario = playerControlledMario.GetComponent<SM64Mario>();
            Vector3 spawnPos = currentSpawnPoint + Vector3.up * 1f;
            if (sm64Mario != null)
            {
                sm64Mario.Teleport(spawnPos);
            }
            else
            {
                playerControlledMario.transform.position = spawnPos;
            }

            SetupFollowCamera();
        }

        public void SetStartupMode(StartupMode mode)
        {
            startupMode = mode;
        }

        /// <summary>
        /// Collects the 30-dimension observation vector for Recording mode.
        /// Maintains the same structure as BackupMarioRLAgent.CollectObservations()
        /// </summary>
        public float[] CollectObservationVector()
        {
            if (playerControlledMario == null)
                return new float[30];

            List<float> obs = new List<float>();
            Vector3 position = playerControlledMario.transform.position;
            Vector3 previousPosition = position;
            RaycastHit[] raycastHitsCache = new RaycastHit[1];
            int raycastCount = 8;
            float raycastDistance = 10f;

            // 0-2: Position (normalized)
            obs.Add(position.x / 25f);
            obs.Add(position.y / 10f);
            obs.Add(position.z / 25f);

            // 3-6: Direction to goal (normalized) or zeros
            if (goal != null)
            {
                Vector3 toGoal = goal.position - position;
                obs.Add(toGoal.x / 25f);
                obs.Add(toGoal.y / 10f);
                obs.Add(toGoal.z / 25f);
                obs.Add(toGoal.magnitude / 30f);
            }
            else
            {
                obs.Add(0f);
                obs.Add(0f);
                obs.Add(0f);
                obs.Add(0f);
            }

            // 7-9: Velocity (normalized, estimated from position change)
            Vector3 velocity = Vector3.zero; // Em modo Recording, mantemos velocity zerado
            obs.Add(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            obs.Add(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            obs.Add(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // 10: Is grounded (raycast down)
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            obs.Add(isGrounded ? 0f : 1f);

            // 11-26: 8 raycasts (distance + height difference each)
            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;

                if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, direction, raycastHitsCache, raycastDistance) > 0)
                {
                    obs.Add(raycastHitsCache[0].distance / raycastDistance);
                    obs.Add(Mathf.Clamp((raycastHitsCache[0].point.y - position.y) / 5f, -1f, 1f));
                }
                else
                {
                    obs.Add(1f);
                    obs.Add(0f);
                }
            }

            // 27: Downward raycast distance
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                obs.Add(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                obs.Add(1f);
            }

            // 28: Jump pressed (0/1) - will be filled by RecordingInputProvider
            obs.Add(0f);

            // 29: Episode time normalized
            obs.Add(0f);

            return obs.ToArray();
        }

        private void LateUpdate()
        {
            if (startupMode != StartupMode.Recording || playerControlledMario == null || playerFollowCamera == null)
                return;

            Vector3 targetPos = playerControlledMario.transform.position + new Vector3(0f, 5f, -8f);
            playerFollowCamera.transform.position = Vector3.Lerp(playerFollowCamera.transform.position, targetPos, 5f * Time.deltaTime);
            playerFollowCamera.transform.LookAt(playerControlledMario.transform.position + Vector3.up);
        }

        public class RecordingInputProvider : SM64InputProvider
        {
            public Transform cameraTransform;
            private BackupParkourEnvironment environment;
            private HybridDataRecorder recorder;

            // Recording state
            private float[] previousObservations;
            private float[] currentObservations;
            private float cumulativeReward = 0f;
            private int stepCount = 0;
            private float episodeStartTime = 0f;
            private float previousDistance = float.MaxValue;
            private Vector3 previousPosition = Vector3.zero;
            private const float MAX_EPISODE_TIME = 30f;
            private const float FALL_THRESHOLD = 3f;

            public void SetEnvironment(BackupParkourEnvironment env)
            {
                environment = env;
            }

            public void SetRecorder(HybridDataRecorder rec)
            {
                recorder = rec;
                episodeStartTime = Time.time;
            }

            public override Vector3 GetCameraLookDirection()
            {
                if (cameraTransform != null)
                    return cameraTransform.forward;
                return Vector3.forward;
            }

            public override Vector2 GetJoystickAxes()
            {
                return new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
            }

            public override bool GetButtonHeld(Button button)
            {
                switch (button)
                {
                    case Button.Jump: return Input.GetButton("Jump");
                    case Button.Kick: return Input.GetMouseButton(0);
                    case Button.Stomp: return Input.GetKey(KeyCode.LeftShift);
                    default: return false;
                }
            }

            void Update()
            {
                // Only record if we have environment and recorder
                if (environment == null || recorder == null || gameObject == null)
                    return;

                // Collect current observations
                currentObservations = environment.CollectObservationVector();
                
                if (previousObservations == null)
                {
                    previousObservations = new float[30];
                    System.Array.Copy(currentObservations, previousObservations, 30);
                    previousPosition = gameObject.transform.position;
                    return;
                }

                // Calculate reward based on progress toward the goal
                float reward = -0.01f; // Penalty por cada passo

                if (environment.goalPublic != null)
                {
                    Vector3 marioPos = gameObject.transform.position;
                    float currentDistance = Vector3.Distance(marioPos, environment.goalPublic.position);
                    
                    if (previousDistance < float.MaxValue)
                    {
                        float distanceDelta = previousDistance - currentDistance;
                        if (distanceDelta > 0.01f)
                        {
                            reward += distanceDelta * 1.0f; // Reward para aproximar do objetivo
                        }
                    }
                    
                    previousDistance = currentDistance;
                }

                // Verificar falha (caiu muito baixo)
                Vector3 spawnPos = environment.marioSpawnPointPublic != null ? environment.marioSpawnPointPublic.position : Vector3.zero;
                if (gameObject.transform.position.y < spawnPos.y - FALL_THRESHOLD)
                {
                    reward = -5.0f;
                    RecordFinalStep(reward, true);
                    return;
                }

                // Verificar tempo maximum
                float elapsedTime = Time.time - episodeStartTime;
                if (elapsedTime >= MAX_EPISODE_TIME)
                {
                    reward = -5.0f;
                    RecordFinalStep(reward, false);
                    return;
                }

                // Gravar step normal
                if (stepCount % 2 == 0) // Gravar a cada 2 frames para reduzir dados
                {
                    float[] actions = new float[3]
                    {
                        GetJoystickAxes().x,
                        GetJoystickAxes().y,
                        GetButtonHeld(Button.Jump) ? 1f : 0f
                    };

                    recorder.RecordStep(previousObservations, actions, reward, currentObservations, false);
                    cumulativeReward += reward;
                }

                stepCount++;
                System.Array.Copy(currentObservations, previousObservations, 30);
                previousPosition = gameObject.transform.position;
            }

            private void RecordFinalStep(float reward, bool success)
            {
                // Record last step
                float[] actions = new float[3]
                {
                    GetJoystickAxes().x,
                    GetJoystickAxes().y,
                    GetButtonHeld(Button.Jump) ? 1f : 0f
                };

                recorder.RecordStep(previousObservations, actions, reward, currentObservations, true);
                cumulativeReward += reward;

                // Salvar episode
                Debug.Log($"[RecordingProvider] Episode finished. Success={success}, Reward={cumulativeReward:F2}, Steps={stepCount}");
                
                // Para salvar, precisamos usar a estrutura esperada pelo SaveEpisode
                // For now, we call FlushBatch to force saving already registered data
                recorder.FlushBatch();

                // Reset para next episode
                cumulativeReward = 0f;
                stepCount = 0;
                episodeStartTime = Time.time;
                previousObservations = null;
                
                // Respawn Mario
                if (environment != null)
                {
                    environment.ResetEnvironment();
                    recorder.StartEpisode();
                }
            }
        }
    }
}

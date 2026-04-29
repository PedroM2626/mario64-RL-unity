using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Linq;

namespace ParkourRL
{
    public class MarioRLAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;
        [SerializeField] private Material marioMaterial;

        [Header("Environment")]
        [SerializeField] private ParkourEnvironment environment;
        [SerializeField] private Transform targetGoal;
        
        [Header("Dynamic Platforms")]
        [Tooltip("Automatically detect SM64StaticTerrain platforms")]
        [SerializeField] private bool autoDetectPlatforms = true;
        [Tooltip("Reward for reaching each platform")]
        [SerializeField] private float platformReward = 15f;
        [Tooltip("Tolerance radius to consider reaching the platform (uses the larger: this value or half the platform size)")]
        [SerializeField] private float platformReachRadius = 3f;

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        private Vector3 startPosition;
        private Vector3 previousPosition;
        private float initialDistanceToGoal;
        private float previousDistanceToGoal;
        private float episodeTime;
        private float bestDistanceToGoal;
        private float bestDistanceWhileGrounded; // Tracks SAFE progress (when he lands on a platform)
        private bool episodeResultReported;
        
        private int currentPlatformIndex = 0; // Next platform Mario must reach
        private bool[] platformsReached;
        private Collider[] detectedPlatforms; // Colliders of detected platforms
        private Vector3[] platformCenters; // Calculated centers of platforms
        
        // Dynamic checkpoint - proximity to next platform
        private float initialDistanceToNextPlatform;
        private float bestDistanceToNextPlatform;
        private float previousDistanceToNextPlatform;

        private const float MAX_EPISODE_TIME = 30f; // Old style: more episodes per hour to converge faster
        
        // Cache array for Raycasts to prevent ALLOC_TEMP_MAIN leakage
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        private float GetCurriculumLessonValue()
        {
            var academy = Unity.MLAgents.Academy.Instance;
            if (academy == null)
                return -1f;

            return academy.EnvironmentParameters.GetWithDefault("spawn_lesson", -1f);
        }

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
        }

        public override void Initialize()
        {
            base.Initialize();
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
        }

        public override void OnEpisodeBegin()
        {
            if (environment != null)
            {
                environment.ResetEnvironment();
                // After reset (which teleports Mario), update startPosition to reflect actual spawn
                startPosition = environment.GetCurrentSpawnPoint();
            }
            else
            {
                transform.position = startPosition;
            }

            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;

            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            bestDistanceWhileGrounded = initialDistanceToGoal; // Initialize safe distance
            previousPosition = transform.position;
            episodeTime = 0f;
            episodeResultReported = false;
            
            // Detect and reset platforms
            DetectPlatforms();
            currentPlatformIndex = 0;
            if (detectedPlatforms != null && detectedPlatforms.Length > 0)
            {
                platformsReached = new bool[detectedPlatforms.Length];
            }
            
            // Initialize checkpoint distances (next platform)
            InitializeCheckpointDistances();

        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Force correct VectorObservationSize
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null && behaviorParams.BrainParameters.VectorObservationSize != 30)
            {
                behaviorParams.BrainParameters.VectorObservationSize = 30;
            }

            Vector3 position = transform.position;
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Mario position (normalized and RELATIVE to the environment)
            sensor.AddObservation(localPosition.x / 25f);
            sensor.AddObservation(localPosition.y / 10f);
            sensor.AddObservation(localPosition.z / 25f);

            // [4 obs] Direction and distance to goal
            if (targetGoal != null)
            {
                Vector3 toGoal = targetGoal.position - position;
                sensor.AddObservation(toGoal.x / 25f);
                sensor.AddObservation(toGoal.y / 10f);
                sensor.AddObservation(toGoal.z / 25f);
                sensor.AddObservation(toGoal.magnitude / 30f);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }

            // [3 obs] Mario velocity
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1 obs] Is airborne?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [16 obs] Raycasts to detect terrain/obstacles (no layer mask)
            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;

                if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, direction, raycastHitsCache, raycastDistance) > 0)
                {
                    sensor.AddObservation(raycastHitsCache[0].distance / raycastDistance);
                    sensor.AddObservation(Mathf.Clamp((raycastHitsCache[0].point.y - position.y) / 5f, -1f, 1f));
                }
                else
                {
                    sensor.AddObservation(1f); // Nothing detected = maximum distance
                    sensor.AddObservation(0f);
                }
            }

            // [1 obs] Ground height below
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f); // No ground = falling
            }

            // [1 obs] Jump button active
            sensor.AddObservation(jumpPressed ? 1f : 0f);

            // [1 obs] Normalized time
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);

            // Total: 3 + 4 + 3 + 1 + 16 + 1 + 1 + 1 = 30
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Continuous actions: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Discrete action: Jump + Kick (for competitive mode too)
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = false;
            stompPressed = false;

            // Point camera toward goal (movement direction)
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== OLD STYLE REWARD (COMPLETE THE REAL PARKOUR) ==========
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();
            float distanceDelta = previousDistanceToGoal - currentDistance;

            // Clear time pressure to prevent staying idle on a platform.
            AddReward(-0.01f);

            // ===== CHECKPOINT REWARD: Proximity to next platform =====
            // This prevents Mario from jumping into the void trying to go directly to the goal
            UpdateCheckpointReward();

            // Dense reward for real progress toward the goal.
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 1.0f);
            }

            // Intermediate progress milestone to stabilize exploration.
            if (currentDistance < bestDistanceToGoal - 2.0f)
            {
                AddReward(2.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // Fall: strong penalty, but without completely preventing exploration.
            if (currentPos.y < startPosition.y - 3.0f)
            {
                AddReward(-5.0f);
                ReportEpisodeResult(false);
                EndEpisode();
                return;
            }

            // Timeout proportional to progress (old style): punishes stalling, rewards real attempt.
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f);
                ReportEpisodeResult(false);
                EndEpisode();
                return;
            }

            // Periodic log
            if (StepCount % 500 == 0 && StepCount > 0)
            {
                Debug.Log($"[Mario] Step {StepCount}: Dist={currentDistance:F1}, Best={bestDistanceToGoal:F1}, " +
                          $"Reward={GetCumulativeReward():F2}, Pos={currentPos}, Y={currentPos.y:F2}");
            }

            // Platform check - reward for following the correct path
            CheckPlatforms();

            // Immediate goal proximity check - uses approximate area
            if (IsInGoalArea() && !episodeResultReported)
            {
                AddReward(50f);
                ReportEpisodeResult(true);
                EndEpisode();
            }
        }

        /// <summary>
        /// Detects SM64StaticTerrain platforms ONLY from the same environment and sorts by distance from spawn
        /// </summary>
        private void DetectPlatforms()
        {
            if (!autoDetectPlatforms)
                return;

            // Calculate environment offset (for parallel environments)
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 spawnPosLocal = startPosition - envOffset; // Position relativa ao environment

            // Find all objects with SM64StaticTerrain
            LibSM64.SM64StaticTerrain[] terrains = FindObjectsOfType<LibSM64.SM64StaticTerrain>();
            List<Collider> platformColliders = new List<Collider>();
            List<Vector3> centers = new List<Vector3>();

            foreach (var terrain in terrains)
            {
                // Skip the death floor (DeathFloor) and floors too far below
                if (terrain.gameObject.name.ToLower().Contains("death") || 
                    terrain.transform.position.y < -15f)
                    continue;

                // FILTER: Only platforms from the same environment (near spawn)
                Vector3 terrainPosLocal = terrain.transform.position - envOffset;
                float horizontalDistToSpawn = Vector3.Distance(
                    new Vector3(terrainPosLocal.x, 0, terrainPosLocal.z),
                    new Vector3(spawnPosLocal.x, 0, spawnPosLocal.z)
                );
                
                // Ignore platforms from other environments (too far > 50m)
                if (horizontalDistToSpawn > 50f)
                    continue;

                Collider col = terrain.GetComponent<Collider>();
                if (col != null)
                {
                    platformColliders.Add(col);
                    centers.Add(col.bounds.center);
                }
            }

            // Sort by distance from spawn (nearest to farthest)
            var sortedIndices = Enumerable.Range(0, platformColliders.Count)
                .OrderBy(i => Vector3.Distance(startPosition, centers[i]))
                .ToList();

            detectedPlatforms = sortedIndices.Select(i => platformColliders[i]).ToArray();
            platformCenters = sortedIndices.Select(i => centers[i]).ToArray();

            if (detectedPlatforms.Length > 0)
            {
                Debug.Log($"[Mario] {detectedPlatforms.Length} platforms detected in environment (offset: {envOffset})");
                for (int i = 0; i < detectedPlatforms.Length; i++)
                {
                    Debug.Log($"  Plataforma {i}: {detectedPlatforms[i].name} em {platformCenters[i]}");
                }
            }
            else
            {
                Debug.LogWarning($"[Mario] No platforms detected in environment (offset: {envOffset})!");
            }
        }

        /// <summary>
        /// Checks if Mario is in the area of the next platform (uses platform bounds + tolerance)
        /// </summary>
        private void CheckPlatforms()
        {
            if (detectedPlatforms == null || detectedPlatforms.Length == 0)
                return;
            if (currentPlatformIndex >= detectedPlatforms.Length)
                return;

            Collider targetPlatform = detectedPlatforms[currentPlatformIndex];
            if (targetPlatform == null)
                return;

            // Calculate reach radius based on platform size
            Bounds bounds = targetPlatform.bounds;
            float platformRadius = Mathf.Max(bounds.extents.x, bounds.extents.z) * 0.8f;
            float reachRadius = Mathf.Max(platformReachRadius, platformRadius);

            // Horizontal distance to platform center
            float horizontalDist = Vector3.Distance(
                new Vector3(transform.position.x, 0, transform.position.z),
                new Vector3(platformCenters[currentPlatformIndex].x, 0, platformCenters[currentPlatformIndex].z)
            );

            // Check if above the platform (height)
            bool isAbovePlatform = transform.position.y >= (bounds.min.y - 0.5f) && 
                                   transform.position.y <= (bounds.max.y + 5f);

            // Check if within the platform area
            if (horizontalDist < reachRadius && isAbovePlatform && !platformsReached[currentPlatformIndex])
            {
                platformsReached[currentPlatformIndex] = true;
                AddReward(platformReward);
                Debug.Log($"[Mario] Plataforma {currentPlatformIndex} ({targetPlatform.name}) reached! " +
                          $"+{platformReward} reward | Dist: {horizontalDist:F1}m, Raio: {reachRadius:F1}m");
                currentPlatformIndex++;
                
                // Reset checkpoint distances for the next platform
                InitializeCheckpointDistances();
            }
        }

        /// <summary>
        /// Initializes checkpoint distances (next platform)
        /// </summary>
        private void InitializeCheckpointDistances()
        {
            if (detectedPlatforms == null || detectedPlatforms.Length == 0)
                return;
            if (currentPlatformIndex >= detectedPlatforms.Length)
                return;

            float distToNext = GetDistanceToNextPlatform();
            initialDistanceToNextPlatform = distToNext;
            bestDistanceToNextPlatform = distToNext;
            previousDistanceToNextPlatform = distToNext;
        }

        /// <summary>
        /// Calculates distance to the next platform (current checkpoint)
        /// </summary>
        private float GetDistanceToNextPlatform()
        {
            if (detectedPlatforms == null || currentPlatformIndex >= detectedPlatforms.Length)
                return float.MaxValue;
            if (platformCenters == null || currentPlatformIndex >= platformCenters.Length)
                return float.MaxValue;

            // Horizontal distance (ignoring height) to the next platform
            return Vector3.Distance(
                new Vector3(transform.position.x, 0, transform.position.z),
                new Vector3(platformCenters[currentPlatformIndex].x, 0, platformCenters[currentPlatformIndex].z)
            );
        }

        /// <summary>
        /// Updates rewards based on proximity to checkpoint (next platform)
        /// Isso evita que o Mario pule no void tentando ir direto ao goal
        /// </summary>
        private void UpdateCheckpointReward()
        {
            if (detectedPlatforms == null || detectedPlatforms.Length == 0)
                return;
            if (currentPlatformIndex >= detectedPlatforms.Length)
                return;

            float currentDistToNext = GetDistanceToNextPlatform();
            
            // Reward for progressing toward the next platform (checkpoint)
            float checkpointDelta = previousDistanceToNextPlatform - currentDistToNext;
            if (checkpointDelta > 0.01f)
            {
                // Reward proportional to progress toward the platform
                // Multiplier larger than goal to prioritize platforms
                AddReward(checkpointDelta * 2.0f);
            }
            
            // Checkpoint progress milestone (every 2m closer)
            if (currentDistToNext < bestDistanceToNextPlatform - 2.0f)
            {
                AddReward(3.0f); // Bonus larger than goal's (which is 2.0f)
                bestDistanceToNextPlatform = currentDistToNext;
                Debug.Log($"[Mario] Checkpoint: {currentDistToNext:F1}m ate plataforma {currentPlatformIndex} | +3 reward");
            }
            
            // Penalty for moving too far from checkpoint (prevent "exploration" in the void)
            if (currentDistToNext > bestDistanceToNextPlatform + 5.0f && bestDistanceToNextPlatform < initialDistanceToNextPlatform - 2.0f)
            {
                // Mario was already progressing but went back/moved too far away
                AddReward(-0.5f);
            }
            
            previousDistanceToNextPlatform = currentDistToNext;
        }

        /// <summary>
        /// Checks if Mario is in the goal area (not exact coordinates)
        /// </summary>
        private bool IsInGoalArea()
        {
            if (targetGoal == null)
                return false;

            // Try to get the goal's Collider to use bounds
            Collider goalCollider = targetGoal.GetComponent<Collider>();
            if (goalCollider != null && goalCollider.isTrigger)
            {
                // If it has a trigger, check if within bounds
                return goalCollider.bounds.Contains(transform.position);
            }

            // Fallback: use distance with larger radius (approximate area)
            float dist = Vector3.Distance(
                new Vector3(transform.position.x, 0, transform.position.z),
                new Vector3(targetGoal.position.x, 0, targetGoal.position.z)
            );
            return dist < 5f; // 5 meter radius for the goal
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");
            continuousActions[1] = Input.GetAxis("Vertical");

            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = Input.GetButton("Jump") ? 1 : 0;
        }

        void Start()
        {
            if (GetComponent<MarioInputProvider>() == null)
            {
                gameObject.AddComponent<MarioInputProvider>();
            }
        }
        
        void FixedUpdate()
        {
            // Stuck Mario detection
            if (StepCount > 0 && StepCount % 300 == 0)
            {
                float moved = Vector3.Distance(transform.position, startPosition);
                if (moved < 0.5f)
                {
                    Debug.LogWarning($"[Mario] PRESO! Step {StepCount}, Pos: {transform.position}, CumReward: {GetCumulativeReward():F2}");
                }
            }
        }

        private float GetDistanceToGoal()
        {
            if (targetGoal == null) return float.MaxValue;
            // Horizontal distance (XZ) to avoid penalizing vertical jumps
            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal") && !episodeResultReported)
            {
                AddReward(50f);
                ReportEpisodeResult(true);
                EndEpisode();
            }
        }

        void OnTriggerStay(Collider other)
        {
            // Backup: if the agent stays inside the goal but OnTriggerEnter doesn't fire
            if (other.CompareTag("Goal") && !episodeResultReported)
            {
                AddReward(50f);
                ReportEpisodeResult(true);
                EndEpisode();
            }
        }

        private void ReportEpisodeResult(bool success)
        {
            if (episodeResultReported)
                return;

            episodeResultReported = true;
            if (environment != null)
            {
                environment.ReportEpisodeResult(success);
            }
        }

        public void SetEnvironment(ParkourEnvironment env)
        {
            environment = env;
        }

        public void SetTargetGoal(Transform goal)
        {
            targetGoal = goal;
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = Color.yellow;
            Vector3 position = transform.position + Vector3.up * 0.5f;

            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
                Gizmos.DrawRay(position, direction * raycastDistance);
            }
        }
    }
}

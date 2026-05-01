using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Hybrid Mario agent: supports recording mode (for IL/Offline RL) and normal training mode.
    /// Works in parallel with a player-controlled Mario for comparison.
    /// </summary>
    public class MarioHybridAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;
        [SerializeField] private Material marioMaterial;

        [Header("Environment")]
        [SerializeField] private HybridParkourEnvironment environment;
        [SerializeField] private Transform targetGoal;
        
        [Header("Hybrid Mode")]
        [Tooltip("Operation mode: Recording = records data for IL/Offline, Training = normal RL training")]
        [SerializeField] private HybridMode currentMode = HybridMode.Training;
        // Field removed - synchronization is done by HybridParkourEnvironment
        
        [Header("Recording Settings")]
        [Tooltip("Reference to the data recorder")]
        [SerializeField] private HybridDataRecorder dataRecorder;
        [Tooltip("Only records successful episodes")]
        [SerializeField] private bool onlyRecordSuccesses = false;
        // Field removed - using local variable

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        // Estado interno
        private Vector3 startPosition;
        private Vector3 previousPosition;
        private float initialDistanceToGoal;
        private float previousDistanceToGoal;
        private float episodeTime;
        private float bestDistanceToGoal;
        // Field removed - not used
        private List<HybridTransition> currentEpisodeData = new List<HybridTransition>();
        
        private const float MAX_EPISODE_TIME = 30f;
        private const float SPAWN_GRACE_PERIOD = 2.0f;  // Grace period after spawn to prevent immediate fall death
        private float lastSpawnTime = 0f;
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        public enum HybridMode
        {
            Training,      // Treino RL normal
            Recording      // Recording for IL/Offline RL
        }

        // Structure to store transitions
        public struct HybridTransition
        {
            public Vector3 position;
            public Vector3 velocity;
            public float[] observations;
            public float[] actions;
            public float reward;
            public bool done;
            public float timestamp;
        }

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
        }

        public void SetMode(HybridMode mode)
        {
            currentMode = mode;
            Debug.Log($"[MarioHybrid] Mode changed to: {mode}");
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            
            // Se estiver em modo Recording, desabilitar ML-Agents behavior
            if (currentMode == HybridMode.Recording)
            {
                DisableMLAgentsBehavior();
            }
        }

        private void DisableMLAgentsBehavior()
        {
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null)
            {
                behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.HeuristicOnly;
            }
        }

        public override void OnEpisodeBegin()
        {
            // Cancel any pending actions to prevent duplicates
            lastSpawnTime = Time.time;  // Reset grace period
            
            if (environment != null)
            {
                // Don't call ResetEnvironment here - it causes infinite loop
                // Just get the spawn point
                startPosition = environment.GetCurrentSpawnPoint();
                
                // IMPORTANT: Do NOT call SetActive in OnEpisodeBegin - it causes recursion!
                // Just reposition and teleport
                Vector3 spawnPos = startPosition + Vector3.up * 2f;
                transform.position = spawnPos;
                
                // Reset velocity
                Rigidbody rb = GetComponent<Rigidbody>();
                if (rb != null)
                {
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;
                }
                
                // Teleport in SM64
                if (marioComponent != null && marioComponent.isActiveAndEnabled)
                {
                    marioComponent.Teleport(spawnPos);
                }
            }
            else
            {
                transform.position = startPosition;
            }

            ResetInputs();
            
            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;
            
            // Clear data from previous episode
            currentEpisodeData.Clear();
            
            // Notify recorder of new episode
            if (dataRecorder != null && currentMode == HybridMode.Recording)
            {
                dataRecorder.StartEpisode();
            }
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Normalized position
            sensor.AddObservation(localPosition.x / 25f);
            sensor.AddObservation(localPosition.y / 10f);
            sensor.AddObservation(localPosition.z / 25f);

            // [4 obs] Direction to goal
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

            // [3 obs] Velocidade
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1 obs] Is airborne?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [16 obs] Raycasts
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
                    sensor.AddObservation(1f);
                    sensor.AddObservation(0f);
                }
            }

            // [1 obs] Ground height
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            // [1 obs] Jump button
            sensor.AddObservation(jumpPressed ? 1f : 0f);

            // [1 obs] Normalized time
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Controles (funciona igual em ambos os modos)
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );
            jumpPressed = actions.DiscreteActions[0] == 1;

            // Camera points toward goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ===== REWARD E REGISTRO =====
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();
            float distanceDelta = previousDistanceToGoal - currentDistance;

            // Default rewards
            AddReward(-0.01f);
            
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 1.0f);
            }

            if (currentDistance < bestDistanceToGoal - 2.0f)
            {
                AddReward(2.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // ===== DATA RECORDING (Recording mode) =====
            if (currentMode == HybridMode.Recording)
            {
                RecordTransition(actions, GetCumulativeReward(), false);
            }

            // Termination conditions - skip during grace period
            if (Time.time - lastSpawnTime > SPAWN_GRACE_PERIOD)
            {
                if (currentPos.y < startPosition.y - 3.0f)
                {
                    AddReward(-5.0f);
                    
                    if (currentMode == HybridMode.Recording)
                    {
                        RecordTransition(actions, GetCumulativeReward(), true);
                        SaveEpisodeData(false);
                    }
                    
                    EndEpisode();
                    return;
                }
            }
            
            if (currentPos.y < startPosition.y - 10f)
            {
                // Hard fail - fell way too far, end episode even during grace period
                AddReward(-10.0f);
                EndEpisode();
                return;
            }

            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f);
                
                if (currentMode == HybridMode.Recording)
                {
                    RecordTransition(actions, GetCumulativeReward(), true);
                    SaveEpisodeData(false);
                }
                
                EndEpisode();
                return;
            }
        }

        private void RecordTransition(ActionBuffers actions, float reward, bool done)
        {
            Vector3 position = transform.position;
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            
            var transition = new HybridTransition
            {
                position = position,
                velocity = velocity,
                reward = reward,
                done = done,
                timestamp = Time.time,
                actions = new float[] { joystickInput.x, joystickInput.y, jumpPressed ? 1f : 0f }
            };
            
            currentEpisodeData.Add(transition);
        }

        private void SaveEpisodeData(bool success)
        {
            if (dataRecorder == null) return;
            
            if (onlyRecordSuccesses && !success)
            {
                Debug.Log("[MarioHybrid] Failed episode discarded (onlyRecordSuccesses=true)");
                return;
            }
            
            dataRecorder.SaveEpisode(currentEpisodeData.ToArray(), success);
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(50f);
                
                if (currentMode == HybridMode.Recording)
                {
                    SaveEpisodeData(true);
                }
                
                EndEpisode();
            }
        }

        private float GetDistanceToGoal()
        {
            if (targetGoal == null) return float.MaxValue;
            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        public void SetEnvironment(HybridParkourEnvironment env)
        {
            environment = env;
        }

        public void SetTargetGoal(Transform goal)
        {
            targetGoal = goal;
        }

        public void SetDataRecorder(HybridDataRecorder recorder)
        {
            dataRecorder = recorder;
        }
        
        private System.Collections.IEnumerator TeleportAfterActivation(Vector3 position)
        {
            // Wait for activation to complete
            yield return new WaitForFixedUpdate();
            yield return null;
            
            if (marioComponent != null && marioComponent.isActiveAndEnabled)
            {
                marioComponent.Teleport(position);
                Debug.Log($"[MarioHybrid] Teleported to {position}");
            }
            
            // Validate after a few frames
            yield return new WaitForFixedUpdate();
            yield return new WaitForFixedUpdate();
            
            if (transform.position.y < position.y - 2f)
            {
                Debug.LogWarning($"[MarioHybrid] Mario fell after spawn! Repositioning...");
                transform.position = position;
                if (marioComponent != null && marioComponent.isActiveAndEnabled)
                {
                    marioComponent.Teleport(position);
                }
            }
        }
    }
}

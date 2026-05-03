using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL
{
    /// <summary>
    /// Dreamer-optimized Mario agent.
    /// Designed for world-model based reinforcement learning with enhanced observations.
    /// </summary>
    public class MarioDreamerAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;

        [Header("Team")]
        public int teamId = 0;
        public Color teamColor = Color.magenta;

        [Header("Environment")]
        [SerializeField] private DreamerParkourEnvironment dreamerEnv;
        [SerializeField] private Transform targetGoal;

        [Header("Dreamer Observations")]
        [SerializeField] private int raycastCount = 12;
        [SerializeField] private float raycastDistance = 15f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        [Header("Visual Observations")]
        [HideInInspector] public bool useVisualObservations = false;
        [HideInInspector] public Vector2Int cameraResolution = new Vector2Int(64, 64);

        private Vector3 startPosition;
        private Vector3 previousPosition;
        private float initialDistanceToGoal;
        private float previousDistanceToGoal;
        private float bestDistanceToGoal;
        private float episodeTime;
        // bestCompletionTime removido - não utilizado
        private bool hasFinished = false;
        private int ranking = 0;

        private const float MAX_EPISODE_TIME = 60f;
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        // Fallback detection
        private bool actionReceivedThisEpisode = false;
        private float timeSinceEpisodeStart = 0f;
        private const float TRAINER_GRACE_PERIOD = 1.0f;
        private bool usingFallbackActions = false;
        private float randomActionTimer = 0f;
        private const float RANDOM_ACTION_INTERVAL = 0.3f;

        // References to other agents
        private List<MarioDreamerAgent> rivals = new List<MarioDreamerAgent>();

        // Dreamer-specific tracking
        private float[] lastActions;
        private Vector3 lastVelocity;
        private float cumulativeReward = 0f;

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
            lastActions = new float[5];
            
            Debug.Log($"[MarioDreamerAgent {teamId}] Awake - Position: {transform.position}");
        }

        new void OnEnable()
        {
            base.OnEnable();
            // Verificar se BehaviorParameters está correto
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp != null)
            {
                Debug.Log($"[MarioDreamerAgent {teamId}] BehaviorType: {bp.BehaviorType}, Name: {bp.BehaviorName}");
                
                // Garantir que está como Default (não HeuristicOnly)
                if (bp.BehaviorType == Unity.MLAgents.Policies.BehaviorType.HeuristicOnly)
                {
                    Debug.LogWarning($"[MarioDreamerAgent {teamId}] WARNING: BehaviorType is HeuristicOnly! Changing to Default.");
                    bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
                }
            }
            else
            {
                Debug.LogError($"[MarioDreamerAgent {teamId}] ERROR: No BehaviorParameters found!");
            }
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp != null)
            {
                if (useVisualObservations)
                {
                    bp.BrainParameters.VectorObservationSize = 0;
                }
                else
                {
                    bp.BrainParameters.VectorObservationSize = 42;
                }
            }
        }

        public override void OnEpisodeBegin()
        {
            ResetInputs();
            hasFinished = false;
            ranking = 0;
            actionReceivedThisEpisode = false;
            timeSinceEpisodeStart = 0f;
            usingFallbackActions = false;
            randomActionTimer = 0f;
            cumulativeReward = 0f;

            if (dreamerEnv != null)
            {
                dreamerEnv.RespawnAgent(this);
                startPosition = dreamerEnv.GetCurrentSpawnPoint(this);
            }
            else
            {
                transform.position = startPosition;
            }

            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;

            // Reset last actions
            for (int i = 0; i < lastActions.Length; i++)
                lastActions[i] = 0f;
            lastVelocity = Vector3.zero;
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
        }

        public void SetRivals(List<MarioDreamerAgent> allAgents)
        {
            rivals = new List<MarioDreamerAgent>();
            foreach (var a in allAgents)
            {
                if (a != this)
                    rivals.Add(a);
            }
        }

        // ===== DREAMER-OPTIMIZED OBSERVATIONS: 42 obs =====
        // Enhanced for world model learning with temporal and spatial features
        public override void CollectObservations(VectorSensor sensor)
        {
            if (useVisualObservations)
            {
                CollectVisualObservations();
                return;
            }

            Vector3 position = transform.position;
            Vector3 envOffset = dreamerEnv != null ? dreamerEnv.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Mario position (normalized)
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

            // [3 obs] Mario velocity (current)
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [3 obs] Last velocity (temporal feature for world model)
            sensor.AddObservation(Mathf.Clamp(lastVelocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lastVelocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(lastVelocity.z / 10f, -1f, 1f));

            // [1 obs] Is airborne?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [1 obs] Is moving?
            sensor.AddObservation(velocity.magnitude > 0.1f ? 1f : 0f);

            // [24 obs] Enhanced raycasts (12 rays, 2 values each: distance + height diff)
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

            // [1 obs] Ground height below
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            // [2 obs] Temporal features
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);
            sensor.AddObservation(GetCurrentRanking());

            // [5 obs] Last actions (for temporal consistency)
            for (int i = 0; i < lastActions.Length; i++)
            {
                sensor.AddObservation(lastActions[i]);
            }

            // Total: 3 + 4 + 3 + 3 + 1 + 1 + 24 + 1 + 2 + 5 = 47 (adjusted to 42)
            // Adjusted version: 3 + 4 + 3 + 3 + 1 + 1 + 16 + 1 + 2 + 5 = 39 + 3 padding = 42
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);

            lastVelocity = velocity;
        }

        private void CollectVisualObservations()
        {
            // Visual observations are handled by the render texture
            // The Python side will capture the camera feed
            if (dreamerEnv != null && targetGoal != null)
            {
                Vector3 toGoal = targetGoal.position - transform.position;
                toGoal.y = 0;
                if (toGoal.sqrMagnitude > 0.01f)
                {
                    dreamerEnv.UpdateObservationCamera(transform.position, toGoal.normalized);
                }
            }
        }

        // lastActionLogTime e firstActionValue removidos - não utilizados
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            if (hasFinished) 
            {
                return;
            }
            
            // Marcar que recebemos ação do Python
            actionReceivedThisEpisode = true;
            
            if (usingFallbackActions)
            {
                usingFallbackActions = false;
            }

            // Continuous actions: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Discrete actions: [0] Jump, [1] Kick/Punch, [2] Stomp
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = actions.DiscreteActions[1] == 1;
            stompPressed = actions.DiscreteActions[2] == 1;
            
            // Debug: log first few actions
            if (Time.time < 30f && teamId == 0)
            {
                Debug.Log($"[MarioDreamerAgent {teamId}] OnActionReceived: Joystick={joystickInput}, Jump={jumpPressed}, Kick={kickPressed}, Stomp={stompPressed}");
            }
            
            // Store last actions for observations
            lastActions[0] = joystickInput.x;
            lastActions[1] = joystickInput.y;
            lastActions[2] = jumpPressed ? 1f : 0f;
            lastActions[3] = kickPressed ? 1f : 0f;
            lastActions[4] = stompPressed ? 1f : 0f;

            // Camera points toward goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== DREAMER-OPTIMIZED REWARDS ==========
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // -- Small step penalty to encourage efficiency --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = -0.002f;
            AddReward(stepPenalty);
            cumulativeReward += stepPenalty;

            // -- Progress reward (shaped) --
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                float progressReward = distanceDelta * 2.5f;
                AddReward(progressReward);
                cumulativeReward += progressReward;
            }
            else if (distanceDelta < -0.01f)
            {
                float regressPenalty = distanceDelta * 0.5f;
                AddReward(regressPenalty);
                cumulativeReward += regressPenalty;
            }

            // -- Distance milestone --
            if (currentDistance < bestDistanceToGoal - 1.0f)
            {
                AddReward(3.0f);
                cumulativeReward += 3.0f;
                bestDistanceToGoal = currentDistance;
            }

            // -- Speed bonus (Dreamer encourages smooth trajectories) --
            float speed = moveDelta / Time.fixedDeltaTime;
            if (speed > 2f && distanceDelta > 0)
            {
                float speedBonus = 0.01f * speed;
                AddReward(speedBonus);
                cumulativeReward += speedBonus;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");
            continuousActions[1] = Input.GetAxis("Vertical");

            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = Input.GetButton("Jump") ? 1 : 0;
            discreteActions[1] = Input.GetMouseButton(0) ? 1 : 0;
            discreteActions[2] = Input.GetKey(KeyCode.LeftShift) ? 1 : 0;
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
            if (hasFinished) return;

            timeSinceEpisodeStart += Time.fixedDeltaTime;
            episodeTime += Time.fixedDeltaTime;

            // Log inicial para debug (apenas nos primeiros segundos)
            if (timeSinceEpisodeStart < 3f && Time.frameCount % 30 == 0)
            {
                Debug.Log($"[MarioDreamerAgent {teamId}] t={timeSinceEpisodeStart:F1}s | " +
                          $"ActionReceived: {actionReceivedThisEpisode} | " +
                          $"Fallback: {usingFallbackActions} | " +
                          $"Joystick: {joystickInput} | " +
                          $"Jump: {jumpPressed} | " +
                          $"Pos: {transform.position}"
                );
            }

            // Death by fall
            Vector3 currentPos = transform.position;
            if (currentPos.y < -5f)
            {
                AddReward(-15f);
                cumulativeReward -= 15f;
                EndEpisode();
                return;
            }

            // Timeout
            if (episodeTime > MAX_EPISODE_TIME)
            {
                AddReward(-5f);
                cumulativeReward -= 5f;
                EndEpisode();
                return;
            }

            // Fallback actions if trainer not connected
            if (!actionReceivedThisEpisode && timeSinceEpisodeStart > TRAINER_GRACE_PERIOD)
            {
                if (!usingFallbackActions)
                {
                    usingFallbackActions = true;
                    Debug.Log($"[MarioDreamerAgent {teamId}] Ativando fallback actions!");
                }

                randomActionTimer += Time.fixedDeltaTime;
                if (randomActionTimer >= RANDOM_ACTION_INTERVAL)
                {
                    randomActionTimer = 0f;
                    joystickInput = new Vector2(
                        Random.Range(-0.5f, 0.5f),
                        Random.Range(0.2f, 1f)
                    );
                    jumpPressed = Random.value > 0.7f;
                }
            }
        }

        void OnTriggerEnter(Collider other)
        {
            if (hasFinished) return;

            if (other.CompareTag("Goal"))
            {
                hasFinished = true;
                
                float completionTime = episodeTime;
                float timeBonus = Mathf.Max(0, (MAX_EPISODE_TIME - completionTime) * 0.5f);
                
                AddReward(50f + timeBonus);
                cumulativeReward += 50f + timeBonus;

                if (dreamerEnv != null)
                {
                    ranking = dreamerEnv.RegisterFinish(this);
                }

                EndEpisode();
            }
        }

        void OnCollisionEnter(Collision collision)
        {
            if (hasFinished) return;

            // Mario vs Mario collision in competitive mode
            MarioDreamerAgent otherMario = collision.gameObject.GetComponent<MarioDreamerAgent>();
            if (otherMario != null)
            {
                Vector3 impact = collision.relativeVelocity;
                if (impact.y > 3f && transform.position.y > collision.transform.position.y + 0.5f)
                {
                    AddReward(2f);
                    cumulativeReward += 2f;
                }
            }
        }

        public void SetGoal(Transform goal)
        {
            targetGoal = goal;
        }

        public void SetDreamerEnv(DreamerParkourEnvironment env)
        {
            dreamerEnv = env;
        }

        public float GetDistanceToGoal()
        {
            if (targetGoal == null) return 999f;
            return Vector3.Distance(transform.position, targetGoal.position);
        }

        private float GetCurrentRanking()
        {
            if (rivals.Count == 0) return 0.5f;
            
            int agentsAhead = 0;
            float myDist = GetDistanceToGoal();
            
            foreach (var rival in rivals)
            {
                if (rival != null && rival.gameObject.activeInHierarchy && !rival.HasFinished)
                {
                    if (rival.GetDistanceToGoal() < myDist)
                        agentsAhead++;
                }
            }
            
            return 1f - (agentsAhead / (float)rivals.Count);
        }

        public void OnRivalFinished(MarioDreamerAgent rival)
        {
            if (hasFinished) return;
            
            // Encouragement to finish faster
            AddReward(-1f);
            cumulativeReward -= 1f;
        }

        public bool HasFinished => hasFinished;
        public int Ranking => ranking;
        public float EpisodeTime => episodeTime;
        public float CumulativeReward => cumulativeReward;
    }
}

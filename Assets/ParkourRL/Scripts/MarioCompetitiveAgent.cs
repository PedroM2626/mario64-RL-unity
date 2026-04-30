using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL
{
    /// <summary>
    /// Competitive Mario agent: competes against other Marios on the same map.
    /// Can punch (Kick/B), stomp (Stomp/Z) and run over other agents.
    /// Rewards based on relative position, speed and combat.
    /// </summary>
    public class MarioCompetitiveAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;

        [Header("Team")]
        public int teamId = 0;
        public Color teamColor = Color.red;

        [Header("Environment")]
        [SerializeField] private CompetitiveParkourEnvironment competitiveEnv;
        [SerializeField] private Transform targetGoal;

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
        private float bestDistanceToGoal;
        private float episodeTime;
        private float bestCompletionTime;
        private bool hasFinished = false;
        private int ranking = 0; // 0 = not finished, 1 = primeiro, etc.

        private const float MAX_EPISODE_TIME = 45f; // More time for competition
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        // Fallback: detecta se OnActionReceived nunca foi chamado (sem trainer)
        private bool actionReceivedThisEpisode = false;
        private float timeSinceEpisodeStart = 0f;
        private const float TRAINER_GRACE_PERIOD = 1.0f; // segundos para esperar trainer
        private bool usingFallbackActions = false;
        private float randomActionTimer = 0f;
        private const float RANDOM_ACTION_INTERVAL = 0.3f;

        // Reference to other agents in the same environment
        private List<MarioCompetitiveAgent> rivals = new List<MarioCompetitiveAgent>();

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
            bestCompletionTime = MAX_EPISODE_TIME;
        }
        


        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            
            // Ensure BehaviorParameters has the correct observation size (42)
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp != null && bp.BrainParameters.VectorObservationSize != 42)
            {
                bp.BrainParameters.VectorObservationSize = 42;
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

            if (competitiveEnv != null)
            {
                competitiveEnv.RespawnAgent(this);
                startPosition = competitiveEnv.GetCurrentSpawnPoint(this);
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
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
        }

        public void SetRivals(List<MarioCompetitiveAgent> allAgents)
        {
            rivals = new List<MarioCompetitiveAgent>();
            foreach (var a in allAgents)
            {
                if (a != this)
                    rivals.Add(a);
            }
        }

        // ===== OBSERVATIONS: 30 + 12 (rivals) = 42 obs =====
        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = competitiveEnv != null ? competitiveEnv.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Mario position (normalized and relative)
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

            // [16 obs] Raycasts to detect terrain/obstacles/rivals
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

            // [1 obs] Normalized time
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);

            // [1 obs] Relative ranking position (0=last, 1=first)
            float myRank = GetCurrentRanking();
            sensor.AddObservation(myRank);

            // [12 obs] Information about up to 3 closest rivals (4 obs each)
            // For each rival: [directionX, directionZ, distance, progressDifference]
            List<MarioCompetitiveAgent> sortedRivals = new List<MarioCompetitiveAgent>(rivals);
            sortedRivals.Sort((a, b) => 
                Vector3.Distance(position, a.transform.position).CompareTo(
                    Vector3.Distance(position, b.transform.position)));

            for (int i = 0; i < 3; i++)
            {
                if (i < sortedRivals.Count && sortedRivals[i] != null && sortedRivals[i].gameObject.activeInHierarchy)
                {
                    Vector3 toRival = sortedRivals[i].transform.position - position;
                    float rivalDist = toRival.magnitude;
                    sensor.AddObservation(toRival.x / 25f);
                    sensor.AddObservation(toRival.z / 25f);
                    sensor.AddObservation(Mathf.Clamp(rivalDist / 20f, 0, 1));
                    // Relative progress: positive = rival is closer to goal than me
                    float rivalGoalDist = sortedRivals[i].GetDistanceToGoal();
                    float progressDiff = (GetDistanceToGoal() - rivalGoalDist) / Mathf.Max(initialDistanceToGoal, 0.1f);
                    sensor.AddObservation(Mathf.Clamp(progressDiff, -1f, 1f));
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(1f); // Longe
                    sensor.AddObservation(0f);
                }
            }

            // Total: 3 + 4 + 3 + 1 + 16 + 1 + 1 + 1 + 12 = 42
        }

        private float lastActionLogTime = 0f;
        private bool firstActionReceived = false;
        
        public override void OnActionReceived(ActionBuffers actions)
        {
            if (hasFinished) 
            {
                return;
            }
            actionReceivedThisEpisode = true;
            
            // If was using fallback, deactivate
            if (usingFallbackActions)
            {
                usingFallbackActions = false;
            }
            
            // Log on first action received
            if (!firstActionReceived)
            {
                firstActionReceived = true;
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
            
            // Log every 2 seconds for debug
            if (Time.time - lastActionLogTime > 2f)
            {
                lastActionLogTime = Time.time;
            }

            // Camera points toward goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== COMPETITIVE REWARDS ==========
            // NOTE: episodeTime is now incremented in FixedUpdate for accuracy
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // -- Inactivity penalty --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.015f : -0.003f;
            AddReward(stepPenalty);

            // -- Progress reward --
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 3.0f);
            }
            else if (distanceDelta < -0.01f)
            {
                AddReward(distanceDelta * 0.3f);
            }

            // -- Distance milestone --
            if (currentDistance < bestDistanceToGoal - 1.0f)
            {
                AddReward(5.0f);
                bestDistanceToGoal = currentDistance;
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
            discreteActions[1] = Input.GetMouseButton(0) ? 1 : 0;   // Kick
            discreteActions[2] = Input.GetKey(KeyCode.LeftShift) ? 1 : 0; // Stomp
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

            // Episode timer - runs every FixedUpdate for accuracy
            timeSinceEpisodeStart += Time.fixedDeltaTime;
            episodeTime += Time.fixedDeltaTime;

            // Death by fall (checked every physics frame)
            Vector3 currentPos = transform.position;
            if (currentPos.y < startPosition.y - 3.0f)
            {
                float currentDistance = GetDistanceToGoal();
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-2.0f + progressRatio * 1.0f);
                EndEpisode();
                return;
            }

            // Timeout (checked every physics frame)
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float currentDistance = GetDistanceToGoal();
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-8.0f + progressRatio * 3.0f);
                EndEpisode();
                return;
            }
        }

        private void GenerateRandomActions()
        {
            // Generate random actions for exploration
            joystickInput = new Vector2(
                Random.Range(-1f, 1f),
                Random.Range(-1f, 1f)
            );

            // 30% jump chance, 10% kick, 10% stomp
            jumpPressed = Random.value < 0.3f;
            kickPressed = Random.value < 0.1f;
            stompPressed = Random.value < 0.1f;

            // Camera points toward goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }
        }

        private void ApplyFallbackRewards()
        {
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // -- Inactivity penalty --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.015f : -0.003f;
            AddReward(stepPenalty);

            // -- Progress reward --
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 3.0f);
            }
            else if (distanceDelta < -0.01f)
            {
                AddReward(distanceDelta * 0.3f);
            }

            // -- Distance milestone --
            if (currentDistance < bestDistanceToGoal - 1.0f)
            {
                AddReward(5.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;
        }

        // ===== PUBLIC METHODS =====

        public float GetDistanceToGoal()
        {
            if (targetGoal == null) return float.MaxValue;
            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        public float GetCurrentRanking()
        {
            if (rivals.Count == 0) return 1f;
            int aheadCount = 0;
            float myDist = GetDistanceToGoal();
            foreach (var rival in rivals)
            {
                if (rival != null && rival.gameObject.activeInHierarchy)
                {
                    if (myDist < rival.GetDistanceToGoal())
                        aheadCount++;
                }
            }
            return (float)aheadCount / Mathf.Max(rivals.Count, 1);
        }

        public void OnRivalFinished(MarioCompetitiveAgent rival)
        {
            // Rival finished first -- no penalty in simultaneous mode
        }

        public void SetGoal(Transform g) { targetGoal = g; }
        public void SetCompetitiveEnv(CompetitiveParkourEnvironment env) { competitiveEnv = env; }
        public bool HasFinished => hasFinished;
        public float EpisodeTimeElapsed => episodeTime;

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal") && !hasFinished)
            {
                hasFinished = true;

                // Time bonus
                float timeRemaining = Mathf.Max(0, MAX_EPISODE_TIME - episodeTime);
                float timeBonus = timeRemaining * 2.0f;

                // Personal record bonus
                float recordBonus = 0f;
                if (episodeTime < bestCompletionTime)
                {
                    recordBonus = (bestCompletionTime - episodeTime) * 3.0f;
                    bestCompletionTime = episodeTime;
                }

                // Ranking bonus (We only register the win, no massive bonuses to avoid direct competition for points)
                if (competitiveEnv != null)
                {
                    ranking = competitiveEnv.RegisterFinish(this);
                }

                AddReward(50f + timeBonus + recordBonus);
                EndEpisode();
            }
        }

        void OnDrawGizmosSelected()
        {
            Gizmos.color = teamColor;
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

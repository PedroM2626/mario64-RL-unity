using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace ParkourRL
{
    public enum ChaseRole
    {
        Pursuer = 0,
        Fugitive = 1
    }

    /// <summary>
    /// Agente para treino descentralizado 1v1: perseguidor vs fugitivo.
    /// </summary>
    public class ChaseAgent : Agent
    {
        private const float CombatCooldown = 0.3f;
        private const float MaxEpisodeTime = 120f;
        private const float ButtonThreshold = 0.5f;

        [Header("Role")]
        public ChaseRole role = ChaseRole.Pursuer;
        public ChaseTrainingEnvironment chaseEnvironment;
        public ChaseAgent opponent;

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;
        [HideInInspector] public bool lastKickPressed;
        [HideInInspector] public bool lastStompPressed;

        private float episodeTime;
        private float lastKickTime;
        private float lastStompTime;
        private float previousOpponentDistance;
        private Vector3 previousPosition;
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            previousPosition = transform.position;
        }

        public override void OnEpisodeBegin()
        {
            ResetInputs();
            episodeTime = 0f;
            previousPosition = transform.position;
            previousOpponentDistance = GetOpponentDistance();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = chaseEnvironment != null ? chaseEnvironment.transform.position : Vector3.zero;
            Vector3 localPos = position - envOffset;

            // [3] Posição local
            sensor.AddObservation(localPos.x / 50f);
            sensor.AddObservation(localPos.y / 10f);
            sensor.AddObservation(localPos.z / 50f);

            // [3] Velocidade local
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1] Grounded
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 1f : 0f);

            // [3 + 1 + 3] Info do oponente
            if (opponent != null && opponent.gameObject.activeInHierarchy)
            {
                Vector3 toOpponent = opponent.transform.position - position;
                Vector3 oppVelocity = (opponent.transform.position - opponent.previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
                float dist = toOpponent.magnitude;

                sensor.AddObservation(toOpponent.x / 50f);
                sensor.AddObservation(toOpponent.y / 10f);
                sensor.AddObservation(toOpponent.z / 50f);
                sensor.AddObservation(Mathf.Clamp(dist / 50f, 0f, 1f));

                sensor.AddObservation(Mathf.Clamp(oppVelocity.x / 10f, -1f, 1f));
                sensor.AddObservation(Mathf.Clamp(oppVelocity.y / 10f, -1f, 1f));
                sensor.AddObservation(Mathf.Clamp(oppVelocity.z / 10f, -1f, 1f));
            }
            else
            {
                for (int i = 0; i < 7; i++)
                    sensor.AddObservation(0f);
            }

            // [1] Tempo normalizado
            sensor.AddObservation(Mathf.Clamp01(episodeTime / MaxEpisodeTime));

            // [16] Raycasts
            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;

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

            // Total: 31
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            jumpPressed = actions.ContinuousActions[2] > ButtonThreshold;
            lastKickPressed = actions.ContinuousActions[3] > ButtonThreshold && (Time.time - lastKickTime > CombatCooldown);
            lastStompPressed = actions.ContinuousActions[4] > ButtonThreshold && (Time.time - lastStompTime > CombatCooldown);

            if (lastKickPressed)
                lastKickTime = Time.time;
            if (lastStompPressed)
                lastStompTime = Time.time;

            kickPressed = lastKickPressed;
            stompPressed = lastStompPressed;

            if (opponent != null)
            {
                Vector3 toOpponent = opponent.transform.position - transform.position;
                toOpponent.y = 0f;
                cameraLookDirection = toOpponent.sqrMagnitude > 0.001f ? toOpponent.normalized : Vector3.forward;
            }
            else
            {
                cameraLookDirection = Vector3.forward;
            }

            episodeTime += Time.fixedDeltaTime;

            // Recompensas densas por distância (descentralizado)
            float dist = GetOpponentDistance();
            float delta = previousOpponentDistance - dist;

            if (role == ChaseRole.Pursuer)
            {
                AddReward(-0.001f);
                AddReward(delta * 0.02f);
                if (dist < 3f)
                    AddReward(0.005f);
            }
            else
            {
                AddReward(0.0015f);
                AddReward((-delta) * 0.02f);
                if (dist > 8f)
                    AddReward(0.004f);
            }

            if (chaseEnvironment != null && chaseEnvironment.IsOutOfBounds(transform.position))
            {
                AddReward(-0.05f);
            }

            if (episodeTime >= MaxEpisodeTime)
            {
                chaseEnvironment?.ResolveTimeout();
                return;
            }

            previousOpponentDistance = dist;
            previousPosition = transform.position;
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");
            continuousActions[1] = Input.GetAxis("Vertical");
            continuousActions[2] = Input.GetButton("Jump") ? 1f : 0f;
            continuousActions[3] = Input.GetMouseButton(0) ? 1f : 0f;      // soco/chute (B)
            continuousActions[4] = Input.GetKey(KeyCode.LeftShift) ? 1f : 0f; // rasteira/stomp (Z)
        }

        public void OnSuccessfulHit(float reward)
        {
            AddReward(reward);
        }

        private float GetOpponentDistance()
        {
            if (opponent == null || !opponent.gameObject.activeInHierarchy)
                return 50f;
            return Vector3.Distance(transform.position, opponent.transform.position);
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            lastKickPressed = false;
            lastStompPressed = false;
            cameraLookDirection = Vector3.forward;
            lastKickTime = 0f;
            lastStompTime = 0f;
        }
    }
}

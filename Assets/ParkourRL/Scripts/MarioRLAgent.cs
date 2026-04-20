using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

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

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;
        [SerializeField] private LayerMask terrainLayer;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        private Vector3 startPosition;
        private float previousDistanceToGoal;
        private float episodeTime;
        private const float MAX_EPISODE_TIME = 60f;

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
            if (environment == null)
                environment = FindObjectOfType<ParkourEnvironment>();

            startPosition = transform.position;
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
            }
            else
            {
                transform.position = startPosition;
            }

            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;

            previousDistanceToGoal = GetDistanceToGoal();
            episodeTime = 0f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Posição e velocidade do Mario
            Vector3 position = transform.position;
            sensor.AddObservation(position.x);
            sensor.AddObservation(position.y);
            sensor.AddObservation(position.z);

            // Posição relativa ao objetivo
            if (targetGoal != null)
            {
                Vector3 toGoal = targetGoal.position - position;
                sensor.AddObservation(toGoal.x);
                sensor.AddObservation(toGoal.y);
                sensor.AddObservation(toGoal.z);
                sensor.AddObservation(toGoal.magnitude);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }

            // Raycasts para detectar terreno/obstáculos
            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;

                if (Physics.Raycast(position + Vector3.up * 0.5f, direction, out RaycastHit hit, raycastDistance, terrainLayer))
                {
                    sensor.AddObservation(hit.distance / raycastDistance);
                    sensor.AddObservation(hit.point.y - position.y);
                }
                else
                {
                    sensor.AddObservation(1f);
                    sensor.AddObservation(0f);
                }
            }

            // Altura do chão abaixo do Mario
            if (Physics.Raycast(position + Vector3.up * 0.5f, Vector3.down, out RaycastHit groundHit, 20f, terrainLayer))
            {
                sensor.AddObservation(groundHit.distance);
            }
            else
            {
                sensor.AddObservation(20f);
            }

            // Tempo restante (normalizado)
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Ações contínuas: joystick X, joystick Y
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Ações discretas: Jump, Kick, Stomp
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = actions.DiscreteActions[1] == 1;
            stompPressed = actions.DiscreteActions[2] == 1;

            // Atualizar direção da câmera (apontar para o objetivo)
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
            }

            // Recompensas
            episodeTime += Time.fixedDeltaTime;

            // Recompensa por se aproximar do objetivo
            float currentDistance = GetDistanceToGoal();
            float distanceDelta = previousDistanceToGoal - currentDistance;
            AddReward(distanceDelta * 0.1f);
            previousDistanceToGoal = currentDistance;

            // Penalidade por tempo
            AddReward(-0.001f);

            // Verificar tempo máximo
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                AddReward(-1f);
                EndEpisode();
            }

            // Verificar se caiu
            if (transform.position.y < -10f)
            {
                AddReward(-1f);
                EndEpisode();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");
            continuousActions[1] = Input.GetAxis("Vertical");

            var discreteActions = actionsOut.DiscreteActions;
            discreteActions[0] = Input.GetButton("Jump") ? 1 : 0;
            discreteActions[1] = Input.GetButton("Kick") ? 1 : 0;
            discreteActions[2] = Input.GetButton("Z") ? 1 : 0;
        }

        void Start()
        {
            // Garantir que o input provider está presente (feito em Start para evitar conflito com Awake)
            if (GetComponent<MarioInputProvider>() == null)
            {
                gameObject.AddComponent<MarioInputProvider>();
            }
        }

        private float GetDistanceToGoal()
        {
            if (targetGoal == null) return float.MaxValue;
            return Vector3.Distance(transform.position, targetGoal.position);
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(10f);
                EndEpisode();
            }
            else if (other.CompareTag("Checkpoint"))
            {
                Checkpoint checkpoint = other.GetComponent<Checkpoint>();
                if (checkpoint != null && !checkpoint.IsActivated)
                {
                    checkpoint.Activate();
                    AddReward(2f);
                    environment?.SetCheckpoint(checkpoint.transform.position);
                }
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

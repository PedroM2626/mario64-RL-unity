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

        private const float MAX_EPISODE_TIME = 30f; // Tempo generoso para explorar
        
        // Cache array for Raycasts to prevent ALLOC_TEMP_MAIN leakage
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();

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

            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;

        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Forcar VectorObservationSize correto
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null && behaviorParams.BrainParameters.VectorObservationSize != 30)
            {
                behaviorParams.BrainParameters.VectorObservationSize = 30;
            }

            Vector3 position = transform.position;

            // [3 obs] Posicao do Mario (normalizada)
            sensor.AddObservation(position.x / 25f);
            sensor.AddObservation(position.y / 10f);
            sensor.AddObservation(position.z / 25f);

            // [4 obs] Direcao e distancia ao objetivo
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

            // [3 obs] Velocidade do Mario
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1 obs] Esta no ar?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [16 obs] Raycasts para detectar terreno/obstaculos (sem layer mask)
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
                    sensor.AddObservation(1f); // Nada detectado = distancia maxima
                    sensor.AddObservation(0f);
                }
            }

            // [1 obs] Altura do chao abaixo
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f); // Sem chao = caindo
            }

            // [1 obs] Jump button ativo
            sensor.AddObservation(jumpPressed ? 1f : 0f);

            // [1 obs] Tempo normalizado
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);

            // Total: 3 + 4 + 3 + 1 + 16 + 1 + 1 + 1 = 30
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Acoes continuas: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Acao discreta: apenas Jump (sem kick/stomp que causavam ground-pound)
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = false;
            stompPressed = false;

            // Apontar camera para o goal (direcao do movimento)
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== RECOMPENSAS RIGIDAS ==========
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // 1. Penalidade constante por step (pressao de tempo severa)
            //    Com ~20s de episodio e DecisionPeriod=2, sao ~500 steps
            //    Total: 500 * -0.01 = -5.0 se ficar parado
            AddReward(-0.01f);

            // 2. Recompensa por progresso real (delta de distancia)
            //    So recompensa se REALMENTE se aproximou (delta > threshold)
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                // Recompensa proporcional ao progresso real
                AddReward(distanceDelta * 1.0f);
            }

            // 3. Bonus por marco de distancia (a cada 2 unidades mais perto)
            if (currentDistance < bestDistanceToGoal - 2.0f)
            {
                float improvement = bestDistanceToGoal - currentDistance;
                AddReward(2.0f); // Bonus grande por progresso significativo
                bestDistanceToGoal = currentDistance;
                Debug.Log($"[Mario] MARCO! Nova melhor distancia: {bestDistanceToGoal:F1} (delta: {improvement:F1})");
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // 4. Morte por queda -- zona de morte BEM baixa para dar liberdade total
            //    Mario e LIVRE para cair, explorar, errar
            if (currentPos.y < -50f)
            {
                AddReward(-3.0f);
                EndEpisode();
                return;
            }

            // 5. Timeout (penalidade alta)
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                // Penalidade proporcional: quem nao fez progresso leva mais
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f); // -5 se zero progresso, -2 se quase la
                EndEpisode();
                return;
            }

            // Log periodico
            if (StepCount % 500 == 0 && StepCount > 0)
            {
                Debug.Log($"[Mario] Step {StepCount}: Dist={currentDistance:F1}, Best={bestDistanceToGoal:F1}, " +
                          $"Reward={GetCumulativeReward():F2}, Pos={currentPos}, Y={currentPos.y:F2}");
            }
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
            // Deteccao de Mario preso
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
            // Distancia horizontal (XZ) para nao penalizar saltos verticais
            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                Debug.Log($"[Mario] ====== GOAL ATINGIDO! ====== Step: {StepCount}");
                AddReward(50f); // Recompensa massiva pelo goal
                EndEpisode();
            }
            else if (other.CompareTag("Checkpoint"))
            {
                Checkpoint checkpoint = other.GetComponent<Checkpoint>();
                if (checkpoint != null && !checkpoint.IsActivated)
                {
                    checkpoint.Activate();
                    AddReward(5f);
                    environment?.SetCheckpoint(checkpoint.transform.position);
                    Debug.Log($"[Mario] Checkpoint atingido!");
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

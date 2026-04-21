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
        private Vector3 previousPosition;
        private float previousDistanceToGoal;
        private float episodeTime;
        private float bestDistanceToGoal;
        private const float MAX_EPISODE_TIME = 30f; // Reduzido de 60s para episodios mais rapidos
        
        // Cache array for Raycasts to prevent ALLOC_TEMP_MAIN leakage
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

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
            bestDistanceToGoal = previousDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Verificar se VectorObservationSize esta correto no BehaviorParameters
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null && behaviorParams.BrainParameters.VectorObservationSize != 30)
            {
                behaviorParams.BrainParameters.VectorObservationSize = 30;
            }

            Vector3 position = transform.position;

            // [3 obs] Posicao do Mario (normalizada)
            sensor.AddObservation(position.x / 20f);
            sensor.AddObservation(position.y / 20f);
            sensor.AddObservation(position.z / 20f);

            // [4 obs] Posicao relativa ao objetivo
            if (targetGoal != null)
            {
                Vector3 toGoal = targetGoal.position - position;
                sensor.AddObservation(toGoal.x / 20f);
                sensor.AddObservation(toGoal.y / 20f);
                sensor.AddObservation(toGoal.z / 20f);
                sensor.AddObservation(toGoal.magnitude / 30f);
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }

            // [3 obs] Velocidade do Mario (calculada por delta de posicao)
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(velocity.x / 10f);
            sensor.AddObservation(velocity.y / 10f);
            sensor.AddObservation(velocity.z / 10f);

            // [1 obs] Esta no ar?
            int groundCheck = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f);
            sensor.AddObservation(groundCheck == 0 ? 1f : 0f);

            // [16 obs] Raycasts para detectar terreno/obstaculos
            // CORRECAO CRITICA: sem LayerMask (detecta TUDO) - antes usava terrainLayer que era 0 (nada)
            for (int i = 0; i < raycastCount; i++)
            {
                float angle = (360f / raycastCount) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;

                if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, direction, raycastHitsCache, raycastDistance) > 0)
                {
                    sensor.AddObservation(raycastHitsCache[0].distance / raycastDistance);
                    sensor.AddObservation(raycastHitsCache[0].point.y - position.y);
                }
                else
                {
                    sensor.AddObservation(1f);
                    sensor.AddObservation(0f);
                }
            }

            // [1 obs] Altura do chao abaixo do Mario (sem LayerMask)
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            // [1 obs] Acoes atuais (esta pulando?)
            sensor.AddObservation(jumpPressed ? 1f : 0f);

            // [1 obs] Tempo restante (normalizado)
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);

            // Total: 3 + 4 + 3 + 1 + 16 + 1 + 1 + 1 = 30 observacoes
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

            // Log de debug (a cada 300 steps para nao floodar)
            if (StepCount % 300 == 0)
            {
                Debug.Log($"[MarioRLAgent] Step {StepCount}: Dist={GetDistanceToGoal():F1}, Reward={GetCumulativeReward():F2}, Pos={transform.position}");
            }

            // Atualizar direcao da camera (apontar para o objetivo)
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
            }

            // === RECOMPENSAS ===
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // 1. Recompensa por se aproximar do objetivo (principal driver)
            float distanceDelta = previousDistanceToGoal - currentDistance;
            AddReward(distanceDelta * 0.2f);

            // 2. Bonus por novo recorde de proximidade (incentiva progresso)
            if (currentDistance < bestDistanceToGoal - 0.5f)
            {
                AddReward(0.5f);
                bestDistanceToGoal = currentDistance;
            }

            // 3. Recompensa por estar no ar E se movendo em direcao ao goal
            //    (incentiva pular em direcao a plataforma ao inves de ficar parado na borda)
            int groundHits = Physics.RaycastNonAlloc(currentPos, Vector3.down, raycastHitsCache, 0.3f);
            bool isAirborne = (groundHits == 0);
            if (isAirborne && distanceDelta > 0)
            {
                // Mario esta subindo E se aproximando do goal = muito bom!
                AddReward(0.1f);
            }

            // 4. Recompensa por velocidade horizontal (incentiva movimento, nao ficar parado)
            float horizontalSpeed = new Vector2(currentPos.x - previousPosition.x, currentPos.z - previousPosition.z).magnitude / Time.fixedDeltaTime;
            if (horizontalSpeed > 0.5f)
            {
                AddReward(0.001f); // Pequeno bonus por se mover
            }

            // 5. Penalidade por tempo (leve)
            AddReward(-0.0005f);

            // 6. Penalidade por ficar parado (anti-estagnacao)
            float movedDist = Vector3.Distance(currentPos, previousPosition);
            if (movedDist < 0.01f)
            {
                AddReward(-0.005f); // Penaliza ficar parado
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // Verificar tempo maximo
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                AddReward(-0.5f);
                EndEpisode();
            }

            // Verificar se caiu (penalidade reduzida para incentivar risco)
            if (currentPos.y < -10f)
            {
                AddReward(-0.3f); // Era -1.0, reduzido para o agente nao ter tanto medo de cair
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
            
            // Debug.Log("[MarioRLAgent] Inicializado - Target: " + (targetGoal != null ? targetGoal.name : "null"));
        }
        
        void FixedUpdate()
        {
            // O ML-Agents Academy gerencia o ciclo de decisões automaticamente
            // Verificar se Mario está preso (não se moveu nas últimas 500 steps)
            if (StepCount > 0 && StepCount % 500 == 0)
            {
                float moved = Vector3.Distance(transform.position, startPosition);
                if (moved < 1f)
                {
                    Debug.LogWarning($"[MarioRLAgent] Mario parece preso! Step {StepCount}, Pos: {transform.position}, Reward: {GetCumulativeReward():F2}");
                }
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

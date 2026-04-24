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

        [Header("Curriculum / Reward Shaping")]
        [Tooltip("Licao a partir da qual a penalidade por velocidade fica mais agressiva.")]
        [SerializeField] private int aggressiveShapingStartsAtLesson = 2;
        [Tooltip("Penalidade base por passo nas primeiras licoes.")]
        [SerializeField] private float earlyLessonExistentialPenalty = -0.01f;
        [Tooltip("Penalidade base por passo nas licoes mais avancadas.")]
        [SerializeField] private float lateLessonExistentialPenalty = -0.02f;
        [Tooltip("Bonus por progresso em direcao ao goal nas primeiras licoes.")]
        [SerializeField] private float earlyLessonProgressReward = 0.12f;
        [Tooltip("Bonus por progresso em direcao ao goal nas licoes avancadas.")]
        [SerializeField] private float lateLessonProgressReward = 0.10f;
        [Tooltip("Bonus por progresso nas fases 5-8 (repeticao com randomizacao progressiva e full map).")]
        [SerializeField] private float advancedPhaseProgressReward = 0.15f;
        [Tooltip("Penalidade por andar para tras nas primeiras licoes.")]
        [SerializeField] private float earlyLessonBackwardPenalty = 0.01f;
        [Tooltip("Penalidade por andar para tras nas licoes avancadas.")]
        [SerializeField] private float lateLessonBackwardPenalty = 0.01f;
        [Tooltip("Recompensa por concluir full map (fases 7+).")]
        [SerializeField] private float fullMapCompletionBonus = 50.0f;
        [Tooltip("Pequeno bonus por tentar saltar e ganhar altura, para evitar congelamento nas plataformas.")]
        [SerializeField] private float jumpAttemptReward = 0.05f;

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
        private float bestDistanceWhileGrounded; // Rastreia o progresso SEGURO (quando ele pousa em uma plataforma)
        private float bestCompletionTime; // Melhor tempo de conclusao entre episodios
        private bool episodeResultReported;

        private const float MAX_EPISODE_TIME = 30f; // Estilo old: mais episodios por hora para convergir mais rapido
        
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

            bestCompletionTime = MAX_EPISODE_TIME; // Inicializa com o pior tempo possivel
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
                // Apos o reset (que teleporta o Mario), atualizamos a startPosition para refletir o spawn real
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
            bestDistanceWhileGrounded = initialDistanceToGoal; // Inicia a distancia segura
            previousPosition = transform.position;
            episodeTime = 0f;
            episodeResultReported = false;

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
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Posicao do Mario (normalizada e RELATIVA ao ambiente)
            sensor.AddObservation(localPosition.x / 25f);
            sensor.AddObservation(localPosition.y / 10f);
            sensor.AddObservation(localPosition.z / 25f);

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

            // Acao discreta: Jump + Kick (para o modo competitivo tambem)
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

            // ========== REWARD ESTILO OLD (COMPLETAR PARKOUR DE VERDADE) ==========
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();
            float distanceDelta = previousDistanceToGoal - currentDistance;

            // Pressao temporal clara para evitar ficar parado em uma plataforma.
            AddReward(-0.01f);

            // Recompensa densa por progresso real na direcao do goal.
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 1.0f);
            }

            // Marco intermediario de progresso para estabilizar exploracao.
            if (currentDistance < bestDistanceToGoal - 2.0f)
            {
                AddReward(2.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // Queda: penalidade forte, mas sem impedir exploracao por completo.
            if (currentPos.y < startPosition.y - 3.0f)
            {
                AddReward(-5.0f);
                ReportEpisodeResult(false);
                EndEpisode();
                return;
            }

            // Timeout proporcional ao progresso (estilo old): pune travamento, recompensa tentativa real.
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f);
                ReportEpisodeResult(false);
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

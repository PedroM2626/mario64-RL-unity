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
        private float bestCompletionTime; // Melhor tempo de conclusao entre episodios

        private const float MAX_EPISODE_TIME = 30f;
        
        // Cache array for Raycasts to prevent ALLOC_TEMP_MAIN leakage
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

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

            // ========== SISTEMA DE RECOMPENSAS CORRIGIDO ==========
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // -- PENALIDADE POR INATIVIDADE (escalonada) --
            // Se o Mario nao se moveu quase nada, penalidade DOBRADA.
            // Isso pune "pular no lugar" mais do que "correr e cair".
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.015f : -0.005f;
            AddReward(stepPenalty);

            // -- RECOMPENSA POR PROGRESSO --
            // Cada unidade de distancia que ele se aproxima do goal vale MUITO.
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                // Progresso escalado agressivamente
                AddReward(distanceDelta * 3.0f);
            }
            else if (distanceDelta < -0.01f)
            {
                // Penalidade LEVE por se afastar (menos que cair/timeout)
                AddReward(distanceDelta * 0.5f);
            }

            // -- BONUS POR MARCO DE DISTANCIA (a cada 1 unidade mais perto) --
            if (currentDistance < bestDistanceToGoal - 1.0f)
            {
                float improvement = bestDistanceToGoal - currentDistance;
                AddReward(5.0f); // Bonus grande por progresso significativo
                bestDistanceToGoal = currentDistance;
                Debug.Log($"[Mario] MARCO! Nova melhor distancia: {bestDistanceToGoal:F1} (delta: {improvement:F1})");
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // -- MORTE POR QUEDA --
            // Cair da plataforma e ruim, mas e MELHOR do que ficar parado.
            // Quem cai tentando ganha reset rapido para tentar de novo.
            if (currentPos.y < startPosition.y - 3.0f)
            {
                // Penalidade proporcional: se caiu com progresso, penalidade menor
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                float deathPenalty = -2.0f + progressRatio * 1.0f; // -2.0 se zero progresso, -1.0 se quase la
                AddReward(deathPenalty);
                EndEpisode();
                return;
            }

            // -- TIMEOUT (PIOR RESULTADO POSSIVEL) --
            // Ficar parado ate o fim e PIOR do que morrer tentando.
            // Isso forca o agente a arriscar pular em vez de ficar seguro.
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                float timeoutPenalty = -8.0f + progressRatio * 3.0f; // -8 se zero progresso, -5 se quase la
                AddReward(timeoutPenalty);
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
                // Bonus de tempo: quanto mais rapido, maior a recompensa.
                float timeRemaining = Mathf.Max(0, MAX_EPISODE_TIME - episodeTime);
                float timeBonus = timeRemaining * 2.0f; // ate +60 extras

                // Bonus por recorde pessoal (bateu seu melhor tempo?)
                float recordBonus = 0f;
                if (episodeTime < bestCompletionTime)
                {
                    recordBonus = (bestCompletionTime - episodeTime) * 3.0f;
                    bestCompletionTime = episodeTime;
                    Debug.Log($"[Mario] NOVO RECORDE! Tempo: {episodeTime:F1}s (anterior: {bestCompletionTime:F1}s, bonus: +{recordBonus:F1})");
                }

                Debug.Log($"[Mario] ====== GOAL! ====== Tempo: {episodeTime:F1}s | TimeBonus: +{timeBonus:F1} | RecordBonus: +{recordBonus:F1} | Step: {StepCount}");
                AddReward(50f + timeBonus + recordBonus);
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

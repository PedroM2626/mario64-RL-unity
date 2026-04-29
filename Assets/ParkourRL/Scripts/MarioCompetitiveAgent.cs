using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL
{
    /// <summary>
    /// Agente Mario competitivo: compete contra outros Marios no mesmo mapa.
    /// Pode socar (Kick/B), chutar (Stomp/Z) e atropelar outros agentes.
    /// Recompensas baseadas em posicao relativa, velocidade e combate.
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
        private int ranking = 0; // 0 = nao terminou, 1 = primeiro, etc.

        private const float MAX_EPISODE_TIME = 45f; // Mais tempo para competicao
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        // Fallback: detecta se OnActionReceived nunca foi chamado (sem trainer)
        private bool actionReceivedThisEpisode = false;
        private float timeSinceEpisodeStart = 0f;
        private const float TRAINER_GRACE_PERIOD = 1.0f; // segundos para esperar trainer
        private bool usingFallbackActions = false;
        private float randomActionTimer = 0f;
        private const float RANDOM_ACTION_INTERVAL = 0.3f;

        // Referencia a outros agentes no mesmo ambiente
        private List<MarioCompetitiveAgent> rivals = new List<MarioCompetitiveAgent>();

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
            bestCompletionTime = MAX_EPISODE_TIME;
        }
        
        new void OnEnable()
        {
            base.OnEnable();
            Debug.Log($"[{name}] OnEnable chamado - BehaviorParameters: {GetComponent<Unity.MLAgents.Policies.BehaviorParameters>() != null}");
        }
        
        new void OnDisable()
        {
            base.OnDisable();
            Debug.Log($"[{name}] OnDisable chamado");
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            
            // Garantir que o BehaviorParameters tenha o tamanho de observacao correto (42)
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp != null && bp.BrainParameters.VectorObservationSize != 42)
            {
                bp.BrainParameters.VectorObservationSize = 42;
                Debug.Log($"[{name}] VectorObservationSize corrigido para 42");
            }
            Debug.Log($"[{name}] Initialize concluido. BehaviorName: {(bp != null ? bp.BehaviorName : "null")}");
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

        // ===== OBSERVACOES: 30 + 12 (rivais) = 42 obs =====
        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = competitiveEnv != null ? competitiveEnv.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Posicao do Mario (normalizada e relativa)
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

            // [16 obs] Raycasts para detectar terreno/obstaculos/rivais
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

            // [1 obs] Altura do chao abaixo
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            // [1 obs] Tempo normalizado
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);

            // [1 obs] Posicao relativa no ranking (0=ultimo, 1=primeiro)
            float myRank = GetCurrentRanking();
            sensor.AddObservation(myRank);

            // [12 obs] Informacoes sobre ate 3 rivais mais proximos (4 obs cada)
            // Para cada rival: [direcaoX, direcaoZ, distancia, diferencaProgresso]
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
                    // Progresso relativo: positivo = rival esta mais perto do goal que eu
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
            if (hasFinished) return;
            
            actionReceivedThisEpisode = true;
            
            // Se estava usando fallback, desativar
            if (usingFallbackActions)
            {
                usingFallbackActions = false;
                Debug.Log($"[{name}] Trainer conectado! Desativando fallback.");
            }
            
            // Log na primeira vez que recebe acao
            if (!firstActionReceived)
            {
                Debug.Log($"[{name}] PRIMEIRA ACAO RECEBIDA! Continuous: [{actions.ContinuousActions[0]:F2}, {actions.ContinuousActions[1]:F2}], Discrete: [{actions.DiscreteActions[0]}, {actions.DiscreteActions[1]}, {actions.DiscreteActions[2]}]");
                firstActionReceived = true;
            }

            // Acoes continuas: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Acoes discretas: [0] Jump, [1] Kick/Punch, [2] Stomp
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = actions.DiscreteActions[1] == 1;
            stompPressed = actions.DiscreteActions[2] == 1;
            
            // Log a cada 2 segundos para debug
            if (Time.time - lastActionLogTime > 2f)
            {
                Debug.Log($"[ActionDebug] {name}: Joystick={joystickInput}, Jump={jumpPressed}, Kick={kickPressed}, Stomp={stompPressed}");
                lastActionLogTime = Time.time;
            }

            // Camera aponta para o goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== RECOMPENSAS COMPETITIVAS ==========
            // NOTA: episodeTime agora e incrementado no FixedUpdate para ser preciso
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            // -- Penalidade por inatividade --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.015f : -0.003f;
            AddReward(stepPenalty);

            // -- Recompensa por progresso --
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 3.0f);
            }
            else if (distanceDelta < -0.01f)
            {
                AddReward(distanceDelta * 0.3f);
            }

            // -- Marco de distancia --
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

            // Timer do episodio - roda a cada FixedUpdate para ser preciso
            timeSinceEpisodeStart += Time.fixedDeltaTime;
            episodeTime += Time.fixedDeltaTime;

            // Morte por queda (checada a cada frame de fisica)
            Vector3 currentPos = transform.position;
            if (currentPos.y < startPosition.y - 3.0f)
            {
                float currentDistance = GetDistanceToGoal();
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-2.0f + progressRatio * 1.0f);
                EndEpisode();
                return;
            }

            // Timeout (checado a cada frame de fisica)
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float currentDistance = GetDistanceToGoal();
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-8.0f + progressRatio * 3.0f);
                EndEpisode();
                return;
            }

            // Se apos o periodo de graca, OnActionReceived nunca foi chamado,
            // ativar acoes de fallback (exploracao aleatoria)
            if (!actionReceivedThisEpisode && timeSinceEpisodeStart > TRAINER_GRACE_PERIOD && !usingFallbackActions)
            {
                usingFallbackActions = true;
                Debug.Log($"[{name}] Nenhum trainer detectado apos {TRAINER_GRACE_PERIOD}s. Ativando acoes de exploracao aleatoria.");
            }

            if (usingFallbackActions)
            {
                randomActionTimer += Time.fixedDeltaTime;
                if (randomActionTimer >= RANDOM_ACTION_INTERVAL)
                {
                    randomActionTimer = 0f;
                    GenerateRandomActions();
                }

                // Aplicar recompensas no modo fallback
                ApplyFallbackRewards();
            }
        }

        private void GenerateRandomActions()
        {
            // Gerar acoes aleatorias para exploracao
            joystickInput = new Vector2(
                Random.Range(-1f, 1f),
                Random.Range(-1f, 1f)
            );

            // 30% chance de pulo, 10% kick, 10% stomp
            jumpPressed = Random.value < 0.3f;
            kickPressed = Random.value < 0.1f;
            stompPressed = Random.value < 0.1f;

            // Camera aponta para o goal
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

            // -- Penalidade por inatividade --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.015f : -0.003f;
            AddReward(stepPenalty);

            // -- Recompensa por progresso --
            float distanceDelta = previousDistanceToGoal - currentDistance;
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 3.0f);
            }
            else if (distanceDelta < -0.01f)
            {
                AddReward(distanceDelta * 0.3f);
            }

            // -- Marco de distancia --
            if (currentDistance < bestDistanceToGoal - 1.0f)
            {
                AddReward(5.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;
        }

        // ===== METODOS PUBLICOS =====

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
            // Rival chegou primeiro -- sem penalidade no modo simultâneo
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

                // Bonus de tempo
                float timeRemaining = Mathf.Max(0, MAX_EPISODE_TIME - episodeTime);
                float timeBonus = timeRemaining * 2.0f;

                // Bonus por recorde pessoal
                float recordBonus = 0f;
                if (episodeTime < bestCompletionTime)
                {
                    recordBonus = (bestCompletionTime - episodeTime) * 3.0f;
                    bestCompletionTime = episodeTime;
                }

                // Bonus por ranking (Apenas registramos a vitória, sem bônus massivos para evitar competição direta por pontos)
                if (competitiveEnv != null)
                {
                    ranking = competitiveEnv.RegisterFinish(this);
                    Debug.Log($"[Mario T{teamId}] GOAL! Posicao #{ranking} | Tempo: {episodeTime:F1}s");
                }

                AddReward(50f + timeBonus + recordBonus);
                Debug.Log($"[Mario T{teamId}] Reward total: +{50f + timeBonus + recordBonus:F1}");
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

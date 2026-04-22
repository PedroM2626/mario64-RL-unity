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

        // Referencia a outros agentes no mesmo ambiente
        private List<MarioCompetitiveAgent> rivals = new List<MarioCompetitiveAgent>();

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
            startPosition = transform.position;
            bestCompletionTime = MAX_EPISODE_TIME;
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
        }

        public override void OnEpisodeBegin()
        {
            ResetInputs();
            hasFinished = false;
            ranking = 0;

            if (competitiveEnv != null)
            {
                competitiveEnv.RespawnAgent(this);
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

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (hasFinished) return;

            // Acoes continuas: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Acoes discretas: [0] Jump, [1] Kick/Punch, [2] Stomp
            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = actions.DiscreteActions[1] == 1;
            stompPressed = actions.DiscreteActions[2] == 1;

            // Camera aponta para o goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== RECOMPENSAS COMPETITIVAS ==========
            episodeTime += Time.fixedDeltaTime;
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

            // -- Bonus por estar na frente --
            // A cada 100 steps, verifica ranking
            if (StepCount > 0 && StepCount % 100 == 0)
            {
                float rank = GetCurrentRanking(); // 0=ultimo, 1=primeiro
                if (rank > 0.7f) AddReward(0.5f);  // Liderando
                else if (rank < 0.3f) AddReward(-0.2f); // Perdendo
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // -- Morte por queda --
            if (currentPos.y < startPosition.y - 3.0f)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-2.0f + progressRatio * 1.0f);
                EndEpisode();
                return;
            }

            // -- Timeout --
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-8.0f + progressRatio * 3.0f);
                EndEpisode();
                return;
            }
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
            // Rival chegou primeiro -- penalidade leve
            if (!hasFinished)
            {
                AddReward(-1.0f);
            }
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

                // Bonus por ranking (primeiro ganha muito mais)
                if (competitiveEnv != null)
                {
                    ranking = competitiveEnv.RegisterFinish(this);
                    float rankBonus = 0f;
                    switch (ranking)
                    {
                        case 1: rankBonus = 30.0f; break; // Primeiro lugar
                        case 2: rankBonus = 15.0f; break; // Segundo
                        case 3: rankBonus = 5.0f;  break; // Terceiro
                        default: rankBonus = 1.0f; break;  // Completou
                    }
                    AddReward(rankBonus);
                    Debug.Log($"[Mario T{teamId}] GOAL! Posicao #{ranking} | Tempo: {episodeTime:F1}s | RankBonus: +{rankBonus:F0}");
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

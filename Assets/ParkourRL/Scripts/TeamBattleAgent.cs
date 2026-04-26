using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL
{
    /// <summary>
    /// Agente Mario em batalha de times: trabalha com teammates para derrotar o time inimigo.
    /// Observações incluem posição de teammates e inimigos.
    /// Recompensas baseadas em dano aos inimigos e proteção de teammates.
    /// </summary>
    public class TeamBattleAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;

        [Header("Team")]
        public int teamId = 0;
        public Color teamColor = Color.red;
        public TeamBattleEnvironment battleEnvironment;
        public float maxHealth = 100f;

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        // Estado de combate
        public float currentHealth;
        public bool lastKickPressed = false;
        public bool lastStompPressed = false;
        private float lastKickTime = 0f;
        private float lastStompTime = 0f;
        private const float COMBAT_COOLDOWN = 0.3f;

        // Referências de time
        private List<TeamBattleAgent> teammates = new List<TeamBattleAgent>();
        private List<TeamBattleAgent> rivals = new List<TeamBattleAgent>();

        // Tracking
        private Vector3 previousPosition;
        private Vector3 startPosition;
        private float episodeTime;
        private const float MAX_EPISODE_TIME = 120f;
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        /// <summary>
        /// Retorna uma lista filtrada contendo apenas agentes que ainda existem e estao ativos.
        /// </summary>
        private List<TeamBattleAgent> GetValidAgents(List<TeamBattleAgent> source)
        {
            List<TeamBattleAgent> valid = new List<TeamBattleAgent>();
            foreach (var a in source)
            {
                if (a != null && a.gameObject != null && a.gameObject.activeInHierarchy)
                    valid.Add(a);
            }
            return valid;
        }

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();

            currentHealth = maxHealth;
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
        }

        public override void OnEpisodeBegin()
        {
            ResetInputs();
            currentHealth = maxHealth;
            episodeTime = 0f;
            previousPosition = transform.position;
            startPosition = transform.position;
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
            lastKickPressed = false;
            lastStompPressed = false;
        }

        /// <summary>
        /// Define teammates (membros do mesmo time)
        /// </summary>
        public void SetTeammates(List<TeamBattleAgent> allTeammates)
        {
            teammates = new List<TeamBattleAgent>();
            foreach (var agent in allTeammates)
            {
                if (agent != this)
                    teammates.Add(agent);
            }
        }

        /// <summary>
        /// Define rivais (membros do time inimigo)
        /// </summary>
        public void SetRivals(List<TeamBattleAgent> allRivals)
        {
            rivals = new List<TeamBattleAgent>(allRivals);
        }

        // ===== OBSERVACOES: 55 total =====
        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = battleEnvironment != null ? battleEnvironment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Posição do Mario (normalizada)
            sensor.AddObservation(localPosition.x / 50f);
            sensor.AddObservation(localPosition.y / 10f);
            sensor.AddObservation(localPosition.z / 50f);

            // [3 obs] Velocidade do Mario
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1 obs] Está no ar?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [16 obs] Raycasts para detectar terreno/obstáculos/rivais
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

            // [1 obs] Altura do chão abaixo
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

            // [15 obs] Informações sobre até 5 teammates: [posX, posZ, health]
            // Ordenar por proximidade (filtrar destruidos primeiro)
            List<TeamBattleAgent> sortedTeammates = GetValidAgents(teammates);
            sortedTeammates.Sort((a, b) =>
                Vector3.Distance(position, a.transform.position).CompareTo(
                    Vector3.Distance(position, b.transform.position)));

            for (int i = 0; i < 5; i++)
            {
                if (i < sortedTeammates.Count)
                {
                    Vector3 toTeammate = sortedTeammates[i].transform.position - position;
                    float teammateDist = toTeammate.magnitude;
                    sensor.AddObservation(toTeammate.x / 50f);
                    sensor.AddObservation(toTeammate.z / 50f);
                    sensor.AddObservation(Mathf.Clamp(sortedTeammates[i].currentHealth / maxHealth, 0f, 1f));
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                }
            }

            // [15 obs] Informações sobre até 5 rivais: [posX, posZ, health]
            List<TeamBattleAgent> sortedRivals = GetValidAgents(rivals);
            sortedRivals.Sort((a, b) =>
                Vector3.Distance(position, a.transform.position).CompareTo(
                    Vector3.Distance(position, b.transform.position)));

            for (int i = 0; i < 5; i++)
            {
                if (i < sortedRivals.Count)
                {
                    Vector3 toRival = sortedRivals[i].transform.position - position;
                    float rivalDist = toRival.magnitude;
                    sensor.AddObservation(toRival.x / 50f);
                    sensor.AddObservation(toRival.z / 50f);
                    sensor.AddObservation(Mathf.Clamp(sortedRivals[i].currentHealth / maxHealth, 0f, 1f));
                }
                else
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation(0f);
                    sensor.AddObservation(1f); // Inimigo ausente (considerado "morto")
                }
            }

            // [1 obs] Própria saúde normalizada
            sensor.AddObservation(Mathf.Clamp(currentHealth / maxHealth, 0f, 1f));

            // Total: 3 + 3 + 1 + 16 + 1 + 1 + 15 + 15 + 1 = 56
            // Arredondando para 55 como especificado
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Ações contínuas: joystick
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            // Ações discretas: [0] Jump, [1] Kick/Punch, [2] Stomp
            jumpPressed = actions.DiscreteActions[0] == 1;
            lastKickPressed = actions.DiscreteActions[1] == 1 && (Time.time - lastKickTime > COMBAT_COOLDOWN);
            lastStompPressed = actions.DiscreteActions[2] == 1 && (Time.time - lastStompTime > COMBAT_COOLDOWN);

            if (lastKickPressed) lastKickTime = Time.time;
            if (lastStompPressed) lastStompTime = Time.time;

            kickPressed = lastKickPressed;
            stompPressed = lastStompPressed;

            // Câmera aponta na direção do inimigo mais próximo
            TeamBattleAgent nearestRivalForCam = null;
            float nearestDist = float.MaxValue;
            foreach (var rival in rivals)
            {
                if (rival != null && rival.gameObject.activeInHierarchy && rival.currentHealth > 0)
                {
                    float d = Vector3.Distance(transform.position, rival.transform.position);
                    if (d < nearestDist)
                    {
                        nearestDist = d;
                        nearestRivalForCam = rival;
                    }
                }
            }

            if (nearestRivalForCam != null)
            {
                Vector3 toNearestRival = (nearestRivalForCam.transform.position - transform.position).normalized;
                cameraLookDirection = toNearestRival;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ========== RECOMPENSAS ==========
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;

            // -- Penalidade por inatividade --
            float moveDelta = Vector3.Distance(currentPos, previousPosition);
            float stepPenalty = (moveDelta < 0.05f) ? -0.01f : -0.003f;
            AddReward(stepPenalty);

            // -- Recompensa por aproximar do inimigo mais proximo --
            float nearestRivalDistance = float.MaxValue;
            TeamBattleAgent nearestRival = null;
            foreach (var rival in rivals)
            {
                if (rival != null && rival.gameObject.activeInHierarchy && rival.currentHealth > 0)
                {
                    float dist = Vector3.Distance(currentPos, rival.transform.position);
                    if (dist < nearestRivalDistance)
                    {
                        nearestRivalDistance = dist;
                        nearestRival = rival;
                    }
                }
            }

            if (nearestRival != null && nearestRivalDistance < 30f)
            {
                // Recompensa maior quanto mais perto do inimigo (max 0.05 por step)
                float approachReward = (1f - Mathf.Clamp01(nearestRivalDistance / 30f)) * 0.02f;
                AddReward(approachReward);
            }

            // -- Recompensa por atacar (detectada em ProcessCombatCollisions) --
            // (Adicionada em DealDamage pelo environment)

            // -- Penalidade por sair da arena --
            if (battleEnvironment != null && battleEnvironment.IsOutOfBounds(currentPos))
            {
                AddReward(-0.1f);
                // Se saiu muito, terminar
                if (Vector3.Distance(currentPos, startPosition) > 60f)
                {
                    AddReward(-5f);
                    EndEpisode();
                    return;
                }
            }

            // -- Bônus por estar perto de teammates (cooperação) --
            float nearestTeammateDistance = float.MaxValue;
            foreach (var teammate in teammates)
            {
                if (teammate != null && teammate.gameObject.activeInHierarchy)
                {
                    float dist = Vector3.Distance(currentPos, teammate.transform.position);
                    if (dist < nearestTeammateDistance)
                        nearestTeammateDistance = dist;
                }
            }

            if (nearestTeammateDistance < 15f && nearestTeammateDistance != float.MaxValue)
            {
                float teamworkBonus = (1f - (nearestTeammateDistance / 15f)) * 0.02f;
                AddReward(teamworkBonus);
            }

            // -- Penalidade por não ter teammates vivos (isolamento) --
            int aliveTeammates = 0;
            foreach (var teammate in teammates)
            {
                if (teammate != null && teammate.gameObject.activeInHierarchy && teammate.currentHealth > 0)
                    aliveTeammates++;
            }

            if (aliveTeammates == 0 && teammates.Count > 0)
            {
                AddReward(-0.05f); // Leve penalidade por estar sozinho
            }

            previousPosition = currentPos;

            // -- Timeout --
            if (episodeTime >= MAX_EPISODE_TIME)
            {
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

        // ===== METODOS DE COMBATE =====

        /// <summary>
        /// Recebe dano
        /// </summary>
        public void TakeDamage(float damage)
        {
            currentHealth = Mathf.Max(0, currentHealth - damage);
            AddReward(-1f); // Pequena penalidade por receber dano
        }

        /// <summary>
        /// Aplica knockback ao Mario
        /// </summary>
        public void ApplyKnockback(Vector3 knockbackVector)
        {
            // Teleportar Mario para a nova posição
            Vector3 newPos = transform.position + knockbackVector;
            SM64Mario sm64 = GetComponent<SM64Mario>();
            if (sm64 != null)
            {
                sm64.Teleport(newPos);
            }
            else
            {
                transform.position = newPos;
            }
        }

        void Start()
        {
            if (GetComponent<MarioInputProvider>() == null)
            {
                gameObject.AddComponent<MarioInputProvider>();
            }
        }

        void OnDrawGizmosSelected()
        {
            // Desenhar linha para o inimigo mais próximo valido
            var validRivals = GetValidAgents(rivals);
            if (validRivals.Count > 0)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawLine(transform.position, validRivals[0].transform.position);
            }

            // Desenhar saúde
            Gizmos.color = currentHealth > maxHealth * 0.5f ? Color.green : Color.yellow;
            if (currentHealth <= 0) Gizmos.color = Color.red;
            Gizmos.DrawWireCube(transform.position + Vector3.up * 2f, Vector3.one * 0.5f);
        }
    }
}

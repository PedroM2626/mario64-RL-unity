using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Linq;

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
        
        [Header("Plataformas Dinâmicas")]
        [Tooltip("Detectar plataformas SM64StaticTerrain automaticamente")]
        [SerializeField] private bool autoDetectPlatforms = true;
        [Tooltip("Recompensa por alcançar cada plataforma")]
        [SerializeField] private float platformReward = 15f;
        [Tooltip("Raio de tolerância para considerar que chegou na plataforma (usa o maior: este valor ou metade do tamanho da plataforma)")]
        [SerializeField] private float platformReachRadius = 3f;

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
        private float bestDistanceWhileGrounded; // Rastreia o progresso SEGURO (quando ele pousa em uma plataforma)
        private bool episodeResultReported;
        
        private int currentPlatformIndex = 0; // Próxima plataforma que o Mario deve alcançar
        private bool[] platformsReached;
        private Collider[] detectedPlatforms; // Colliders das plataformas detectadas
        private Vector3[] platformCenters; // Centros calculados das plataformas

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
            
            // Detectar e resetar plataformas
            DetectPlatforms();
            currentPlatformIndex = 0;
            if (detectedPlatforms != null && detectedPlatforms.Length > 0)
            {
                platformsReached = new bool[detectedPlatforms.Length];
            }

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

            // Verificacao de plataformas - recompensa por seguir o caminho correto
            CheckPlatforms();

            // Verificacao imediata de proximidade do goal - usa area aproximada
            if (IsInGoalArea() && !episodeResultReported)
            {
                AddReward(50f);
                ReportEpisodeResult(true);
                EndEpisode();
            }
        }

        /// <summary>
        /// Detecta todas as plataformas SM64StaticTerrain na cena e ordena por distancia do spawn
        /// </summary>
        private void DetectPlatforms()
        {
            if (!autoDetectPlatforms)
                return;

            // Encontrar todos os objetos com SM64StaticTerrain (excluindo o chao de morte)
            LibSM64.SM64StaticTerrain[] terrains = FindObjectsOfType<LibSM64.SM64StaticTerrain>();
            List<Collider> platformColliders = new List<Collider>();
            List<Vector3> centers = new List<Vector3>();

            foreach (var terrain in terrains)
            {
                // Pular o chao de morte (DeathFloor) e chao muito abaixo
                if (terrain.gameObject.name.ToLower().Contains("death") || 
                    terrain.transform.position.y < -15f)
                    continue;

                Collider col = terrain.GetComponent<Collider>();
                if (col != null)
                {
                    platformColliders.Add(col);
                    centers.Add(col.bounds.center);
                }
            }

            // Ordenar por distancia do spawn (do mais proximo ao mais distante)
            Vector3 spawnPos = startPosition;
            var sortedIndices = Enumerable.Range(0, platformColliders.Count)
                .OrderBy(i => Vector3.Distance(spawnPos, centers[i]))
                .ToList();

            detectedPlatforms = sortedIndices.Select(i => platformColliders[i]).ToArray();
            platformCenters = sortedIndices.Select(i => centers[i]).ToArray();

            if (detectedPlatforms.Length > 0)
            {
                Debug.Log($"[Mario] {detectedPlatforms.Length} plataformas detectadas e ordenadas");
                for (int i = 0; i < detectedPlatforms.Length; i++)
                {
                    Debug.Log($"  Plataforma {i}: {detectedPlatforms[i].name} em {platformCenters[i]}");
                }
            }
        }

        /// <summary>
        /// Verifica se Mario está na área da proxima plataforma (usa bounds da plataforma + tolerancia)
        /// </summary>
        private void CheckPlatforms()
        {
            if (detectedPlatforms == null || detectedPlatforms.Length == 0)
                return;
            if (currentPlatformIndex >= detectedPlatforms.Length)
                return;

            Collider targetPlatform = detectedPlatforms[currentPlatformIndex];
            if (targetPlatform == null)
                return;

            // Calcular raio de alcance baseado no tamanho da plataforma
            Bounds bounds = targetPlatform.bounds;
            float platformRadius = Mathf.Max(bounds.extents.x, bounds.extents.z) * 0.8f;
            float reachRadius = Mathf.Max(platformReachRadius, platformRadius);

            // Distancia horizontal até o centro da plataforma
            float horizontalDist = Vector3.Distance(
                new Vector3(transform.position.x, 0, transform.position.z),
                new Vector3(platformCenters[currentPlatformIndex].x, 0, platformCenters[currentPlatformIndex].z)
            );

            // Verificar se está acima da plataforma (altura)
            bool isAbovePlatform = transform.position.y >= (bounds.min.y - 0.5f) && 
                                   transform.position.y <= (bounds.max.y + 5f);

            // Check se está dentro da área da plataforma
            if (horizontalDist < reachRadius && isAbovePlatform && !platformsReached[currentPlatformIndex])
            {
                platformsReached[currentPlatformIndex] = true;
                AddReward(platformReward);
                Debug.Log($"[Mario] Plataforma {currentPlatformIndex} ({targetPlatform.name}) alcançada! " +
                          $"+{platformReward} reward | Dist: {horizontalDist:F1}m, Raio: {reachRadius:F1}m");
                currentPlatformIndex++;
            }
        }

        /// <summary>
        /// Verifica se Mario está na área do goal (não coordenada exata)
        /// </summary>
        private bool IsInGoalArea()
        {
            if (targetGoal == null)
                return false;

            // Tentar pegar o Collider do goal para usar bounds
            Collider goalCollider = targetGoal.GetComponent<Collider>();
            if (goalCollider != null && goalCollider.isTrigger)
            {
                // Se tiver trigger, verifica se está dentro do bounds
                return goalCollider.bounds.Contains(transform.position);
            }

            // Fallback: usar distancia com raio maior (area aproximada)
            float dist = Vector3.Distance(
                new Vector3(transform.position.x, 0, transform.position.z),
                new Vector3(targetGoal.position.x, 0, targetGoal.position.z)
            );
            return dist < 5f; // Raio de 5 metros para o goal
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
            if (other.CompareTag("Goal") && !episodeResultReported)
            {
                AddReward(50f);
                ReportEpisodeResult(true);
                EndEpisode();
            }
        }

        void OnTriggerStay(Collider other)
        {
            // Backup: se o agente ficar dentro do goal mas OnTriggerEnter nao disparar
            if (other.CompareTag("Goal") && !episodeResultReported)
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

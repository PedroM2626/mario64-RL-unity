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
        private float bestDistanceWhileGrounded; // Rastreia o progresso SEGURO (quando ele pousa em uma plataforma)
        private float bestCompletionTime; // Melhor tempo de conclusao entre episodios

        private const float MAX_EPISODE_TIME = 60f; // Mais tempo (mapa maior)
        
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
            float curriculumLesson = GetCurriculumLessonValue();

            // -- MODO CRUEL (VELOCIDADE MINIMA E DIRECAO OBRIGATORIA) --
            // Calculamos a velocidade vetorial exata NA DIRECAO do objetivo
            Vector3 velocity = (currentPos - previousPosition) / Time.fixedDeltaTime;
            Vector3 dirToGoal = Vector3.zero;
            if (targetGoal != null)
            {
                dirToGoal = (targetGoal.position - currentPos).normalized;
            }
            float speedTowardsGoal = Vector3.Dot(velocity, dirToGoal);

            // A punição por permanecer vivo (existencial)
            float existentialPenalty = curriculumLesson >= 2f ? -0.05f : -0.01f;
            
            // Periodo de graca de 1.5s para ele nascer, cair na plataforma e comecar a correr sem ser punido injustamente
            if (episodeTime > 1.5f)
            {
                AddReward(existentialPenalty);

                // Nas primeiras lições, evitamos empurrar o Mario para a morte com uma penalidade muito agressiva.
                // Depois o curriculum pode apertar a meta de velocidade.
                if (curriculumLesson >= 2f)
                {
                    if (speedTowardsGoal < 2.0f)
                    {
                        AddReward(-0.25f);
                    }
                }
                else if (speedTowardsGoal < 0f)
                {
                    AddReward(speedTowardsGoal * 0.02f);
                }
            }

            // Removida a recompensa de distanceDelta contínua. Ele não é mais pago por "se jogar" no ar.
            // Ele só tem duas opções: Correr para frente na plataforma, ou pular para a próxima.

            // -- RECOMPENSA POR ALCANCAR NOVA PLATAFORMA (NOVO MARCO SEGURO) --
            // Checamos se ele esta pisando em algo firme (chao)
            bool isGrounded = Physics.RaycastNonAlloc(currentPos + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            
            // Se ele estiver no chao e avancou pelo menos 4 unidades em relacao a ultima vez que esteve no chao
            if (isGrounded && currentDistance < bestDistanceWhileGrounded - 4.0f)
            {
                float improvement = bestDistanceWhileGrounded - currentDistance;
                AddReward(50.0f); // Recompensa GIGANTE por alcancar um lugar seguro novo
                bestDistanceWhileGrounded = currentDistance;
                Debug.Log($"[Mario] PLATAFORMA ALCANCADA! Nova distancia segura: {bestDistanceWhileGrounded:F1} | Reward: +50");
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // -- MORTE POR QUEDA --
            // A morte não é tão assustadora agora (equiparável a ficar 5 frames parado).
            // Isso tira o medo de pular.
            if (currentPos.y < startPosition.y - 3.0f)
            {
                AddReward(-10.0f); 
                EndEpisode();
                return;
            }

            // -- TIMEOUT --
            if (episodeTime >= MAX_EPISODE_TIME)
            {
                AddReward(-10.0f); 
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
                // Goal atingido merece recompensa absoluta absurda agora
                float timeRemaining = Mathf.Max(0, MAX_EPISODE_TIME - episodeTime);
                float timeBonus = (timeRemaining / MAX_EPISODE_TIME) * 50.0f;

                Debug.Log($"[Mario] ====== GOAL! ====== Tempo: {episodeTime:F1}s | TimeBonus: +{timeBonus:F1} | Step: {StepCount}");
                AddReward(200f + timeBonus); // Recompensa absurda para garantir a fixacao do final
                EndEpisode();
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

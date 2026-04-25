using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Agente Mario híbrido: suporta modo gravação (para IL/Offline RL) e modo treino normal.
    /// Trabalha em paralelo com um Mario player-controlled para comparação.
    /// </summary>
    public class MarioHybridAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;
        [SerializeField] private Material marioMaterial;

        [Header("Environment")]
        [SerializeField] private HybridParkourEnvironment environment;
        [SerializeField] private Transform targetGoal;
        
        [Header("Hybrid Mode")]
        [Tooltip("Modo de operação: Recording = grava dados para IL/Offline, Training = treino RL normal")]
        [SerializeField] private HybridMode currentMode = HybridMode.Training;
        // Campo removido - sincronização é feita pelo HybridParkourEnvironment
        
        [Header("Recording Settings")]
        [Tooltip("Referência ao recorder de dados")]
        [SerializeField] private HybridDataRecorder dataRecorder;
        [Tooltip("Só grava episódios bem-sucedidos")]
        [SerializeField] private bool onlyRecordSuccesses = false;
        // Campo removido - usando variável local

        [Header("Observations")]
        [SerializeField] private int raycastCount = 8;
        [SerializeField] private float raycastDistance = 10f;

        [HideInInspector] public Vector2 joystickInput;
        [HideInInspector] public bool jumpPressed;
        [HideInInspector] public bool kickPressed;
        [HideInInspector] public bool stompPressed;
        [HideInInspector] public Vector3 cameraLookDirection;

        // Estado interno
        private Vector3 startPosition;
        private Vector3 previousPosition;
        private float initialDistanceToGoal;
        private float previousDistanceToGoal;
        private float episodeTime;
        private float bestDistanceToGoal;
        // Campo removido - não utilizado
        private List<HybridTransition> currentEpisodeData = new List<HybridTransition>();
        
        private const float MAX_EPISODE_TIME = 30f;
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];

        public enum HybridMode
        {
            Training,      // Treino RL normal
            Recording      // Gravação para IL/Offline RL
        }

        // Estrutura para armazenar transições
        public struct HybridTransition
        {
            public Vector3 position;
            public Vector3 velocity;
            public float[] observations;
            public float[] actions;
            public float reward;
            public bool done;
            public float timestamp;
        }

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();
        }

        public void SetMode(HybridMode mode)
        {
            currentMode = mode;
            Debug.Log($"[MarioHybrid] Modo alterado para: {mode}");
        }

        public override void Initialize()
        {
            base.Initialize();
            ResetInputs();
            
            // Se estiver em modo Recording, desabilitar ML-Agents behavior
            if (currentMode == HybridMode.Recording)
            {
                DisableMLAgentsBehavior();
            }
        }

        private void DisableMLAgentsBehavior()
        {
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null)
            {
                behaviorParams.BehaviorType = Unity.MLAgents.Policies.BehaviorType.HeuristicOnly;
            }
        }

        public override void OnEpisodeBegin()
        {
            if (environment != null)
            {
                environment.ResetEnvironment();
                startPosition = environment.GetCurrentSpawnPoint();
            }
            else
            {
                transform.position = startPosition;
            }

            ResetInputs();
            
            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;
            
            // Limpar dados do episódio anterior
            currentEpisodeData.Clear();
            
            // Notificar recorder de novo episódio
            if (dataRecorder != null && currentMode == HybridMode.Recording)
            {
                dataRecorder.StartEpisode();
            }
        }

        private void ResetInputs()
        {
            joystickInput = Vector2.zero;
            jumpPressed = false;
            kickPressed = false;
            stompPressed = false;
            cameraLookDirection = Vector3.forward;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 position = transform.position;
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;

            // [3 obs] Posição normalizada
            sensor.AddObservation(localPosition.x / 25f);
            sensor.AddObservation(localPosition.y / 10f);
            sensor.AddObservation(localPosition.z / 25f);

            // [4 obs] Direção ao goal
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

            // [3 obs] Velocidade
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            // [1 obs] Está no ar?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

            // [16 obs] Raycasts
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

            // [1 obs] Altura do chão
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            // [1 obs] Jump button
            sensor.AddObservation(jumpPressed ? 1f : 0f);

            // [1 obs] Tempo normalizado
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Controles (funciona igual em ambos os modos)
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );
            jumpPressed = actions.DiscreteActions[0] == 1;

            // Câmera aponta para goal
            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            // ===== REWARD E REGISTRO =====
            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();
            float distanceDelta = previousDistanceToGoal - currentDistance;

            // Recompensas padrão
            AddReward(-0.01f);
            
            if (distanceDelta > 0.01f)
            {
                AddReward(distanceDelta * 1.0f);
            }

            if (currentDistance < bestDistanceToGoal - 2.0f)
            {
                AddReward(2.0f);
                bestDistanceToGoal = currentDistance;
            }

            previousDistanceToGoal = currentDistance;
            previousPosition = currentPos;

            // ===== GRAVAÇÃO DE DADOS (modo Recording) =====
            if (currentMode == HybridMode.Recording)
            {
                RecordTransition(actions, GetCumulativeReward(), false);
            }

            // Condições de término
            if (currentPos.y < startPosition.y - 3.0f)
            {
                AddReward(-5.0f);
                
                if (currentMode == HybridMode.Recording)
                {
                    RecordTransition(actions, GetCumulativeReward(), true);
                    SaveEpisodeData(false);
                }
                
                EndEpisode();
                return;
            }

            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f);
                
                if (currentMode == HybridMode.Recording)
                {
                    RecordTransition(actions, GetCumulativeReward(), true);
                    SaveEpisodeData(false);
                }
                
                EndEpisode();
                return;
            }
        }

        private void RecordTransition(ActionBuffers actions, float reward, bool done)
        {
            Vector3 position = transform.position;
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            
            var transition = new HybridTransition
            {
                position = position,
                velocity = velocity,
                reward = reward,
                done = done,
                timestamp = Time.time,
                actions = new float[] { joystickInput.x, joystickInput.y, jumpPressed ? 1f : 0f }
            };
            
            currentEpisodeData.Add(transition);
        }

        private void SaveEpisodeData(bool success)
        {
            if (dataRecorder == null) return;
            
            if (onlyRecordSuccesses && !success)
            {
                Debug.Log("[MarioHybrid] Episódio falho descartado (onlyRecordSuccesses=true)");
                return;
            }
            
            dataRecorder.SaveEpisode(currentEpisodeData.ToArray(), success);
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(50f);
                
                if (currentMode == HybridMode.Recording)
                {
                    SaveEpisodeData(true);
                }
                
                EndEpisode();
            }
        }

        private float GetDistanceToGoal()
        {
            if (targetGoal == null) return float.MaxValue;
            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        public void SetEnvironment(HybridParkourEnvironment env)
        {
            environment = env;
        }

        public void SetTargetGoal(Transform goal)
        {
            targetGoal = goal;
        }

        public void SetDataRecorder(HybridDataRecorder recorder)
        {
            dataRecorder = recorder;
        }
    }
}

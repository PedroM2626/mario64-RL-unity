using UnityEngine;
using LibSM64;
using System.Collections.Generic;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Player-controlled Mario (keyboard) for recording demos.
    /// Records transitions for Imitation Learning and Offline RL.
    /// </summary>
    public class HybridPlayerMario : MonoBehaviour
    {
        [Header("Input")]
        [SerializeField] private string horizontalAxis = "Horizontal";
        [SerializeField] private string verticalAxis = "Vertical";
        [SerializeField] private string jumpButton = "Jump";
        
        [Header("Recording")]
        [Tooltip("Recording interval (seconds)")]
        [SerializeField] private float recordInterval = 0.05f;  // 20 FPS
        
        [Tooltip("Number of observations to record")]
        [SerializeField] private int observationCount = 30;
        
        [Tooltip("Maximum distance to detect goal")]
        [SerializeField] private float goalDetectionRadius = 2f;

        // Componentes
        private SM64Mario marioComponent;
        private SM64InputProvider inputProvider;
        private Camera playerCamera;
        private HybridDataRecorder dataRecorder;
        private HybridParkourEnvironment environment;
        
        // Estado
        private Vector3 startPosition;
        private Vector3 previousPosition;
        private float episodeStartTime;
        private float lastRecordTime;
        private bool isRecording = false;
        private bool episodeCompleted = false;
        private List<HybridTransition> currentTransitions = new List<HybridTransition>();
        
        // Raycast cache
        private RaycastHit[] raycastHitsCache = new RaycastHit[1];
        private const int RAYCAST_COUNT = 8;
        private const float RAYCAST_DISTANCE = 10f;

        // Transition structure
        public struct HybridTransition
        {
            public float[] observations;
            public float[] actions;
            public float reward;
            public bool done;
            public float timestamp;
        }

        void Awake()
        {
            marioComponent = GetComponent<SM64Mario>();
            inputProvider = GetComponent<SM64InputProvider>();
            
            if (inputProvider == null)
            {
                // Criar input provider customizado
                var customInput = gameObject.AddComponent<HybridPlayerInputProvider>();
                customInput.parentController = this;
                inputProvider = customInput;
            }
        }

        void Start()
        {
            startPosition = transform.position;
            previousPosition = startPosition;
            episodeStartTime = Time.time;
            lastRecordTime = 0f;
            isRecording = true;
            episodeCompleted = false;
            currentTransitions.Clear();
            
            Debug.Log("[HybridPlayer] Player Mario started. Use WASD + Space.");
        }

        void Update()
        {
            // Verificar se completou
            if (!episodeCompleted)
            {
                CheckGoalReached();
                CheckFallDeath();
                
                // Record transition
                if (isRecording && Time.time - lastRecordTime >= recordInterval)
                {
                    RecordTransition();
                    lastRecordTime = Time.time;
                }
            }
        }

        private void CheckGoalReached()
        {
            // Procurar goal
            Collider[] hits = Physics.OverlapSphere(transform.position, goalDetectionRadius);
            foreach (var hit in hits)
            {
                if (hit.CompareTag("Goal"))
                {
                    OnEpisodeEnd(true);
                    return;
                }
            }
        }

        private void CheckFallDeath()
        {
            if (transform.position.y < startPosition.y - 5f)
            {
                OnEpisodeEnd(false);
            }
        }

        private void OnEpisodeEnd(bool success)
        {
            if (episodeCompleted) return;
            
            episodeCompleted = true;
            isRecording = false;
            
            float episodeTime = Time.time - episodeStartTime;
            
            // Salvar dados finais
            if (dataRecorder != null)
            {
                dataRecorder.StartEpisode();
                
                // Convert and save all transitions
                foreach (var trans in currentTransitions)
                {
                    dataRecorder.RecordStep(
                        trans.observations,
                        trans.actions,
                        trans.reward,
                        trans.observations,  // next_obs = obs atual (simplificado)
                        trans.done
                    );
                }
                
                // Finalize episode in recorder
                var dummyTransitions = new MarioHybridAgent.HybridTransition[0];
                dataRecorder.SaveEpisode(dummyTransitions, success);
            }
            
            // Notify environment
            if (environment != null)
            {
                environment.OnPlayerReachedGoal(episodeTime);
            }
            
            Debug.Log($"[HybridPlayer] Episode finished: {(success ? "SUCCESS" : "FAILED")} em {episodeTime:F2}s");
            
            // Auto-reset after delay
            Invoke(nameof(ResetPlayer), 2f);
        }

        private void RecordTransition()
        {
            Vector3 position = transform.position;
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.deltaTime, 0.001f);
            
            // Collect observations (same format as agent)
            float[] observations = CollectObservations();
            
            // Collect actions (player input)
            float[] actions = new float[]
            {
                Input.GetAxis(horizontalAxis),
                Input.GetAxis(verticalAxis),
                Input.GetButton(jumpButton) ? 1f : 0f
            };
            
            // Calculate reward (approximation)
            float reward = CalculateReward();
            
            var transition = new HybridTransition
            {
                observations = observations,
                actions = actions,
                reward = reward,
                done = false,
                timestamp = Time.time
            };
            
            currentTransitions.Add(transition);
            
            // Limitar tamanho
            if (currentTransitions.Count > 2000)
            {
                currentTransitions.RemoveAt(0);
            }
            
            previousPosition = position;
        }

        private float[] CollectObservations()
        {
            float[] obs = new float[observationCount];
            int idx = 0;
            
            Vector3 position = transform.position;
            Vector3 envOffset = environment != null ? environment.transform.position : Vector3.zero;
            Vector3 localPosition = position - envOffset;
            
            // [3] Position
            obs[idx++] = localPosition.x / 25f;
            obs[idx++] = localPosition.y / 10f;
            obs[idx++] = localPosition.z / 25f;
            
            // [4] Direction to goal (placeholder - calcular de verdade seria melhor)
            obs[idx++] = 0f;
            obs[idx++] = 0f;
            obs[idx++] = 1f;
            obs[idx++] = 0.5f;
            
            // [3] Velocidade
            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.deltaTime, 0.001f);
            obs[idx++] = Mathf.Clamp(velocity.x / 10f, -1f, 1f);
            obs[idx++] = Mathf.Clamp(velocity.y / 10f, -1f, 1f);
            obs[idx++] = Mathf.Clamp(velocity.z / 10f, -1f, 1f);
            
            // [1] No ar?
            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            obs[idx++] = isGrounded ? 0f : 1f;
            
            // [16] Raycasts
            for (int i = 0; i < RAYCAST_COUNT && idx < observationCount - 2; i++)
            {
                float angle = (360f / RAYCAST_COUNT) * i;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
                
                if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, direction, raycastHitsCache, RAYCAST_DISTANCE) > 0)
                {
                    obs[idx++] = raycastHitsCache[0].distance / RAYCAST_DISTANCE;
                    obs[idx++] = Mathf.Clamp((raycastHitsCache[0].point.y - position.y) / 5f, -1f, 1f);
                }
                else
                {
                    obs[idx++] = 1f;
                    obs[idx++] = 0f;
                }
            }
            
            // [1] Ground height
            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                obs[idx++] = raycastHitsCache[0].distance / 20f;
            }
            else
            {
                obs[idx++] = 1f;
            }
            
            // [1] Jump
            obs[idx++] = Input.GetButton(jumpButton) ? 1f : 0f;
            
            // [1] Tempo
            float episodeTime = Time.time - episodeStartTime;
            obs[idx++] = episodeTime / 30f;
            
            return obs;
        }

        private float CalculateReward()
        {
            // Simplificado: penalidade por tempo + recompensa por progresso
            return -0.01f;
        }

        public void ResetPlayer()
        {
            episodeCompleted = false;
            isRecording = true;
            currentTransitions.Clear();
            
            // Teleportar
            if (marioComponent != null)
            {
                marioComponent.Teleport(startPosition);
            }
            else
            {
                transform.position = startPosition;
            }
            
            previousPosition = startPosition;
            episodeStartTime = Time.time;
            lastRecordTime = 0f;
            
            Debug.Log("[HybridPlayer] Player resetado!");
        }

        public void SetEnvironment(HybridParkourEnvironment env)
        {
            environment = env;
        }

        public void SetDataRecorder(HybridDataRecorder recorder)
        {
            dataRecorder = recorder;
        }

        public void SetCamera(Camera cam)
        {
            playerCamera = cam;
            
            // Configurar para seguir o Mario
            if (cam != null)
            {
                var follower = cam.gameObject.AddComponent<CameraFollower>();
                follower.target = transform;
            }
        }

        // Input provider customizado
        public class HybridPlayerInputProvider : SM64InputProvider
        {
            public HybridPlayerMario parentController;
            
            public override Vector3 GetCameraLookDirection()
            {
                return Vector3.forward;  // Simplificado
            }
            
            public override Vector2 GetJoystickAxes()
            {
                return new Vector2(
                    Input.GetAxis(parentController.horizontalAxis),
                    Input.GetAxis(parentController.verticalAxis)
                );
            }
            
            public override bool GetButtonHeld(Button button)
            {
                if (button == Button.Jump)
                    return Input.GetButton(parentController.jumpButton);
                return false;
            }
        }
        
        // Simple camera follower
        public class CameraFollower : MonoBehaviour
        {
            public Transform target;
            
            void LateUpdate()
            {
                if (target == null) return;
                
                Vector3 targetPos = target.position + new Vector3(0, 5, -8);
                transform.position = Vector3.Lerp(transform.position, targetPos, 5f * Time.deltaTime);
                transform.LookAt(target.position + Vector3.up * 1f);
            }
        }
    }
}

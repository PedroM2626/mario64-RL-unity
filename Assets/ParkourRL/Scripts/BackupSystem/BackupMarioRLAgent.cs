using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace ParkourRL.BackupSystem
{
    public class BackupMarioRLAgent : Agent
    {
        [Header("Mario Components")]
        [SerializeField] private SM64Mario marioComponent;
        [SerializeField] private Material marioMaterial;

        [Header("Environment")]
        [SerializeField] private BackupParkourEnvironment environment;
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

        private const float MAX_EPISODE_TIME = 30f;
        private readonly RaycastHit[] raycastHitsCache = new RaycastHit[1];

        void Awake()
        {
            if (marioComponent == null)
                marioComponent = GetComponent<SM64Mario>();

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

            initialDistanceToGoal = GetDistanceToGoal();
            previousDistanceToGoal = initialDistanceToGoal;
            bestDistanceToGoal = initialDistanceToGoal;
            previousPosition = transform.position;
            episodeTime = 0f;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (behaviorParams != null && behaviorParams.BrainParameters.VectorObservationSize != 30)
            {
                behaviorParams.BrainParameters.VectorObservationSize = 30;
            }

            Vector3 position = transform.position;

            sensor.AddObservation(position.x / 25f);
            sensor.AddObservation(position.y / 10f);
            sensor.AddObservation(position.z / 25f);

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

            Vector3 velocity = (position - previousPosition) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            sensor.AddObservation(Mathf.Clamp(velocity.x / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.y / 10f, -1f, 1f));
            sensor.AddObservation(Mathf.Clamp(velocity.z / 10f, -1f, 1f));

            bool isGrounded = Physics.RaycastNonAlloc(position + Vector3.up * 0.1f, Vector3.down, raycastHitsCache, 0.5f) > 0;
            sensor.AddObservation(isGrounded ? 0f : 1f);

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

            if (Physics.RaycastNonAlloc(position + Vector3.up * 0.5f, Vector3.down, raycastHitsCache, 20f) > 0)
            {
                sensor.AddObservation(raycastHitsCache[0].distance / 20f);
            }
            else
            {
                sensor.AddObservation(1f);
            }

            sensor.AddObservation(jumpPressed ? 1f : 0f);
            sensor.AddObservation(episodeTime / MAX_EPISODE_TIME);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            joystickInput = new Vector2(
                Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f),
                Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f)
            );

            jumpPressed = actions.DiscreteActions[0] == 1;
            kickPressed = false;
            stompPressed = false;

            if (targetGoal != null)
            {
                cameraLookDirection = (targetGoal.position - transform.position).normalized;
                cameraLookDirection.y = 0;
                if (cameraLookDirection.sqrMagnitude < 0.01f)
                    cameraLookDirection = Vector3.forward;
            }

            episodeTime += Time.fixedDeltaTime;
            Vector3 currentPos = transform.position;
            float currentDistance = GetDistanceToGoal();

            AddReward(-0.01f);

            float distanceDelta = previousDistanceToGoal - currentDistance;
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

            if (currentPos.y < startPosition.y - 3.0f)
            {
                AddReward(-5.0f);
                EndEpisode();
                return;
            }

            if (episodeTime >= MAX_EPISODE_TIME)
            {
                float progressRatio = 1.0f - (currentDistance / Mathf.Max(initialDistanceToGoal, 0.1f));
                AddReward(-5.0f + progressRatio * 3.0f);
                EndEpisode();
                return;
            }
        }

        private float GetDistanceToGoal()
        {
            if (targetGoal == null)
                return float.MaxValue;

            Vector3 a = transform.position;
            Vector3 b = targetGoal.position;
            return Vector3.Distance(new Vector3(a.x, 0, a.z), new Vector3(b.x, 0, b.z));
        }

        void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
            {
                AddReward(50f);
                EndEpisode();
            }
            else if (other.CompareTag("Checkpoint"))
            {
                BackupCheckpoint checkpoint = other.GetComponent<BackupCheckpoint>();
                if (checkpoint != null && !checkpoint.IsActivated)
                {
                    checkpoint.Activate();
                    AddReward(5f);
                    environment?.SetCheckpoint(checkpoint.transform.position);
                }
            }
        }

        public void SetEnvironment(BackupParkourEnvironment env)
        {
            environment = env;
        }

        public void SetTargetGoal(Transform goal)
        {
            targetGoal = goal;
        }
    }
}

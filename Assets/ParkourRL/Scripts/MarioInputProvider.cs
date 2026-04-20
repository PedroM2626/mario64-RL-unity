using UnityEngine;
using LibSM64;

namespace ParkourRL
{
    [RequireComponent(typeof(MarioRLAgent))]
    public class MarioInputProvider : SM64InputProvider
    {
        private MarioRLAgent agent;

        void Awake()
        {
            agent = GetComponent<MarioRLAgent>();
        }

        public override Vector3 GetCameraLookDirection()
        {
            if (agent != null)
                return agent.cameraLookDirection.normalized;
            return Vector3.forward;
        }

        public override Vector2 GetJoystickAxes()
        {
            if (agent != null)
                return agent.joystickInput;
            return Vector2.zero;
        }

        public override bool GetButtonHeld(Button button)
        {
            if (agent == null) return false;

            switch (button)
            {
                case Button.Jump: return agent.jumpPressed;
                case Button.Kick: return agent.kickPressed;
                case Button.Stomp: return agent.stompPressed;
                default: return false;
            }
        }
    }
}

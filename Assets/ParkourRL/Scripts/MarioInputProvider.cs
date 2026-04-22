using UnityEngine;
using LibSM64;

namespace ParkourRL
{
    public class MarioInputProvider : SM64InputProvider
    {
        private MarioRLAgent agent;
        private MarioCompetitiveAgent competitiveAgent;

        void Awake()
        {
            // Lazy init no primeiro uso
        }
        
        private void FindAgents()
        {
            if (agent == null)
                agent = GetComponent<MarioRLAgent>();
            if (competitiveAgent == null)
                competitiveAgent = GetComponent<MarioCompetitiveAgent>();
        }

        public override Vector3 GetCameraLookDirection()
        {
            FindAgents();
            if (agent != null)
                return agent.cameraLookDirection.normalized;
            if (competitiveAgent != null)
                return competitiveAgent.cameraLookDirection.normalized;
            return Vector3.forward;
        }

        public override Vector2 GetJoystickAxes()
        {
            FindAgents();
            if (agent != null)
                return agent.joystickInput;
            if (competitiveAgent != null)
                return competitiveAgent.joystickInput;
            return Vector2.zero;
        }

        public override bool GetButtonHeld(Button button)
        {
            FindAgents();

            if (agent != null)
            {
                switch (button)
                {
                    case Button.Jump: return agent.jumpPressed;
                    case Button.Kick: return agent.kickPressed;
                    case Button.Stomp: return agent.stompPressed;
                    default: return false;
                }
            }

            if (competitiveAgent != null)
            {
                switch (button)
                {
                    case Button.Jump: return competitiveAgent.jumpPressed;
                    case Button.Kick: return competitiveAgent.kickPressed;
                    case Button.Stomp: return competitiveAgent.stompPressed;
                    default: return false;
                }
            }

            return false;
        }
    }
}

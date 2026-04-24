using UnityEngine;
using LibSM64;

namespace ParkourRL.BackupSystem
{
    public class BackupMarioInputProvider : SM64InputProvider
    {
        private BackupMarioRLAgent agent;

        private BackupMarioRLAgent GetAgent()
        {
            if (agent == null)
                agent = GetComponent<BackupMarioRLAgent>();
            return agent;
        }

        public override Vector3 GetCameraLookDirection()
        {
            var a = GetAgent();
            if (a != null)
                return a.cameraLookDirection.normalized;
            return Vector3.forward;
        }

        public override Vector2 GetJoystickAxes()
        {
            var a = GetAgent();
            if (a != null)
                return a.joystickInput;
            return Vector2.zero;
        }

        public override bool GetButtonHeld(Button button)
        {
            var a = GetAgent();
            if (a == null)
                return false;

            switch (button)
            {
                case Button.Jump: return a.jumpPressed;
                case Button.Kick: return a.kickPressed;
                case Button.Stomp: return a.stompPressed;
                default: return false;
            }
        }
    }
}

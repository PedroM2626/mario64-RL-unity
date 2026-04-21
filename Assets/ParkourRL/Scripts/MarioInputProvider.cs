using UnityEngine;
using LibSM64;

namespace ParkourRL
{
    public class MarioInputProvider : SM64InputProvider
    {
        private MarioRLAgent agent;

        void Awake()
        {
            // Não buscar aqui - fazer lazy no primeiro uso
            // O MarioRLAgent pode ser adicionado depois deste componente
        }
        
        MarioRLAgent GetAgent()
        {
            if (agent == null)
                agent = GetComponent<MarioRLAgent>();
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
            {
                Vector2 input = a.joystickInput;
                // Log quando há input significativo (para debug de movimento)
                if (input.magnitude > 0.5f && Time.frameCount % 60 == 0)
                {
                    Debug.Log($"[MarioInput] Enviando input: {input}");
                }
                return input;
            }
            return Vector2.zero;
        }

        public override bool GetButtonHeld(Button button)
        {
            var a = GetAgent();
            if (a == null) return false;

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

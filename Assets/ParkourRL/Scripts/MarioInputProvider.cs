using UnityEngine;
using LibSM64;
using System;
using System.Reflection;

namespace ParkourRL
{
    public class MarioInputProvider : SM64InputProvider
    {
        private MarioRLAgent agent;
        private MarioCompetitiveAgent competitiveAgent;
        private TeamBattleAgent teamBattleAgent;
        private Component backupAgent;
        private Type backupAgentType;

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
            if (teamBattleAgent == null)
                teamBattleAgent = GetComponent<TeamBattleAgent>();
            if (backupAgent == null)
            {
                if (backupAgentType == null)
                    backupAgentType = Type.GetType("ParkourRL.BackupSystem.BackupMarioRLAgent, Assembly-CSharp");

                if (backupAgentType != null)
                    backupAgent = GetComponent(backupAgentType);
            }
        }

        private Vector2 ReadBackupVector2(string fieldName)
        {
            if (backupAgent == null)
                return Vector2.zero;

            FieldInfo f = backupAgent.GetType().GetField(fieldName, BindingFlags.Public | BindingFlags.Instance);
            if (f == null)
                return Vector2.zero;

            object value = f.GetValue(backupAgent);
            return value is Vector2 v ? v : Vector2.zero;
        }

        private Vector3 ReadBackupVector3(string fieldName)
        {
            if (backupAgent == null)
                return Vector3.forward;

            FieldInfo f = backupAgent.GetType().GetField(fieldName, BindingFlags.Public | BindingFlags.Instance);
            if (f == null)
                return Vector3.forward;

            object value = f.GetValue(backupAgent);
            return value is Vector3 v ? v : Vector3.forward;
        }

        private bool ReadBackupBool(string fieldName)
        {
            if (backupAgent == null)
                return false;

            FieldInfo f = backupAgent.GetType().GetField(fieldName, BindingFlags.Public | BindingFlags.Instance);
            if (f == null)
                return false;

            object value = f.GetValue(backupAgent);
            return value is bool b && b;
        }

        public override Vector3 GetCameraLookDirection()
        {
            FindAgents();
            if (agent != null)
                return agent.cameraLookDirection.normalized;
            if (competitiveAgent != null)
                return competitiveAgent.cameraLookDirection.normalized;
            if (teamBattleAgent != null)
                return teamBattleAgent.cameraLookDirection.normalized;
            if (backupAgent != null)
                return ReadBackupVector3("cameraLookDirection").normalized;
            return Vector3.forward;
        }

        public override Vector2 GetJoystickAxes()
        {
            FindAgents();
            if (agent != null)
                return agent.joystickInput;
            if (competitiveAgent != null)
                return competitiveAgent.joystickInput;
            if (teamBattleAgent != null)
                return teamBattleAgent.joystickInput;
            if (backupAgent != null)
                return ReadBackupVector2("joystickInput");
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

            if (teamBattleAgent != null)
            {
                switch (button)
                {
                    case Button.Jump: return teamBattleAgent.jumpPressed;
                    case Button.Kick: return teamBattleAgent.kickPressed;
                    case Button.Stomp: return teamBattleAgent.stompPressed;
                    default: return false;
                }
            }

            if (backupAgent != null)
            {
                switch (button)
                {
                    case Button.Jump: return ReadBackupBool("jumpPressed");
                    case Button.Kick: return ReadBackupBool("kickPressed");
                    case Button.Stomp: return ReadBackupBool("stompPressed");
                    default: return false;
                }
            }

            return false;
        }
    }
}

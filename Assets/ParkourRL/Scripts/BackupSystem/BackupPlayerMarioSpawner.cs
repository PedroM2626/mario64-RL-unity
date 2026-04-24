using UnityEngine;
using LibSM64;

namespace ParkourRL.BackupSystem
{
    public class BackupPlayerMarioSpawner : MonoBehaviour
    {
        [SerializeField] private Vector3 spawnPosition = new Vector3(0, 2, 0);
        [SerializeField] private Material marioMaterial;

        private GameObject playerMario;
        private Camera playerCam;

        void Start()
        {
            Invoke(nameof(SpawnPlayerMario), 0.8f);
        }

        void SpawnPlayerMario()
        {
            GameObject camObj = new GameObject("BackupPlayerMarioCamera");
            playerCam = camObj.AddComponent<Camera>();
            playerCam.tag = "Untagged";
            camObj.transform.position = spawnPosition + new Vector3(0, 5, -8);
            camObj.transform.LookAt(spawnPosition);

            playerMario = new GameObject("BackupPlayerMario_DEBUG");
            playerMario.SetActive(false);
            playerMario.transform.position = spawnPosition;

            var inputProvider = playerMario.AddComponent<BackupPlayerTestInputProvider>();
            inputProvider.cameraTransform = camObj.transform;

            SM64Mario sm64Mario = playerMario.AddComponent<SM64Mario>();
            if (marioMaterial != null)
            {
                var materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, marioMaterial);
            }

            playerMario.SetActive(true);
        }

        void LateUpdate()
        {
            if (playerMario != null && playerCam != null)
            {
                Vector3 targetPos = playerMario.transform.position + new Vector3(0, 5, -8);
                playerCam.transform.position = Vector3.Lerp(playerCam.transform.position, targetPos, 5f * Time.deltaTime);
                playerCam.transform.LookAt(playerMario.transform.position + Vector3.up * 1f);
            }
        }
    }

    public class BackupPlayerTestInputProvider : SM64InputProvider
    {
        public Transform cameraTransform;

        public override Vector3 GetCameraLookDirection()
        {
            if (cameraTransform != null)
                return cameraTransform.forward;
            return Vector3.forward;
        }

        public override Vector2 GetJoystickAxes()
        {
            return new Vector2(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"));
        }

        public override bool GetButtonHeld(Button button)
        {
            switch (button)
            {
                case Button.Jump: return Input.GetButton("Jump");
                case Button.Kick: return Input.GetMouseButton(0);
                case Button.Stomp: return Input.GetKey(KeyCode.LeftShift);
            }
            return false;
        }
    }
}

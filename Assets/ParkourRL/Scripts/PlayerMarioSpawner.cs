using UnityEngine;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Spawna um Mario controlavel pelo teclado para testar manualmente o nivel.
    /// Adicione este script a um GameObject vazio na cena ParkourTraining.
    /// WASD/Setas = mover, Espaco = pular.
    /// </summary>
    public class PlayerMarioSpawner : MonoBehaviour
    {
        [SerializeField] private Vector3 spawnPosition = new Vector3(0, 2, 0);
        [SerializeField] private Material marioMaterial;
        
        private GameObject playerMario;
        private Camera playerCam;

        void Start()
        {
            // Esperar meio segundo para tudo inicializar
            Invoke(nameof(SpawnPlayerMario), 0.8f);
        }

        void SpawnPlayerMario()
        {
            // Carregar material do prefab se nao atribuido
            if (marioMaterial == null)
            {
                #if UNITY_EDITOR
                GameObject prefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Mario.prefab");
                if (prefab != null)
                {
                    SM64Mario prefabMario = prefab.GetComponent<SM64Mario>();
                    if (prefabMario != null)
                    {
                        var matField = typeof(SM64Mario).GetField("material", 
                            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        if (matField != null)
                            marioMaterial = matField.GetValue(prefabMario) as Material;
                    }
                }
                #endif
            }

            // Criar camera que segue o Mario
            GameObject camObj = new GameObject("PlayerMarioCamera");
            playerCam = camObj.AddComponent<Camera>();
            playerCam.tag = "Untagged"; // Nao marcar como MainCamera
            camObj.transform.position = spawnPosition + new Vector3(0, 5, -8);
            camObj.transform.LookAt(spawnPosition);

            // Criar Mario
            playerMario = new GameObject("PlayerMario_DEBUG");
            playerMario.SetActive(false);
            playerMario.transform.position = spawnPosition;

            // Input provider para teclado (usa a camera criada)
            var inputProvider = playerMario.AddComponent<PlayerTestInputProvider>();
            inputProvider.cameraTransform = camObj.transform;

            // SM64Mario
            SM64Mario sm64Mario = playerMario.AddComponent<SM64Mario>();
            if (marioMaterial != null)
            {
                var materialField = typeof(SM64Mario).GetField("material", 
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, marioMaterial);
            }

            playerMario.SetActive(true);
            Debug.Log($"[PlayerMario] Mario de TESTE spawnado em {spawnPosition}. Use WASD + Espaco para controlar.");
        }

        void LateUpdate()
        {
            if (playerMario != null && playerCam != null)
            {
                // Camera segue o Mario
                Vector3 targetPos = playerMario.transform.position + new Vector3(0, 5, -8);
                playerCam.transform.position = Vector3.Lerp(playerCam.transform.position, targetPos, 5f * Time.deltaTime);
                playerCam.transform.LookAt(playerMario.transform.position + Vector3.up * 1f);
            }
        }
    }

    /// <summary>
    /// Input provider para teclado -- controle direto do Mario via WASD + Espaco.
    /// </summary>
    public class PlayerTestInputProvider : SM64InputProvider
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
                case Button.Jump:  return Input.GetButton("Jump");
                case Button.Kick:  return Input.GetMouseButton(0);
                case Button.Stomp: return Input.GetKey(KeyCode.LeftShift);
            }
            return false;
        }
    }
}

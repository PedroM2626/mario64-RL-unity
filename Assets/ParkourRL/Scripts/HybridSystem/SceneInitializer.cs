using UnityEngine;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Inicializa a cena com GameObjects mínimos para o treinamento híbrido.
    /// Cria cubes simples representando Mario player e AI.
    /// </summary>
    public class SceneInitializer : MonoBehaviour
    {
        [SerializeField] private bool autoInitializeOnStart = true;
        [SerializeField] private Transform playerSpawnPoint;
        [SerializeField] private Transform aiSpawnPoint;
        [SerializeField] private Transform goalPosition;

        private void Start()
        {
            if (autoInitializeOnStart)
            {
                InitializeScene();
            }
        }

        [ContextMenu("Initialize Scene")]
        public void InitializeScene()
        {
            Debug.Log("[SceneInitializer] Inicializando cena...");

            // Criar spawn points se não existirem
            if (playerSpawnPoint == null)
            {
                GameObject playerSpawn = new GameObject("PlayerSpawn");
                playerSpawn.transform.position = new Vector3(-2, 1, 0);
                playerSpawnPoint = playerSpawn.transform;
            }

            if (aiSpawnPoint == null)
            {
                GameObject aiSpawn = new GameObject("AISpawn");
                aiSpawn.transform.position = new Vector3(2, 1, 0);
                aiSpawnPoint = aiSpawn.transform;
            }

            if (goalPosition == null)
            {
                GameObject goal = new GameObject("Goal");
                goal.transform.position = new Vector3(0, 1, 20);
                goalPosition = goal.transform;
            }

            // Criar Player Mario (Cube vermelho)
            if (GameObject.Find("PlayerMario") == null)
            {
                CreatePlayerMario();
            }

            // Criar AI Mario (Cube azul)
            if (GameObject.Find("AIMario") == null)
            {
                CreateAIMario();
            }

            // Criar plataformas
            CreatePlatforms();

            Debug.Log("[SceneInitializer] Cena inicializada com sucesso!");
        }

        private void CreatePlayerMario()
        {
            GameObject playerObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            playerObj.name = "PlayerMario";
            playerObj.transform.position = playerSpawnPoint.position;
            playerObj.transform.localScale = new Vector3(0.8f, 1.6f, 0.8f);

            // Material vermelho para player
            Renderer renderer = playerObj.GetComponent<Renderer>();
            Material playerMat = new Material(Shader.Find("Standard"));
            playerMat.color = Color.red;
            renderer.material = playerMat;

            // Remover collider primitivo e adicionar CapsuleCollider
            Collider primCollider = playerObj.GetComponent<Collider>();
            if (primCollider != null) Destroy(primCollider);

            CapsuleCollider capsule = playerObj.AddComponent<CapsuleCollider>();
            capsule.height = 1.6f;
            capsule.radius = 0.4f;

            // Rigidbody
            Rigidbody rb = playerObj.AddComponent<Rigidbody>();
            rb.mass = 1f;
            rb.drag = 0.1f;
            rb.constraints = RigidbodyConstraints.FreezeRotation;
            rb.useGravity = true;

            // Adicionar agent
            MarioHybridAgent agent = playerObj.AddComponent<MarioHybridAgent>();
            
            Debug.Log("[SceneInitializer] Player Mario criado em " + playerObj.transform.position);
        }

        private void CreateAIMario()
        {
            GameObject aiObj = GameObject.CreatePrimitive(PrimitiveType.Cube);
            aiObj.name = "AIMario";
            aiObj.transform.position = aiSpawnPoint.position;
            aiObj.transform.localScale = new Vector3(0.8f, 1.6f, 0.8f);

            // Material azul para AI
            Renderer renderer = aiObj.GetComponent<Renderer>();
            Material aiMat = new Material(Shader.Find("Standard"));
            aiMat.color = Color.blue;
            renderer.material = aiMat;

            // Remover collider primitivo
            Collider primCollider = aiObj.GetComponent<Collider>();
            if (primCollider != null) Destroy(primCollider);

            CapsuleCollider capsule = aiObj.AddComponent<CapsuleCollider>();
            capsule.height = 1.6f;
            capsule.radius = 0.4f;

            // Rigidbody
            Rigidbody rb = aiObj.AddComponent<Rigidbody>();
            rb.mass = 1f;
            rb.drag = 0.1f;
            rb.constraints = RigidbodyConstraints.FreezeRotation;
            rb.useGravity = true;

            // Adicionar agent
            MarioHybridAgent agent = aiObj.AddComponent<MarioHybridAgent>();

            Debug.Log("[SceneInitializer] AI Mario criado em " + aiObj.transform.position);
        }

        private void CreatePlatforms()
        {
            int platformCount = 5;
            for (int i = 0; i < platformCount; i++)
            {
                // Verificar se plataforma já existe
                if (GameObject.Find("Platform_" + i) != null) continue;

                GameObject platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
                platform.name = "Platform_" + i;
                platform.transform.position = new Vector3(0, 0.5f + i * 2, 5 + i * 3);
                platform.transform.localScale = new Vector3(4, 0.5f, 4);

                // Material verde
                Renderer renderer = platform.GetComponent<Renderer>();
                Material mat = new Material(Shader.Find("Standard"));
                mat.color = Color.green;
                renderer.material = mat;

                // Remover Rigidbody (plataforma estática)
                Rigidbody rb = platform.GetComponent<Rigidbody>();
                if (rb != null) Destroy(rb);
            }

            Debug.Log("[SceneInitializer] " + platformCount + " plataformas criadas");
        }
    }
}

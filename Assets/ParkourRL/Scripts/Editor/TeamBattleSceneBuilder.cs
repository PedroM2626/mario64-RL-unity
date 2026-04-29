using UnityEngine;
using LibSM64;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

namespace ParkourRL
{
    /// <summary>
    /// Script auxiliar para construir rapidamente a cena TeamBattle.
    /// Pode ser executado via Editor ou via script.
    /// </summary>
    public class TeamBattleSceneBuilder : MonoBehaviour
    {
#if UNITY_EDITOR

        [ContextMenu("Build Team Battle Scene")]
        public void BuildTeamBattleScene()
        {
            Debug.Log("[TeamBattleBuilder] Starting Team Battle scene construction...");

            // Limpar cena existente (exceto este script)
            ClearScene();

            // Criar arena
            CreateArenaPlatform();

            // Criar spawn points
            CreateTeamSpawnPoints();

            // Criar arena center marker
            CreateArenaCenterMarker();

            // Criar manager
            CreateTeamBattleManager();

            // Criar lighting
            CreateLighting();

            Debug.Log("[TeamBattleBuilder] Team Battle scene built successfully!");
            EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        }

        private void ClearScene()
        {
            // Remover todos os objetos exceto este
            GameObject[] allObjects = FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj != gameObject)
                {
                    DestroyImmediate(obj);
                }
            }
        }

        private void CreateArenaPlatform()
        {
            // Arena principal
            GameObject arenaPlatform = new GameObject("ArenaPlatform");
            arenaPlatform.transform.position = Vector3.zero;
            arenaPlatform.transform.localScale = new Vector3(40, 1, 40);

            // Adicionar cubo visual
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            MeshFilter meshFilter = cube.GetComponent<MeshFilter>();
            MeshRenderer meshRenderer = cube.GetComponent<MeshRenderer>();
            Collider collider = cube.GetComponent<Collider>();
            DestroyImmediate(collider);

            cube.transform.SetParent(arenaPlatform.transform);
            cube.transform.localPosition = Vector3.zero;
            cube.transform.localScale = Vector3.one;
            cube.name = "Geometry";

            // Material
            Material platformMat = new Material(Shader.Find("Standard"));
            platformMat.color = new Color(0.7f, 0.6f, 0.5f);
            meshRenderer.material = platformMat;

            // Add required components
            BoxCollider bc = arenaPlatform.AddComponent<BoxCollider>();
            bc.size = Vector3.one;

            SM64StaticTerrain terrain = arenaPlatform.AddComponent<SM64StaticTerrain>();

            Debug.Log("[TeamBattleBuilder] Arena criada!");
        }

        private void CreateTeamSpawnPoints()
        {
            // Team A Spawns
            GameObject teamASpawns = new GameObject("TeamASpawns");
            for (int i = 0; i < 5; i++)
            {
                GameObject spawn = new GameObject($"Spawn_A_{i}");
                spawn.transform.SetParent(teamASpawns.transform);
                spawn.transform.position = new Vector3(-18, 2, -10 + i * 5);
            }

            // Team B Spawns
            GameObject teamBSpawns = new GameObject("TeamBSpawns");
            for (int i = 0; i < 5; i++)
            {
                GameObject spawn = new GameObject($"Spawn_B_{i}");
                spawn.transform.SetParent(teamBSpawns.transform);
                spawn.transform.position = new Vector3(18, 2, -10 + i * 5);
            }

            Debug.Log("[TeamBattleBuilder] Spawn points criados!");
        }

        private void CreateArenaCenterMarker()
        {
            GameObject centerMarker = new GameObject("ArenaCenter");
            centerMarker.transform.position = Vector3.zero;
            Debug.Log("[TeamBattleBuilder] Arena center marker criado!");
        }

        private void CreateTeamBattleManager()
        {
            GameObject manager = new GameObject("TeamBattleManager");
            TeamBattleEnvironment env = manager.AddComponent<TeamBattleEnvironment>();

            // Configure references
            Transform[] teamASpawns = new Transform[5];
            Transform[] teamBSpawns = new Transform[5];

            GameObject teamASpawnsObj = GameObject.Find("TeamASpawns");
            GameObject teamBSpawnsObj = GameObject.Find("TeamBSpawns");
            GameObject arenaCenter = GameObject.Find("ArenaCenter");

            if (teamASpawnsObj != null)
            {
                for (int i = 0; i < 5; i++)
                {
                    Transform child = teamASpawnsObj.transform.GetChild(i);
                    if (child != null)
                        teamASpawns[i] = child;
                }
            }

            if (teamBSpawnsObj != null)
            {
                for (int i = 0; i < 5; i++)
                {
                    Transform child = teamBSpawnsObj.transform.GetChild(i);
                    if (child != null)
                        teamBSpawns[i] = child;
                }
            }

            // Usar reflection para setar campos privados
            var teamASpawnsField = typeof(TeamBattleEnvironment).GetField("teamASpawnPoints",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var teamBSpawnsField = typeof(TeamBattleEnvironment).GetField("teamBSpawnPoints",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var arenaCenterField = typeof(TeamBattleEnvironment).GetField("arenaCenter",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (teamASpawnsField != null)
                teamASpawnsField.SetValue(env, teamASpawns);
            if (teamBSpawnsField != null)
                teamBSpawnsField.SetValue(env, teamBSpawns);
            if (arenaCenterField != null && arenaCenter != null)
                arenaCenterField.SetValue(env, arenaCenter.transform);

            // Carregar Mario prefab
            GameObject marioPrefab = AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Mario.prefab");
            var marioPrefabField = typeof(TeamBattleEnvironment).GetField("marioPrefab",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (marioPrefabField != null)
                marioPrefabField.SetValue(env, marioPrefab);

            Debug.Log("[TeamBattleBuilder] TeamBattleManager criado e configurado!");
        }

        private void CreateLighting()
        {
            // Luz direcional
            GameObject light = new GameObject("Directional Light");
            Light lightComponent = light.AddComponent<Light>();
            lightComponent.type = LightType.Directional;
            lightComponent.intensity = 1.5f;
            light.transform.position = new Vector3(10, 10, 10);
            light.transform.rotation = Quaternion.Euler(45, -30, 0);

            Debug.Log("[TeamBattleBuilder] Lighting criado!");
        }

        /// <summary>
        /// Static helper para criar a cena sem precisar de componente
        /// </summary>
        [MenuItem("Assets/Create/Team Battle Scene")]
        public static void CreateTeamBattleSceneMenu()
        {
            // Criar nova cena
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Criar builder object
            GameObject builderObj = new GameObject("_SceneBuilder");
            TeamBattleSceneBuilder builder = builderObj.AddComponent<TeamBattleSceneBuilder>();

            // Executar build
            builder.BuildTeamBattleScene();

            // Salvar cena
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(),
                "Assets/ParkourRL/Scenes/TeamBattle.unity");

            // Remover builder
            DestroyImmediate(builderObj);

            Debug.Log("[TeamBattleBuilder] TeamBattle.unity criada e salva!");
        }

#endif
    }
}

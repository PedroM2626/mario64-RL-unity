using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.IO;
using LibSM64;

#if UNITY_EDITOR
namespace ParkourRL.HybridSystem.Editor
{
    /// <summary>
    /// Editor script para construir automaticamente a cena HybridTraining.
    /// Execute via menu: ParkourRL > Build Hybrid Scene
    /// </summary>
    public class HybridSceneBuilder
    {

        
        static void CreateParkourLevel()
        {
            // Criar plataformas baseadas no README
            var platformData = new (string name, Vector3 pos, Vector3 scale)[]
            {
                ("StartPlatform", new Vector3(0, -0.5f, 0), new Vector3(6, 1, 6)),
                ("Platform_1", new Vector3(5, -0.5f, 0), new Vector3(3, 1, 3)),
                ("Platform_2", new Vector3(9.5f, -0.5f, 0), new Vector3(3, 1, 3)),
                ("Platform_3", new Vector3(14, 0, 0), new Vector3(3, 1, 3)),
                ("GoalPlatform", new Vector3(19, 0, 0), new Vector3(5, 1, 5))
            };
            
            GameObject levelRoot = new GameObject("ParkourLevel");
            
            foreach (var data in platformData)
            {
                GameObject platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
                platform.name = data.name;
                platform.transform.position = data.pos;
                platform.transform.localScale = data.scale;
                platform.transform.parent = levelRoot.transform;
                
                // Adicionar SM64StaticTerrain
                platform.AddComponent<SM64StaticTerrain>();
                
                // Configurar collider
                var collider = platform.GetComponent<BoxCollider>();
                if (collider != null)
                {
                    collider.isTrigger = (data.name == "GoalPlatform");
                }
                
                // Adicionar tag de goal na última plataforma
                if (data.name == "GoalPlatform")
                {
                    platform.tag = "Goal";
                    
                    // Adicionar visual de goal
                    var goalVisual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    goalVisual.name = "GoalVisual";
                    goalVisual.transform.position = data.pos + Vector3.up * 2f;
                    goalVisual.transform.localScale = Vector3.one * 2f;
                    goalVisual.transform.parent = platform.transform;
                    
                    var renderer = goalVisual.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        renderer.sharedMaterial.color = Color.yellow;
                    }
                }
            }
            
            Debug.Log("[HybridSceneBuilder] Plataformas criadas");
        }
        
        static void CreateSpawnPoints()
        {
            GameObject spawnRoot = new GameObject("SpawnPoints");
            
            // Player spawn (esquerda)
            GameObject playerSpawn = new GameObject("PlayerSpawn");
            playerSpawn.transform.position = new Vector3(-7, 2, 0);
            playerSpawn.transform.parent = spawnRoot.transform;
            
            // AI spawn (direita)
            GameObject aiSpawn = new GameObject("AISpawn");
            aiSpawn.transform.position = new Vector3(7, 2, 0);
            aiSpawn.transform.parent = spawnRoot.transform;
            
            // Shared spawn points array (para fallback)
            GameObject sharedSpawns = new GameObject("SharedSpawnPoints");
            sharedSpawns.transform.parent = spawnRoot.transform;
            
            GameObject shared1 = new GameObject("Shared_0");
            shared1.transform.position = new Vector3(-7, 2, 0);
            shared1.transform.parent = sharedSpawns.transform;
            
            GameObject shared2 = new GameObject("Shared_1");
            shared2.transform.position = new Vector3(7, 2, 0);
            shared2.transform.parent = sharedSpawns.transform;
            
            Debug.Log("[HybridSceneBuilder] Spawn points criados");
        }
        
        static void CreateHybridEnvironment()
        {
            GameObject envObj = new GameObject("HybridEnvironment");
            
            // HybridParkourEnvironment
            var env = envObj.AddComponent<HybridParkourEnvironment>();
            
            // Configurar referências (serão preenchidas manualmente ou via inspector)
            envObj.AddComponent<HybridDataRecorder>();
            envObj.AddComponent<HybridTrainingManager>();
            
            // Configurar parâmetros do ambiente
            SerializedObject envSerialized = new SerializedObject(env);
            envSerialized.FindProperty("marioSpacing").floatValue = 15f;
            envSerialized.FindProperty("syncResets").boolValue = true;
            envSerialized.FindProperty("showComparisonUI").boolValue = true;
            envSerialized.ApplyModifiedProperties();
            
            Debug.Log("[HybridSceneBuilder] Ambiente híbrido criado");
        }
        
        static void CreateCameras()
        {
            GameObject camerasRoot = new GameObject("Cameras");
            
            // Player Camera (metade esquerda)
            GameObject playerCamObj = new GameObject("PlayerCamera");
            Camera playerCam = playerCamObj.AddComponent<Camera>();
            playerCam.rect = new Rect(0, 0, 0.5f, 1f);
            playerCam.clearFlags = CameraClearFlags.Skybox;
            playerCam.backgroundColor = new Color(0.5f, 0.7f, 1f);
            playerCamObj.transform.position = new Vector3(-7, 5, -8);
            playerCamObj.transform.LookAt(new Vector3(-7, 0, 0));
            playerCamObj.transform.parent = camerasRoot.transform;
            
            // AI Camera (metade direita)
            GameObject aiCamObj = new GameObject("AICamera");
            Camera aiCam = aiCamObj.AddComponent<Camera>();
            aiCam.rect = new Rect(0.5f, 0, 0.5f, 1f);
            aiCam.clearFlags = CameraClearFlags.Skybox;
            aiCam.backgroundColor = new Color(0.7f, 0.5f, 1f);
            aiCamObj.transform.position = new Vector3(7, 5, -8);
            aiCamObj.transform.LookAt(new Vector3(7, 0, 0));
            aiCamObj.transform.parent = camerasRoot.transform;
            
            Debug.Log("[HybridSceneBuilder] Câmeras criadas");
        }
        
        static void CreateUI()
        {
            // Criar Canvas
            GameObject canvasObj = new GameObject("HybridUI");
            Canvas canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasObj.AddComponent<CanvasScaler>().uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            canvasObj.AddComponent<GraphicRaycaster>();
            
            // Mode Text (topo central)
            GameObject modeTextObj = new GameObject("ModeText");
            modeTextObj.transform.parent = canvasObj.transform;
            UnityEngine.UI.Text modeText = modeTextObj.AddComponent<UnityEngine.UI.Text>();
            modeText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            modeText.fontSize = 24;
            modeText.alignment = TextAnchor.UpperCenter;
            modeText.color = Color.white;
            modeText.text = "Mode: Training";
            
            RectTransform modeRect = modeTextObj.GetComponent<RectTransform>();
            modeRect.anchorMin = new Vector2(0.5f, 1f);
            modeRect.anchorMax = new Vector2(0.5f, 1f);
            modeRect.pivot = new Vector2(0.5f, 1f);
            modeRect.anchoredPosition = new Vector2(0, -20);
            modeRect.sizeDelta = new Vector2(300, 40);
            
            // Info Text (abaixo do modo)
            GameObject infoTextObj = new GameObject("InfoText");
            infoTextObj.transform.parent = canvasObj.transform;
            UnityEngine.UI.Text infoText = infoTextObj.AddComponent<UnityEngine.UI.Text>();
            infoText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            infoText.fontSize = 16;
            infoText.alignment = TextAnchor.UpperCenter;
            infoText.color = new Color(0.8f, 0.8f, 0.8f);
            infoText.text = "M: Toggle Mode | P: Pause | WASD+Space: Control Player";
            
            RectTransform infoRect = infoTextObj.GetComponent<RectTransform>();
            infoRect.anchorMin = new Vector2(0.5f, 1f);
            infoRect.anchorMax = new Vector2(0.5f, 1f);
            infoRect.pivot = new Vector2(0.5f, 1f);
            infoRect.anchoredPosition = new Vector2(0, -60);
            infoRect.sizeDelta = new Vector2(600, 30);
            
            // Stats Panel (canto esquerdo)
            GameObject statsPanelObj = new GameObject("StatsPanel");
            statsPanelObj.transform.parent = canvasObj.transform;
            UnityEngine.UI.Image statsImage = statsPanelObj.AddComponent<UnityEngine.UI.Image>();
            statsImage.color = new Color(0, 0, 0, 0.5f);
            
            RectTransform statsRect = statsPanelObj.GetComponent<RectTransform>();
            statsRect.anchorMin = new Vector2(0, 1);
            statsRect.anchorMax = new Vector2(0, 1);
            statsRect.pivot = new Vector2(0, 1);
            statsRect.anchoredPosition = new Vector2(20, -20);
            statsRect.sizeDelta = new Vector2(250, 150);
            
            // Stats Text
            GameObject statsTextObj = new GameObject("StatsText");
            statsTextObj.transform.parent = statsPanelObj.transform;
            UnityEngine.UI.Text statsText = statsTextObj.AddComponent<UnityEngine.UI.Text>();
            statsText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            statsText.fontSize = 14;
            statsText.alignment = TextAnchor.UpperLeft;
            statsText.color = Color.white;
            statsText.text = "Player (Green): 0 completes\nPlayer Best: --\n\nAI (Blue): 0 completes\nAI Best: --";
            
            RectTransform statsTextRect = statsTextObj.GetComponent<RectTransform>();
            statsTextRect.anchorMin = Vector2.zero;
            statsTextRect.anchorMax = Vector2.one;
            statsTextRect.offsetMin = new Vector2(10, 10);
            statsTextRect.offsetMax = new Vector2(-10, -10);
            
            Debug.Log("[HybridSceneBuilder] UI criada");
        }
        
        static void CreateLighting()
        {
            // Luz direcional
            GameObject lightObj = new GameObject("Directional Light");
            Light light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1f;
            light.color = Color.white;
            light.shadows = LightShadows.Soft;
            lightObj.transform.rotation = Quaternion.Euler(50, -30, 0);
            
            // Ambient light já configurado no início
            
            Debug.Log("[HybridSceneBuilder] Lighting criado");
        }
        
        /// <summary>
        /// Helper: Conecta as referências de um HybridParkourEnvironment existente
        /// </summary>
        public static void SetupReferences()
        {
            // Este método ajuda a conectar as referências após a cena ser criada
            // Útil se o usuário quiser reconectar componentes
            
            var env = Object.FindObjectOfType<HybridParkourEnvironment>();
            if (env == null)
            {
                EditorUtility.DisplayDialog("Erro", "HybridParkourEnvironment não encontrado!", "OK");
                return;
            }
            
            SerializedObject envSO = new SerializedObject(env);
            
            // Encontrar e atribuir goal
            var goal = GameObject.FindWithTag("Goal");
            if (goal != null)
            {
                envSO.FindProperty("goal").objectReferenceValue = goal.transform;
            }
            
            // Encontrar spawn points
            var playerSpawn = GameObject.Find("PlayerSpawn");
            var aiSpawn = GameObject.Find("AISpawn");
            var sharedSpawns = GameObject.Find("SharedSpawnPoints");
            
            if (playerSpawn != null)
                envSO.FindProperty("playerSpawnPoint").objectReferenceValue = playerSpawn.transform;
            if (aiSpawn != null)
                envSO.FindProperty("aiSpawnPoint").objectReferenceValue = aiSpawn.transform;
            if (sharedSpawns != null)
            {
                // Converter children para array
                var transforms = new System.Collections.Generic.List<Transform>();
                foreach (Transform child in sharedSpawns.transform)
                {
                    transforms.Add(child);
                }
                // Nota: Atribuição de arrays em SerializedProperty é complexa
                // Usuário pode precisar fazer manualmente no Inspector
            }
            
            // Conectar DataRecorder
            var recorder = env.GetComponent<HybridDataRecorder>();
            envSO.FindProperty("dataRecorder").objectReferenceValue = recorder;
            
            // Conectar TrainingManager
            var manager = env.GetComponent<HybridTrainingManager>();
            envSO.FindProperty("trainingManager").objectReferenceValue = manager;
            
            envSO.ApplyModifiedProperties();
            
            EditorUtility.DisplayDialog("Setup Complete", 
                "Referências configuradas!\n\n" +
                "Verifique o Inspector do HybridEnvironment:\n" +
                "- Goal atribuído\n" +
                "- Spawn points atribuídos\n" +
                "- Recorder e Manager conectados", "OK");
        }
    }
}
#endif

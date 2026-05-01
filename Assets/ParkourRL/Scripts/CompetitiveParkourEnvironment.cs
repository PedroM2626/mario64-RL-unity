using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Competitive environment for 3 simultaneous Marios (PPO, SAC, DQN) on the same map.
    /// Colors and textures are configurable in the Inspector and applied at runtime.
    /// </summary>
    public class CompetitiveParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Material (Opcional)")]
        [Tooltip("Optional base material. If not assigned, uses marioPrefab as fallback.")]
        [SerializeField] private Material baseMarioMaterial;
        [Tooltip("Optional prefab to obtain fallback material.")]
        [SerializeField] private GameObject marioPrefab;

        [Header("Cores dos Modelos (Inspector)")]
        [Tooltip("PPO Mario color (Team 0).")]
        [SerializeField] private Color ppoColor = new Color(1f, 0.2f, 0.2f, 1f);
        [Tooltip("SAC Mario color (Team 1).")]
        [SerializeField] private Color sacColor = new Color(0.2f, 0.4f, 1f, 1f);
        [Tooltip("DQN Mario color (Team 2).")]
        [SerializeField] private Color dqnColor = new Color(0.2f, 0.9f, 0.2f, 1f);

        [Header("Level")]
        [SerializeField] private Transform[] spawnPoints;
        [SerializeField] private Transform goal;

        [Header("Competition")]
        [SerializeField] private int marioCount = 3;

        private string[] behaviorNames = new string[] { "MarioParkourPPO", "MarioParkourSAC", "MarioParkourDQN" };
        private Color[] teamColors;

        private List<MarioCompetitiveAgent> agents = new List<MarioCompetitiveAgent>();
        private int finishOrder = 0;
        
        // Scoreboard
        private Dictionary<string, int> scores = new Dictionary<string, int>();
        private Text scoreText;

        // Runtime materials
        private Material ppoRuntimeMaterial;
        private Material sacRuntimeMaterial;
        private Material dqnRuntimeMaterial;

        void Awake()
        {
            teamColors = new Color[] { ppoColor, sacColor, dqnColor };
            SetupScoreboard();
        }

        void Start()
        {
            // Garantir MeshColliders
            EnsureAllMeshColliders();

            // Recarregar terreno SM64
            SM64Context.RefreshStaticTerrain();

            // Spawnar todos os Marios
            Invoke(nameof(SpawnAllMarios), 0.5f);
        }

        private void SetupScoreboard()
        {
            scores["PPO"] = 0;
            scores["SAC"] = 0;
            scores["DQN"] = 0;

            GameObject canvasObj = new GameObject("ScoreCanvas");
            Canvas canvas = canvasObj.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasObj.AddComponent<CanvasScaler>();
            canvasObj.AddComponent<GraphicRaycaster>();

            GameObject textObj = new GameObject("ScoreText");
            textObj.transform.SetParent(canvasObj.transform);
            scoreText = textObj.AddComponent<Text>();
            try
            {
                scoreText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            }
            catch
            {
                scoreText.font = Font.CreateDynamicFontFromOSFont("Arial", 16);
            }
            scoreText.fontSize = 26;
            scoreText.color = Color.white;
            scoreText.alignment = TextAnchor.UpperLeft;
            
            RectTransform rt = scoreText.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0, 1);
            rt.anchorMax = new Vector2(0, 1);
            rt.pivot = new Vector2(0, 1);
            rt.anchoredPosition = new Vector2(20, -20);
            rt.sizeDelta = new Vector2(350, 220);

            UpdateScoreboardUI();
        }

        private void UpdateScoreboardUI()
        {
            if (scoreText != null)
            {
                string ppoHex = ColorUtility.ToHtmlStringRGB(ppoColor);
                string sacHex = ColorUtility.ToHtmlStringRGB(sacColor);
                string dqnHex = ColorUtility.ToHtmlStringRGB(dqnColor);

                scoreText.text = $"<b>Wins by Model:</b>\n" +
                                 $"<color=#{ppoHex}>PPO: {scores["PPO"]}</color>\n" +
                                 $"<color=#{sacHex}>SAC: {scores["SAC"]}</color>\n" +
                                 $"<color=#{dqnHex}>DQN: {scores["DQN"]}</color>\n\n" +
                                 $"Total: {scores["PPO"] + scores["SAC"] + scores["DQN"]}";
            }
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            Debug.Log($"[CompetitiveEnv] Found {terrains.Length} SM64 terrains");
            
            int collidersAdded = 0;
            foreach (var terrain in terrains)
            {
                if (terrain.GetComponent<MeshCollider>() == null)
                {
                    MeshFilter mf = terrain.GetComponent<MeshFilter>();
                    if (mf != null && mf.sharedMesh != null)
                    {
                        MeshCollider mc = terrain.gameObject.AddComponent<MeshCollider>();
                        mc.sharedMesh = mf.sharedMesh;
                        mc.convex = false;
                        collidersAdded++;
                    }
                }
            }
            Debug.Log($"[CompetitiveEnv] {collidersAdded} MeshColliders added");
        }

        private Material ResolveBaseMaterial()
        {
            Material matBase = baseMarioMaterial;
            if (matBase == null && marioPrefab != null)
            {
                SM64Mario prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    var matField = typeof(SM64Mario).GetField("material",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matBase = matField.GetValue(prefabMario) as Material;
                }
            }
            return matBase;
        }

        private Material CreateRuntimeMaterial(Material source, Color color, string label)
        {
            if (source == null) return null;

            Material mat = new Material(source);
            mat.name = $"CompetitiveMario_{label}";

            Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            tex.name = $"CompetitiveTex_{label}";
            tex.SetPixel(0, 0, color);
            tex.SetPixel(0, 1, color);
            tex.SetPixel(1, 0, color);
            tex.SetPixel(1, 1, color);
            tex.Apply();

            mat.mainTexture = tex;
            mat.SetTexture("_MainTex", tex);
            mat.color = Color.white;
            return mat;
        }

        private void SpawnAllMarios()
        {

            Material matBase = ResolveBaseMaterial();
            marioCount = 3;

            // Create runtime materials with colors configurable in the Inspector
            ppoRuntimeMaterial = CreateRuntimeMaterial(matBase, ppoColor, "PPO");
            sacRuntimeMaterial = CreateRuntimeMaterial(matBase, sacColor, "SAC");
            dqnRuntimeMaterial = CreateRuntimeMaterial(matBase, dqnColor, "DQN");

            Debug.Log($"[CompetitiveEnv] Starting spawn of {marioCount} Marios.");

            for (int i = 0; i < marioCount; i++)
            {
                Vector3 spawnPos = GetSpawnPositionForIndex(i);
                Material runtimeMat = (i == 0) ? ppoRuntimeMaterial : (i == 1) ? sacRuntimeMaterial : dqnRuntimeMaterial;
                Color teamColor = teamColors[i % teamColors.Length];
                SpawnMario(i, spawnPos, teamColor, runtimeMat);
            }

            Debug.Log($"[CompetitiveEnv] Total agents created: {agents.Count}");
            foreach (var agent in agents)
            {
                agent.SetRivals(agents);
            }

            Debug.Log("[CompetitiveEnv] 3 competitive Marios spawned successfully!");
        }

        private Vector3 GetSpawnPositionForIndex(int index)
        {
            if (spawnPoints != null && index >= 0 && index < spawnPoints.Length && spawnPoints[index] != null)
            {
                return spawnPoints[index].position;
            }

            if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                return spawnPoints[0].position + Vector3.right * (index * 2f);
            }

            return Vector3.right * (index * 2f);
        }

        private void SpawnMario(int index, Vector3 spawnPos, Color teamColor, Material runtimeMat)
        {
            string modelName = index < behaviorNames.Length ? behaviorNames[index] : $"Mario_Team{index}";
            GameObject marioObj = new GameObject($"Mario_{modelName}");
            marioObj.SetActive(false);
            marioObj.transform.position = spawnPos;

            // 1. Behavior Parameters FIRST
            var bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            bp.BehaviorName = modelName;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = 42;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });

            // 2. MarioCompetitiveAgent MUST BE ADDED BEFORE DecisionRequester
            // Otherwise, DecisionRequester's [RequireComponent(typeof(Agent))] will auto-add a base Agent!
            MarioCompetitiveAgent agent = marioObj.AddComponent<MarioCompetitiveAgent>();

            // 3. DecisionRequester
            var dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // 4. Input provider
            marioObj.AddComponent<MarioInputProvider>();
            
            // 5. SM64Mario
            SM64Mario sm64Mario = marioObj.AddComponent<SM64Mario>();
            
            // 5. Apply custom runtime material
            if (runtimeMat != null)
            {
                sm64Mario.useCustomTexture = true;
                sm64Mario.tintColor = Color.white;

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, runtimeMat);
            }

            // Configure agent properties
            agent.teamId = index;
            agent.teamColor = teamColor;
            agent.SetGoal(goal);
            agent.SetCompetitiveEnv(this);
            agents.Add(agent);
            
            // Ativa o objeto apenas após todos os componentes estarem configurados
            marioObj.SetActive(true);
        }

        public int RegisterFinish(MarioCompetitiveAgent agent)
        {
            finishOrder++;
            int position = finishOrder;
            
            if (position == 1)
            {
                if (agent.teamId == 0) scores["PPO"]++;
                else if (agent.teamId == 1) scores["SAC"]++;
                else if (agent.teamId == 2) scores["DQN"]++;
                
                UpdateScoreboardUI();
            }

            foreach (var a in agents)
            {
                if (a != agent && !a.HasFinished)
                {
                    a.OnRivalFinished(agent);
                }
            }

            bool allDone = true;
            foreach (var a in agents)
            {
                if (!a.HasFinished && a.gameObject.activeInHierarchy)
                {
                    allDone = false;
                    break;
                }
            }

            if (allDone)
            {
                finishOrder = 0;
                Debug.Log("[CompetitiveEnv] All Marios finished! Resetting order.");
            }

            return position;
        }

        public void RespawnAgent(MarioCompetitiveAgent agent)
        {
            int idx = agents.IndexOf(agent);
            Vector3 spawnPos;

            if (spawnPoints != null && idx < spawnPoints.Length && idx >= 0 && spawnPoints[idx] != null)
            {
                spawnPos = spawnPoints[idx].position + Vector3.up * 1f;
            }
            else if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                spawnPos = spawnPoints[0].position + Vector3.up * 1f + Vector3.right * (idx * 2f);
            }
            else
            {
                spawnPos = Vector3.up * 2f + Vector3.right * (idx * 2f);
            }

            SM64Mario sm64Mario = agent.GetComponent<SM64Mario>();
            if (sm64Mario != null)
            {
                sm64Mario.Teleport(spawnPos + Vector3.up * 1f);
            }
            else
            {
                agent.transform.position = spawnPos + Vector3.up * 1f;
            }
        }

        public Vector3 GetCurrentSpawnPoint(MarioCompetitiveAgent agent)
        {
            int idx = agents.IndexOf(agent);
            return GetSpawnPositionForIndex(idx >= 0 ? idx : 0);
        }

        void OnDrawGizmos()
        {
            if (spawnPoints != null)
            {
                for (int i = 0; i < spawnPoints.Length; i++)
                {
                    if (spawnPoints[i] != null)
                    {
                        Gizmos.color = (i < teamColors.Length) ? teamColors[i] : Color.white;
                        Gizmos.DrawWireSphere(spawnPoints[i].position, 0.8f);
                    }
                }
            }

            if (goal != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireCube(goal.position, Vector3.one * 3f);
            }
        }
    }
}

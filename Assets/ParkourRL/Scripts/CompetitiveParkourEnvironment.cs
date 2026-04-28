using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Ambiente competitivo para 3 Marios (PPO, SAC, DQN) simultaneos no mesmo mapa.
    /// Cores e texturas sao configuraveis no Inspector e aplicadas em runtime.
    /// </summary>
    public class CompetitiveParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Material (Opcional)")]
        [Tooltip("Material base opcional. Se nao atribuido, usa o marioPrefab como fallback.")]
        [SerializeField] private Material baseMarioMaterial;
        [Tooltip("Prefab opcional para obter material de fallback.")]
        [SerializeField] private GameObject marioPrefab;

        [Header("Cores dos Modelos (Inspector)")]
        [Tooltip("Cor do Mario PPO (Equipe 0).")]
        [SerializeField] private Color ppoColor = new Color(1f, 0.2f, 0.2f, 1f);
        [Tooltip("Cor do Mario SAC (Equipe 1).")]
        [SerializeField] private Color sacColor = new Color(0.2f, 0.4f, 1f, 1f);
        [Tooltip("Cor do Mario DQN (Equipe 2).")]
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
        
        // Placar
        private Dictionary<string, int> scores = new Dictionary<string, int>();
        private Text scoreText;

        // Materiais runtime
        private Material ppoRuntimeMaterial;
        private Material sacRuntimeMaterial;
        private Material dqnRuntimeMaterial;

        void Awake()
        {
            teamColors = new Color[] { ppoColor, sacColor, dqnColor };
        }

        void Start()
        {
            EnsureAllMeshColliders();
            SetupScoreboard();
            StartCoroutine(InitTerrainAndSpawn());
        }

        private IEnumerator InitTerrainAndSpawn()
        {
            // Aguarda 1 ciclo de fisica para garantir que os MeshColliders estejam prontos
            yield return new WaitForFixedUpdate();
            
            // Garantir que todos os terrains tenham MeshColliders
            EnsureAllMeshColliders();

            // Recarregar terreno SM64
            SM64Context.RefreshStaticTerrain();
            Debug.Log("[CompetitiveEnv] Terreno SM64 recarregado.");

            yield return new WaitForFixedUpdate();

            // Spawnar todos os Marios
            SpawnAllMarios();
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

                scoreText.text = $"<b>Vitorias por Modelo:</b>\n" +
                                 $"<color=#{ppoHex}>PPO: {scores["PPO"]}</color>\n" +
                                 $"<color=#{sacHex}>SAC: {scores["SAC"]}</color>\n" +
                                 $"<color=#{dqnHex}>DQN: {scores["DQN"]}</color>\n\n" +
                                 $"Total: {scores["PPO"] + scores["SAC"] + scores["DQN"]}";
            }
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
            Debug.Log($"[CompetitiveEnv] Encontrados {terrains.Length} terrains SM64");
            
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
            Debug.Log($"[CompetitiveEnv] {collidersAdded} MeshColliders adicionados");
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

            // Criar materiais runtime com cores configuraveis no Inspector
            ppoRuntimeMaterial = CreateRuntimeMaterial(matBase, ppoColor, "PPO");
            sacRuntimeMaterial = CreateRuntimeMaterial(matBase, sacColor, "SAC");
            dqnRuntimeMaterial = CreateRuntimeMaterial(matBase, dqnColor, "DQN");

            Debug.Log($"[CompetitiveEnv] Iniciando spawn de {marioCount} Marios.");

            for (int i = 0; i < marioCount; i++)
            {
                Vector3 spawnPos = GetSpawnPositionForIndex(i);
                Material runtimeMat = (i == 0) ? ppoRuntimeMaterial : (i == 1) ? sacRuntimeMaterial : dqnRuntimeMaterial;
                Color teamColor = teamColors[i % teamColors.Length];
                SpawnMario(i, spawnPos, teamColor, runtimeMat);
            }

            Debug.Log($"[CompetitiveEnv] Total de agentes criados: {agents.Count}");
            foreach (var agent in agents)
            {
                agent.SetRivals(agents);
            }

            Debug.Log("[CompetitiveEnv] 3 Marios competitivos spawnados com sucesso!");
        }

        private Vector3 GetSpawnPositionForIndex(int index)
        {
            if (spawnPoints != null && index >= 0 && index < spawnPoints.Length && spawnPoints[index] != null)
            {
                Vector3 pos = spawnPoints[index].position;
                Debug.Log($"[SpawnDebug] SpawnPoint[{index}] position: {pos} (name: {spawnPoints[index].name})");
                return pos;
            }

            if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                return spawnPoints[0].position + Vector3.right * (index * 2f);
            }

            return Vector3.up * 2f + Vector3.right * (index * 2f);
        }

        private void SpawnMario(int index, Vector3 spawnPos, Color teamColor, Material runtimeMat)
        {
            string modelName = index < behaviorNames.Length ? behaviorNames[index] : $"Mario_Team{index}";
            GameObject marioObj = new GameObject($"Mario_{modelName}");
            marioObj.SetActive(false);
            marioObj.transform.position = spawnPos;

            // 1. Behavior Parameters PRIMEIRO (Agent precisa disso no OnEnable)
            var bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            bp.BehaviorName = modelName;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = 42;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });

            // 2. DecisionRequester
            var dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // 3. Input provider ANTES de SM64Mario
            marioObj.AddComponent<MarioInputProvider>();
            
            // 4. SM64Mario
            SM64Mario sm64Mario = marioObj.AddComponent<SM64Mario>();
            
            // 5. Aplicar material runtime customizado
            if (runtimeMat != null)
            {
                sm64Mario.useCustomTexture = true;
                sm64Mario.tintColor = Color.white;

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, runtimeMat);
            }

            // 6. Agente competitivo DEPOIS de BehaviorParameters
            MarioCompetitiveAgent agent = marioObj.AddComponent<MarioCompetitiveAgent>();
            agent.teamId = index;
            agent.teamColor = teamColor;
            agent.SetGoal(goal);
            agent.SetCompetitiveEnv(this);
            agents.Add(agent);

            // 7. Ativar
            marioObj.SetActive(true);
            
            // 8. Teleportar para garantir posicao correta no terreno SM64 (apos 1 frame)
            StartCoroutine(TeleportMarioNextFrame(sm64Mario, spawnPos));
        }

        private System.Collections.IEnumerator TeleportMarioNextFrame(SM64Mario sm64Mario, Vector3 pos)
        {
            yield return null; // Aguarda 1 frame
            if (sm64Mario != null && sm64Mario.isActiveAndEnabled)
            {
                sm64Mario.Teleport(pos);
                Debug.Log($"[CompetitiveEnv] Mario teleportado para {pos}");
            }
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
                Debug.Log("[CompetitiveEnv] Todos os Marios terminaram! Resetando ordem.");
            }

            return position;
        }

        public void RespawnAgent(MarioCompetitiveAgent agent)
        {
            int idx = agents.IndexOf(agent);
            Vector3 spawnPos = GetSpawnPositionForIndex(idx >= 0 ? idx : 0);

            SM64Mario sm64Mario = agent.GetComponent<SM64Mario>();
            if (sm64Mario != null)
            {
                sm64Mario.Teleport(spawnPos);
            }
            else
            {
                agent.transform.position = spawnPos;
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

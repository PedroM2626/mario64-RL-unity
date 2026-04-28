using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Ambiente para teste simultaneo de algoritmos (PPO, SAC, DQN).
    /// </summary>
    public class CompetitiveParkourEnvironment : MonoBehaviour
    {
        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material baseMarioMaterial;

        [Header("Level")]
        [SerializeField] private Transform[] spawnPoints; // Um por Mario
        [SerializeField] private Transform goal;

        [Header("Competition")]
        [SerializeField] private int marioCount = 3;
        [SerializeField] private Color[] teamColors = new Color[]
        {
            Color.red,   // PPO
            Color.blue,  // SAC
            Color.green  // DQN
        };
        private string[] behaviorNames = new string[] { "MarioParkourPPO", "MarioParkourSAC", "MarioParkourDQN" };

        private List<MarioCompetitiveAgent> agents = new List<MarioCompetitiveAgent>();
        private int finishOrder = 0; // Contador de quem terminou
        
        // Placar
        private Dictionary<string, int> scores = new Dictionary<string, int>();
        private Text scoreText;

        void Start()
        {
            // Garantir MeshColliders
            EnsureAllMeshColliders();

            // Setup UI Scoreboard
            SetupScoreboard();

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
            scoreText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
            scoreText.fontSize = 24;
            scoreText.color = Color.white;
            scoreText.alignment = TextAnchor.UpperRight;
            
            RectTransform rt = scoreText.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(1, 1);
            rt.anchorMax = new Vector2(1, 1);
            rt.pivot = new Vector2(1, 1);
            rt.anchoredPosition = new Vector2(-20, -20);
            rt.sizeDelta = new Vector2(300, 200);

            UpdateScoreboardUI();
        }

        private void UpdateScoreboardUI()
        {
            if (scoreText != null)
            {
                scoreText.text = $"<b>Vitórias por Modelo:</b>\n" +
                                 $"<color=red>PPO: {scores["PPO"]}</color>\n" +
                                 $"<color=blue>SAC: {scores["SAC"]}</color>\n" +
                                 $"<color=green>DQN: {scores["DQN"]}</color>";
            }
        }

        private void EnsureAllMeshColliders()
        {
            SM64StaticTerrain[] terrains = FindObjectsOfType<SM64StaticTerrain>();
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
                    }
                }
            }
        }

        private void SpawnAllMarios()
        {
            if (marioPrefab == null)
            {
                #if UNITY_EDITOR
                marioPrefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Mario.prefab");
                #endif
                if (marioPrefab == null)
                {
                    Debug.LogError("[CompetitiveEnv] Mario prefab nao encontrado!");
                    return;
                }
            }

            // Obter material base
            Material matBase = baseMarioMaterial;
            if (matBase == null)
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

            for (int i = 0; i < marioCount; i++)
            {
                Vector3 spawnPos;
                if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
                {
                    // Todos spawnando exatamente no mesmo lugar
                    spawnPos = spawnPoints[0].position + Vector3.up * 1f;
                }
                else
                {
                    spawnPos = Vector3.up * 2f;
                }

                Color teamColor = teamColors[i % teamColors.Length];
                SpawnMario(i, spawnPos, teamColor, matBase);
            }

            // Conectar rivais
            foreach (var agent in agents)
            {
                agent.SetRivals(agents);
            }

            Debug.Log($"[CompetitiveEnv] {marioCount} Marios competitivos spawnados!");
        }

        private void SpawnMario(int index, Vector3 spawnPos, Color teamColor, Material matBase)
        {
            GameObject marioObj = new GameObject($"Mario_Team{index}");
            marioObj.SetActive(false);
            marioObj.transform.position = spawnPos;

            // Agente competitivo
            MarioCompetitiveAgent agent = marioObj.AddComponent<MarioCompetitiveAgent>();
            agent.teamId = index;
            agent.teamColor = teamColor;

            // Input provider
            MarioInputProvider inputProvider = marioObj.AddComponent<MarioInputProvider>();

            // SM64Mario com material colorido por time
            SM64Mario sm64Mario = marioObj.AddComponent<SM64Mario>();
            if (matBase != null)
            {
                // Criar material unico por time com tint de cor
                Material teamMat = new Material(matBase);
                teamMat.color = teamColor;
                teamMat.name = $"MarioMat_Team{index}";

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, teamMat);
            }

            // Behavior Parameters
            var bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            bp.BehaviorName = index < behaviorNames.Length ? behaviorNames[index] : "MarioParkourPPO";
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = 42; // 30 base + 12 rivais
            bp.BrainParameters.NumStackedVectorObservations = 1;
            // 2 continuas (joystick) + 3 discretas (Jump[2], Kick[2], Stomp[2])
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });

            // Decision Requester
            var dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            // Configurar referencias
            agent.SetGoal(goal);
            agent.SetCompetitiveEnv(this);

            marioObj.SetActive(true);
            agents.Add(agent);

            Debug.Log($"[CompetitiveEnv] Mario Team{index} spawnado em {spawnPos} (cor: {teamColor})");
        }

        public int RegisterFinish(MarioCompetitiveAgent agent)
        {
            finishOrder++;
            int position = finishOrder;
            
            if (position == 1) // Apenas o primeiro que termina ganha ponto
            {
                if (agent.teamId == 0) scores["PPO"]++;
                else if (agent.teamId == 1) scores["SAC"]++;
                else if (agent.teamId == 2) scores["DQN"]++;
                
                UpdateScoreboardUI();
            }

            // Notificar rivais
            foreach (var a in agents)
            {
                if (a != agent && !a.HasFinished)
                {
                    a.OnRivalFinished(agent);
                }
            }

            // Se todos terminaram, resetar contador
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

        /// <summary>
        /// Chamado por agentes individuais para reposicionar no spawn
        /// </summary>
        public void RespawnAgent(MarioCompetitiveAgent agent)
        {
            int idx = agents.IndexOf(agent);
            Vector3 spawnPos;

            if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                spawnPos = spawnPoints[0].position + Vector3.up * 1f;
            }
            else
            {
                spawnPos = Vector3.up * 2f;
            }

            SM64Mario sm64Mario = agent.GetComponent<SM64Mario>();
            if (sm64Mario != null)
            {
                sm64Mario.Teleport(spawnPos + Vector3.up * 1f); // Added extra height
            }
            else
            {
                agent.transform.position = spawnPos + Vector3.up * 1f;
            }
        }

        public Vector3 GetCurrentSpawnPoint(MarioCompetitiveAgent agent)
        {
            if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                return spawnPoints[0].position;
            }
            return Vector3.zero;
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

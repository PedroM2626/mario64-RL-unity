using UnityEngine;
using System.Collections.Generic;
using LibSM64;
using Unity.Barracuda;
using Unity.MLAgents;

namespace ParkourRL
{
    /// <summary>
    /// Ambiente de batalha em times: 2 times com 5 Marios cada competem.
    /// Usa MA-POCA (Multi-Agent POsthumous Credit Assignment) para recompensas cooperativas.
    /// Tipo de MARL: CTDE (Centralized Training with Decentralized Execution).
    /// </summary>
    public class TeamBattleEnvironment : MonoBehaviour
    {
        private const string TeamBattleBehaviorName = "MarioTeamBattle";
        private const int TeamBattleVectorObservationSize = 56;

        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material baseMarioMaterial;
        [Tooltip("Specific material for Team A. If empty, uses baseMarioMaterial with team color.")]
        [SerializeField] private Material teamAMaterial;
        [Tooltip("Specific material for Team B. If empty, uses baseMarioMaterial with team color.")]
        [SerializeField] private Material teamBMaterial;

        [Header("Warm Start")]
        [Tooltip("Uses a base model for warm-starting Mario in BehaviorParameters.")]
        [SerializeField] private bool useWarmStartModel = false;
        [SerializeField] private NNModel warmStartModel;
        [Tooltip("Asset path to auto-load the model in the editor when the field above is empty.")]
        [SerializeField] private string warmStartModelAssetPath = "";

        [Header("Arena Setup")]
        [SerializeField] private Transform[] teamASpawnPoints;  // 5 spawn points para Time A
        [SerializeField] private Transform[] teamBSpawnPoints;  // 5 spawn points para Time B
        [SerializeField] private Transform arenaCenter;         // Centro da arena (para verificar limites)
        [SerializeField] private float arenaRadius = 30f;       // Valid arena radius

        [Header("Battle Settings")]
        [SerializeField] private int marioPerTeam = 5;
        [SerializeField] private float maxBattleTime = 120f;    // Maximum battle time
        [SerializeField] private float marioHealth = 100f;      // Vida inicial de cada Mario
        [SerializeField] private float kickDamage = 20f;        // Dano de um chute
        [SerializeField] private float stompDamage = 15f;       // Dano de um stompo
        [SerializeField] private float knockbackForce = 5f;     // Knockback force when hit

        [Header("Team Colors")]
        [Tooltip("Team A color (default = red). Leave as (0,0,0,0) to use default red.")]
        [SerializeField] private Color teamAColor = Color.red;
        [Tooltip("Team B color (default = light blue). Leave as (0,0,0,0) to use default light blue.")]
        [SerializeField] private Color teamBColor = new Color(0, 0.7f, 1f);

        // Estruturas internas
        private List<TeamBattleAgent>[] agentsByTeam;           // agentsByTeam[0] = Team A, agentsByTeam[1] = Team B
        private SimpleMultiAgentGroup[] teamGroups;             // MA-POCA groups: teamGroups[0] = Team A, teamGroups[1] = Team B
        private float battleStartTime;
        private bool battleActive = true;
        private bool hasSpawned = false;

        void Awake()
        {
            agentsByTeam = new List<TeamBattleAgent>[2];
            agentsByTeam[0] = new List<TeamBattleAgent>();
            agentsByTeam[1] = new List<TeamBattleAgent>();

            // Inicializar grupos MA-POCA para cada time
            teamGroups = new SimpleMultiAgentGroup[2];
            for (int i = 0; i < 2; i++)
            {
                teamGroups[i] = new SimpleMultiAgentGroup();
            }

            TryResolveWarmStartModel();
        }

        void OnValidate()
        {
            TryResolveWarmStartModel();
        }

        private void TryResolveWarmStartModel()
        {
            if (!useWarmStartModel || warmStartModel != null)
                return;

#if UNITY_EDITOR
            if (!string.IsNullOrWhiteSpace(warmStartModelAssetPath))
            {
                warmStartModel = UnityEditor.AssetDatabase.LoadAssetAtPath<NNModel>(warmStartModelAssetPath);
            }
#endif
        }

        void Start()
        {
            // Garantir MeshColliders em todo terreno
            EnsureAllMeshColliders();

            // Reload SM64 terrain
            SM64Context.RefreshStaticTerrain();

            // Spawn all Marios (apenas uma vez)
            if (!hasSpawned)
            {
                hasSpawned = true;
                Invoke(nameof(SpawnAllMarios), 0.5f);
            }
            
            battleStartTime = Time.time;
        }

        void FixedUpdate()
        {
            if (!battleActive) return;

            // Check for victory: a team was eliminated
            if (agentsByTeam[0].Count == 0 || agentsByTeam[1].Count == 0)
            {
                EndBattle();
            }

            // Check timeout
            if (Time.time - battleStartTime > maxBattleTime)
            {
                EndBattle();
            }

            // Detect and process combat collisions
            ProcessCombatCollisions();
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
                    Debug.LogError("[TeamBattleEnv] Mario prefab not found!");
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

            // Spawnar Time A (Red)
            Material matA = (teamAMaterial != null) ? teamAMaterial : matBase;
            for (int i = 0; i < marioPerTeam; i++)
            {
                Vector3 spawnPos = GetSpawnPosition(teamASpawnPoints, i);
                SpawnMario(0, i, spawnPos, teamAColor, matA);
            }

            // Spawnar Time B (Blue)
            Material matB = (teamBMaterial != null) ? teamBMaterial : matBase;
            for (int i = 0; i < marioPerTeam; i++)
            {
                Vector3 spawnPos = GetSpawnPosition(teamBSpawnPoints, i);
                SpawnMario(1, i, spawnPos, teamBColor, matB);
            }

            // Conectar agentes e rivals
            foreach (var agent in agentsByTeam[0])
            {
                agent.SetTeammates(agentsByTeam[0]);
                agent.SetRivals(agentsByTeam[1]);
            }

            foreach (var agent in agentsByTeam[1])
            {
                agent.SetTeammates(agentsByTeam[1]);
                agent.SetRivals(agentsByTeam[0]);
            }

            Debug.Log($"[TeamBattleEnv] Batalha iniciada! Time A (Red): {marioPerTeam} | Time B (Blue): {marioPerTeam}");
        }

        private Vector3 GetSpawnPosition(Transform[] spawnPoints, int index)
        {
            if (spawnPoints != null && index < spawnPoints.Length && spawnPoints[index] != null)
            {
                return spawnPoints[index].position + Vector3.up * 1f;
            }
            else if (spawnPoints != null && spawnPoints.Length > 0 && spawnPoints[0] != null)
            {
                return spawnPoints[0].position + Vector3.up * 1f + Vector3.right * (index * 2f);
            }
            return Vector3.up * 2f + Vector3.right * (index * 2f);
        }

        private void SpawnMario(int teamId, int indexInTeam, Vector3 spawnPos, Color teamColor, Material matBase)
        {
            string teamName = (teamId == 0) ? "A" : "B";
            GameObject marioObj = new GameObject($"Mario_Team{teamName}_{indexInTeam}");
            marioObj.SetActive(false);
            marioObj.transform.position = spawnPos;

            // Agente de batalha em time
            TeamBattleAgent agent = marioObj.AddComponent<TeamBattleAgent>();
            agent.teamId = teamId;
            agent.teamColor = teamColor;
            agent.battleEnvironment = this;
            agent.maxHealth = marioHealth;

            // Input provider
            MarioInputProvider inputProvider = marioObj.AddComponent<MarioInputProvider>();

            // SM64Mario com material colorido por time
            SM64Mario sm64Mario = marioObj.AddComponent<SM64Mario>();
            if (matBase != null)
            {
                Material teamMat = new Material(matBase);
                teamMat.name = $"MarioMat_Team{teamName}_{indexInTeam}";

                // Detectar se estamos usando um material customizado (teamAMaterial/teamBMaterial)
                bool isCustomMaterial = (teamId == 0 && teamAMaterial != null) || (teamId == 1 && teamBMaterial != null);

                // Sempre usar useCustomTexture=true para que SM64Mario.OnEnable()
                // nao sobrescreva _MainTex com a textura nativa do Mario.
                sm64Mario.useCustomTexture = true;
                sm64Mario.tintColor = Color.white; // nao alterar vertex colors nativas

                if (isCustomMaterial)
                {
                    // Material customizado: respeitar sua textura
                    // Nada a fazer, a textura do material ja esta configurada
                }
                else
                {
                    // Sem material customizado: criar textura da cor do time e aplicar no corpo inteiro
                    // O shader faz lerp(vertexColor, textureColor, textureAlpha). Se alpha=1, usa 100% textura.
                    // Criamos uma textura 2x2 da cor do time para garantir cor uniforme.
                    Texture2D teamTex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                    teamTex.SetPixel(0, 0, teamColor);
                    teamTex.SetPixel(0, 1, teamColor);
                    teamTex.SetPixel(1, 0, teamColor);
                    teamTex.SetPixel(1, 1, teamColor);
                    teamTex.Apply();
                    teamMat.mainTexture = teamTex;
                    teamMat.SetTexture("_MainTex", teamTex);
                    teamMat.color = Color.white; // deixar tint neutro, a cor vem da textura
                }

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, teamMat);

                // O OnEnable() do SM64Mario ja rodou e criou o MeshRenderer com o material original.
                // Precisamos atualizar o MeshRenderer filho diretamente para que a cor/textura novas sejam aplicadas.
                Transform rendererChild = marioObj.transform.Find("MARIO");
                if (rendererChild != null)
                {
                    MeshRenderer mr = rendererChild.GetComponent<MeshRenderer>();
                    if (mr != null && teamMat != null)
                    {
                        mr.material = teamMat;
                    }
                }
            }

            // Collider for combat detection
            SphereCollider combatCollider = marioObj.AddComponent<SphereCollider>();
            combatCollider.radius = 1.5f;
            combatCollider.isTrigger = true;

            // Behavior Parameters (forcar em runtime para evitar inconsistencias da cena/prefab)
            var bp = marioObj.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp == null)
            {
                bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            }

            bp.BehaviorName = TeamBattleBehaviorName;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            bp.BrainParameters.VectorObservationSize = TeamBattleVectorObservationSize;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });

            // Sempre configurar TeamId em runtime (varia por time)
            bp.TeamId = teamId;
            // Warm-start model
            TryResolveWarmStartModel();
            if (useWarmStartModel && warmStartModel != null)
            {
                bp.Model = warmStartModel;
            }

            // Decision Requester (esperado no prefab, fallback se ausente)
            var dr = marioObj.GetComponent<Unity.MLAgents.DecisionRequester>();
            if (dr == null)
            {
                dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
                dr.DecisionPeriod = 5;
                dr.TakeActionsBetweenDecisions = true;
            }

            marioObj.SetActive(true);
            agentsByTeam[teamId].Add(agent);

            // Registrar agente no grupo MA-POCA do seu time
            if (teamGroups != null && teamGroups[teamId] != null)
            {
                teamGroups[teamId].RegisterAgent(agent);
            }

            Debug.Log($"[TeamBattleEnv] Mario Team{teamName}_{indexInTeam} spawnado em {spawnPos}");
        }

        /// <summary>
        /// Processes combat collisions: checks Kicks and Stomps between enemies
        /// </summary>
        private void ProcessCombatCollisions()
        {
            // Criar copias das listas para evitar InvalidOperationException
            // caso DealDamage -> EliminateAgent remova um agente da lista original
            List<TeamBattleAgent> teamA = new List<TeamBattleAgent>(agentsByTeam[0]);
            List<TeamBattleAgent> teamB = new List<TeamBattleAgent>(agentsByTeam[1]);

            // Team A atacando Team B
            foreach (var attacker in teamA)
            {
                if (attacker == null || !attacker.isActiveAndEnabled) continue;

                foreach (var defender in teamB)
                {
                    if (defender == null || !defender.isActiveAndEnabled) continue;

                    // Check distance for possible impact
                    float dist = Vector3.Distance(attacker.transform.position, defender.transform.position);
                    if (dist < 2.5f) // Impact distance
                    {
                        if (attacker.lastKickPressed && dist < 2f)
                        {
                            DealDamage(defender, kickDamage, attacker);
                            attacker.lastKickPressed = false; // Resetar para evitar dano multiplo
                        }
                        if (attacker.lastStompPressed && dist < 2.5f)
                        {
                            DealDamage(defender, stompDamage, attacker);
                            attacker.lastStompPressed = false; // Resetar para evitar dano multiplo
                        }
                    }
                }
            }

            // Team B atacando Team A
            foreach (var attacker in teamB)
            {
                if (attacker == null || !attacker.isActiveAndEnabled) continue;

                foreach (var defender in teamA)
                {
                    if (defender == null || !defender.isActiveAndEnabled) continue;

                    float dist = Vector3.Distance(attacker.transform.position, defender.transform.position);
                    if (dist < 2.5f)
                    {
                        if (attacker.lastKickPressed && dist < 2f)
                        {
                            DealDamage(defender, kickDamage, attacker);
                            attacker.lastKickPressed = false; // Resetar para evitar dano multiplo
                        }
                        if (attacker.lastStompPressed && dist < 2.5f)
                        {
                            DealDamage(defender, stompDamage, attacker);
                            attacker.lastStompPressed = false; // Resetar para evitar dano multiplo
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Aplica dano a um agente
        /// </summary>
        public void DealDamage(TeamBattleAgent defender, float damage, TeamBattleAgent attacker)
        {
            if (defender == null || !defender.isActiveAndEnabled) return;

            defender.TakeDamage(damage);

            // Recompensa para o atacante
            if (attacker != null)
            {
                attacker.AddReward(0.5f); // Pequena recompensa por atingir
            }

            // Aplicar knockback
            Vector3 knockbackDir = (defender.transform.position - attacker.transform.position).normalized;
            defender.ApplyKnockback(knockbackDir * knockbackForce);

            // Verificar se foi eliminado
            if (defender.currentHealth <= 0)
            {
                EliminateAgent(defender);
            }

            Debug.Log($"[TeamBattle] Mario Team{(defender.teamId == 0 ? 'A' : 'B')} recebeu {damage} de dano!");
        }

        /// <summary>
        /// Remove um agente da batalha (foi eliminado)
        /// </summary>
        public void EliminateAgent(TeamBattleAgent agent)
        {
            if (agent == null) return;

            Debug.Log($"[TeamBattle] Mario Team{(agent.teamId == 0 ? 'A' : 'B')} foi eliminado!");

            int teamId = agent.teamId;
            agentsByTeam[teamId].Remove(agent);

            // Recompensa de time: eliminar inimigo beneficia TODO o time adversario
            int enemyTeamId = 1 - teamId;
            if (teamGroups != null && teamGroups[enemyTeamId] != null)
            {
                teamGroups[enemyTeamId].AddGroupReward(1.0f); // Group reward por eliminar inimigo
            }

            // Penalidade individual por morte (ja e suficiente)
            agent.AddReward(-2f);
            agent.EndEpisode();

            // Desregistrar do grupo MA-POCA antes de destruir
            if (teamGroups != null && teamGroups[teamId] != null)
            {
                teamGroups[teamId].UnregisterAgent(agent);
            }

            // Desativar imediatamente para que outros agentes nao tentem acessar
            agent.gameObject.SetActive(false);
            // Destroy after a delay for cleanup
            Destroy(agent.gameObject, 0.2f);
        }

        /// <summary>
        /// Finaliza a batalha e distribui recompensas
        /// </summary>
        private void EndBattle()
        {
            if (!battleActive) return;
            battleActive = false;

            // Determinar vencedor
            int teamACount = agentsByTeam[0].FindAll(a => a != null && a.isActiveAndEnabled).Count;
            int teamBCount = agentsByTeam[1].FindAll(a => a != null && a.isActiveAndEnabled).Count;

            float timeElapsed = Time.time - battleStartTime;

            if (teamACount > teamBCount)
            {
                // Time A venceu: group reward para TODO o time (inclusive agentes ja eliminados via MA-POCA)
                if (teamGroups[0] != null)
                {
                    teamGroups[0].AddGroupReward(10f + (maxBattleTime - timeElapsed) * 0.25f);
                    teamGroups[0].EndGroupEpisode();
                }
                if (teamGroups[1] != null)
                {
                    teamGroups[1].AddGroupReward(-3f);
                    teamGroups[1].EndGroupEpisode();
                }
                Debug.Log($"[TeamBattle] Time A VENCEU! Tempo: {timeElapsed:F1}s");
            }
            else if (teamBCount > teamACount)
            {
                // Time B venceu
                if (teamGroups[1] != null)
                {
                    teamGroups[1].AddGroupReward(10f + (maxBattleTime - timeElapsed) * 0.25f);
                    teamGroups[1].EndGroupEpisode();
                }
                if (teamGroups[0] != null)
                {
                    teamGroups[0].AddGroupReward(-3f);
                    teamGroups[0].EndGroupEpisode();
                }
                Debug.Log($"[TeamBattle] Time B VENCEU! Tempo: {timeElapsed:F1}s");
            }
            else
            {
                // Empate
                if (teamGroups[0] != null)
                {
                    teamGroups[0].AddGroupReward(2f);
                    teamGroups[0].EndGroupEpisode();
                }
                if (teamGroups[1] != null)
                {
                    teamGroups[1].AddGroupReward(2f);
                    teamGroups[1].EndGroupEpisode();
                }
                Debug.Log($"[TeamBattle] EMPATE! Tempo: {timeElapsed:F1}s");
            }

            // End remaining individual episodes (agents still alive)
            foreach (var team in agentsByTeam)
            {
                foreach (var agent in team)
                {
                    if (agent != null && agent.isActiveAndEnabled)
                    {
                        agent.EndEpisode();
                    }
                }
            }

            // Reiniciar batalha
            Invoke(nameof(ResetBattle), 2f);
        }

        private void ResetBattle()
        {
            // Destruir todos os GameObjects dos agents antigos e limpar grupos MA-POCA
            for (int t = 0; t < 2; t++)
            {
                // Desregistrar todos os agentes do grupo MA-POCA um por um
                if (teamGroups[t] != null)
                {
                    foreach (var agent in agentsByTeam[t])
                    {
                        if (agent != null)
                            teamGroups[t].UnregisterAgent(agent);
                    }
                }

                foreach (var agent in agentsByTeam[t])
                {
                    if (agent != null && agent.gameObject != null)
                    {
                        Destroy(agent.gameObject);
                    }
                }
                agentsByTeam[t].Clear();
            }

            battleActive = true;
            battleStartTime = Time.time;
            SpawnAllMarios();
        }

        /// <summary>
        /// Retorna teammates de um agente
        /// </summary>
        public List<TeamBattleAgent> GetTeammates(TeamBattleAgent agent)
        {
            return agentsByTeam[agent.teamId];
        }

        /// <summary>
        /// Retorna inimigos de um agente
        /// </summary>
        public List<TeamBattleAgent> GetRivals(TeamBattleAgent agent)
        {
            int enemyTeamId = 1 - agent.teamId;
            return agentsByTeam[enemyTeamId];
        }

        /// <summary>
        /// Checks if a Mario is outside the arena
        /// </summary>
        public bool IsOutOfBounds(Vector3 position)
        {
            if (arenaCenter == null) return false;
            return Vector3.Distance(position, arenaCenter.position) > arenaRadius;
        }

        void OnDrawGizmosSelected()
        {
            // Desenhar zonas de spawn
            if (teamASpawnPoints != null)
            {
                Gizmos.color = Color.red;
                foreach (var sp in teamASpawnPoints)
                {
                    if (sp != null)
                        Gizmos.DrawWireSphere(sp.position, 0.8f);
                }
            }

            if (teamBSpawnPoints != null)
            {
                Gizmos.color = new Color(0, 0.7f, 1f);
                foreach (var sp in teamBSpawnPoints)
                {
                    if (sp != null)
                        Gizmos.DrawWireSphere(sp.position, 0.8f);
                }
            }

            // Desenhar limite da arena
            if (arenaCenter != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(arenaCenter.position, arenaRadius);
            }
        }
    }
}

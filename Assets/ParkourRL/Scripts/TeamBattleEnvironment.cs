using UnityEngine;
using System.Collections.Generic;
using LibSM64;

namespace ParkourRL
{
    /// <summary>
    /// Ambiente de batalha em times: 2 times com 5 Marios cada competem.
    /// Objetivo: derrotar o time adversário (eliminar todos os inimigos ou sair da arena).
    /// </summary>
    public class TeamBattleEnvironment : MonoBehaviour
    {
        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material baseMarioMaterial;

        [Header("Arena Setup")]
        [SerializeField] private Transform[] teamASpawnPoints;  // 5 spawn points para Time A
        [SerializeField] private Transform[] teamBSpawnPoints;  // 5 spawn points para Time B
        [SerializeField] private Transform arenaCenter;         // Centro da arena (para verificar limites)
        [SerializeField] private float arenaRadius = 30f;       // Raio da arena válida

        [Header("Battle Settings")]
        [SerializeField] private int marioPerTeam = 5;
        [SerializeField] private float maxBattleTime = 120f;    // Tempo máximo de batalha
        [SerializeField] private float marioHealth = 100f;      // Vida inicial de cada Mario
        [SerializeField] private float kickDamage = 20f;        // Dano de um chute
        [SerializeField] private float stompDamage = 15f;       // Dano de um stompo
        [SerializeField] private float knockbackForce = 5f;     // Força de knockback ao ser atingido

        // Cores dos times
        private Color teamAColor = Color.red;
        private Color teamBColor = new Color(0, 0.7f, 1f); // Azul claro

        // Estruturas internas
        private List<TeamBattleAgent>[] agentsByTeam;           // agentsByTeam[0] = Team A, agentsByTeam[1] = Team B
        private float battleStartTime;
        private bool battleActive = true;
        private bool hasSpawned = false;

        void Awake()
        {
            agentsByTeam = new List<TeamBattleAgent>[2];
            agentsByTeam[0] = new List<TeamBattleAgent>();
            agentsByTeam[1] = new List<TeamBattleAgent>();
        }

        void Start()
        {
            // Garantir MeshColliders em todo terreno
            EnsureAllMeshColliders();

            // Recarregar terreno SM64
            SM64Context.RefreshStaticTerrain();

            // Spawnar todos os Marios (apenas uma vez)
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

            // Verificar vitória: um time foi eliminado
            if (agentsByTeam[0].Count == 0 || agentsByTeam[1].Count == 0)
            {
                EndBattle();
            }

            // Verificar timeout
            if (Time.time - battleStartTime > maxBattleTime)
            {
                EndBattle();
            }

            // Detectar e processar colisões de combate
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
                    Debug.LogError("[TeamBattleEnv] Mario prefab não encontrado!");
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
            for (int i = 0; i < marioPerTeam; i++)
            {
                Vector3 spawnPos = GetSpawnPosition(teamASpawnPoints, i);
                SpawnMario(0, i, spawnPos, teamAColor, matBase);
            }

            // Spawnar Time B (Blue)
            for (int i = 0; i < marioPerTeam; i++)
            {
                Vector3 spawnPos = GetSpawnPosition(teamBSpawnPoints, i);
                SpawnMario(1, i, spawnPos, teamBColor, matBase);
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
                teamMat.color = teamColor;
                teamMat.name = $"MarioMat_Team{teamName}_{indexInTeam}";

                var materialField = typeof(SM64Mario).GetField("material",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                if (materialField != null)
                    materialField.SetValue(sm64Mario, teamMat);
            }

            // Collider para detecção de combate
            SphereCollider combatCollider = marioObj.AddComponent<SphereCollider>();
            combatCollider.radius = 1.5f;
            combatCollider.isTrigger = true;

            // Behavior Parameters (Agent already requires one; configure it instead of adding a duplicate)
            var bp = marioObj.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp == null)
                bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            bp.BehaviorName = "MarioTeamBattle";
            bp.TeamId = teamId;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            // Observations: 3 pos + 3 vel + 1 grounded + 16 raycasts + 1 ground height + 1 time
            // + 15 teammates (5 agents * 3: pos, health, distance)
            // + 15 enemies (5 agents * 3: pos, health, distance)
            // = 3+3+1+16+1+1 + 15 + 15 = 55
            bp.BrainParameters.VectorObservationSize = 55;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            // 2 continuas (joystick) + 3 discretas (Jump[2], Kick[2], Stomp[2])
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(2, new int[] { 2, 2, 2 });

            // Decision Requester
            var dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;

            marioObj.SetActive(true);
            agentsByTeam[teamId].Add(agent);

            Debug.Log($"[TeamBattleEnv] Mario Team{teamName}_{indexInTeam} spawnado em {spawnPos}");
        }

        /// <summary>
        /// Processa colisões de combate: verifica Kicks e Stomps entre inimigos
        /// </summary>
        private void ProcessCombatCollisions()
        {
            // Team A atacando Team B
            foreach (var attacker in agentsByTeam[0])
            {
                if (attacker == null || !attacker.isActiveAndEnabled) continue;

                foreach (var defender in agentsByTeam[1])
                {
                    if (defender == null || !defender.isActiveAndEnabled) continue;

                    // Verificar distância para possível impacto
                    float dist = Vector3.Distance(attacker.transform.position, defender.transform.position);
                    if (dist < 2.5f) // Distância de impacto
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
            foreach (var attacker in agentsByTeam[1])
            {
                if (attacker == null || !attacker.isActiveAndEnabled) continue;

                foreach (var defender in agentsByTeam[0])
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

            agentsByTeam[agent.teamId].Remove(agent);
            agent.AddReward(-10f); // Penalidade por morte
            agent.EndEpisode();

            // Destruir após um delay para animação
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
                // Time A venceu
                foreach (var agent in agentsByTeam[0])
                {
                    if (agent != null && agent.isActiveAndEnabled)
                    {
                        agent.AddReward(20f + (maxBattleTime - timeElapsed) * 0.5f); // Recompensa vitória
                    }
                }
                foreach (var agent in agentsByTeam[1])
                {
                    if (agent != null && agent.isActiveAndEnabled)
                    {
                        agent.AddReward(-5f); // Penalidade derrota
                    }
                }
                Debug.Log($"[TeamBattle] Time A VENCEU! Tempo: {timeElapsed:F1}s");
            }
            else if (teamBCount > teamACount)
            {
                // Time B venceu
                foreach (var agent in agentsByTeam[1])
                {
                    if (agent != null && agent.isActiveAndEnabled)
                    {
                        agent.AddReward(20f + (maxBattleTime - timeElapsed) * 0.5f);
                    }
                }
                foreach (var agent in agentsByTeam[0])
                {
                    if (agent != null && agent.isActiveAndEnabled)
                    {
                        agent.AddReward(-5f);
                    }
                }
                Debug.Log($"[TeamBattle] Time B VENCEU! Tempo: {timeElapsed:F1}s");
            }
            else
            {
                // Empate
                foreach (var agent in agentsByTeam[0])
                {
                    if (agent != null) agent.AddReward(5f);
                }
                foreach (var agent in agentsByTeam[1])
                {
                    if (agent != null) agent.AddReward(5f);
                }
                Debug.Log($"[TeamBattle] EMPATE! Tempo: {timeElapsed:F1}s");
            }

            // Finalizar todos os episódios
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
            // Destruir todos os GameObjects dos agents antigos
            for (int t = 0; t < 2; t++)
            {
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
        /// Verifica se um Mario está fora da arena
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

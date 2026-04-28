using UnityEngine;
using LibSM64;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Actuators;

namespace ParkourRL
{
    /// <summary>
    /// Ambiente 1v1 de perseguição: um perseguidor precisa atingir o fugitivo
    /// com golpes (soco/chute/rasteira), enquanto o fugitivo tenta sobreviver.
    /// Treino descentralizado com duas policies SAC.
    /// </summary>
    public class ChaseTrainingEnvironment : MonoBehaviour
    {
        private const int ContinuousActionSize = 5;
        private const string PursuerBehaviorName = "ChasePursuer";
        private const string FugitiveBehaviorName = "ChaseFugitive";
        private const int VectorObservationSize = 31;

        [Header("Mario Setup")]
        [SerializeField] private GameObject marioPrefab;
        [SerializeField] private Material baseMarioMaterial;
        [SerializeField] private Material pursuerMaterial;
        [SerializeField] private Material fugitiveMaterial;

        [Header("Arena Setup")]
        [SerializeField] private Transform pursuerSpawnPoint;
        [SerializeField] private Transform fugitiveSpawnPoint;
        [SerializeField] private Transform arenaCenter;
        [SerializeField] private float arenaRadius = 22f;

        [Header("Chase Settings")]
        [SerializeField] private float maxBattleTime = 30f;
        [SerializeField] private float catchGraceAfterSpawn = 1.5f;
        [SerializeField] private float spawnJitterRadius = 1.25f;
        [SerializeField] private float minimumSpawnSeparation = 6f;

        [Header("Combat Settings")]
        [Tooltip("Distância máxima para um ataque (kick/stomp) acertar o oponente.")]
        [SerializeField] private float attackDistance = 2.5f;
        [Tooltip("Distância máxima para um kick acertar (ligeiramente menor que stomp).")]
        [SerializeField] private float kickHitDistance = 2f;
        [Tooltip("Força de knockback aplicada no fugitivo ao ser atingido.")]
        [SerializeField] private float knockbackForce = 5f;

        [Header("Visual")]
        [SerializeField] private Color pursuerColor = Color.red;
        [SerializeField] private Color fugitiveColor = new Color(0.2f, 0.9f, 1f);
        [SerializeField] private bool enableEpisodeLogs = false;

        private ChaseAgent pursuer;
        private ChaseAgent fugitive;
        private Material pursuerRuntimeMaterial;
        private Material fugitiveRuntimeMaterial;
        private Texture2D pursuerRuntimeTexture;
        private Texture2D fugitiveRuntimeTexture;
        // Track which BehaviorParameters have already been sanitized to avoid reprocessing
        private System.Collections.Generic.HashSet<int> sanitizedBPInstanceIDs = new System.Collections.Generic.HashSet<int>();
        // Track which behavior names have already produced a warning to rate-limit logs
        private System.Collections.Generic.HashSet<string> warnedBehaviorNames = new System.Collections.Generic.HashSet<string>();
        private float battleStartTime;
        private float lastSpawnTime;
        private bool duelActive = true;
        private bool hasSpawned;

        void Start()
        {
            EnsureAllMeshColliders();
            SM64Context.RefreshStaticTerrain();
            SanitizeBehaviorParameters();

            if (!hasSpawned)
            {
                hasSpawned = true;
                Invoke(nameof(SpawnDuelAgents), 0.5f);
            }

            battleStartTime = Time.time;
        }

        void FixedUpdate()
        {
            if (!duelActive)
                return;

            if (pursuer == null || fugitive == null || !pursuer.isActiveAndEnabled || !fugitive.isActiveAndEnabled)
                return;

            ProcessCombatCollision();

            if (Time.time - battleStartTime > maxBattleTime)
            {
                ResolveTimeout();
            }

            // Limpar flags de ataque após o processamento deste frame
            pursuer?.ClearAttackFlags();
            fugitive?.ClearAttackFlags();
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

        private void SpawnDuelAgents()
        {
            if (marioPrefab == null)
            {
#if UNITY_EDITOR
                marioPrefab = UnityEditor.AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Mario.prefab");
#endif
                if (marioPrefab == null)
                {
                    Debug.LogError("[ChaseTraining] Mario prefab nao encontrado.");
                    return;
                }
            }

            Material matBase = ResolveBaseMaterial();
            EnsureRuntimeVisualResources(matBase);

            Vector3 pursuerSpawnPos = GetSpawnPosition(pursuerSpawnPoint);
            Vector3 fugitiveSpawnPos = GetSpawnPosition(fugitiveSpawnPoint);
            ApplySpawnJitter(ref pursuerSpawnPos, ref fugitiveSpawnPos);

            if (pursuer == null)
            {
                pursuer = SpawnMario("Pursuer", ChaseRole.Pursuer, pursuerSpawnPos, pursuerRuntimeMaterial, PursuerBehaviorName);
            }
            if (fugitive == null)
            {
                fugitive = SpawnMario("Fugitive", ChaseRole.Fugitive, fugitiveSpawnPos, fugitiveRuntimeMaterial, FugitiveBehaviorName);
            }

            if (pursuer != null)
            {
                pursuer.gameObject.SetActive(true);
                SM64Mario sm64 = pursuer.GetComponent<SM64Mario>();
                if (sm64 != null) sm64.Teleport(pursuerSpawnPos);
                else pursuer.transform.position = pursuerSpawnPos;
            }
            if (fugitive != null)
            {
                fugitive.gameObject.SetActive(true);
                SM64Mario sm64 = fugitive.GetComponent<SM64Mario>();
                if (sm64 != null) sm64.Teleport(fugitiveSpawnPos);
                else fugitive.transform.position = fugitiveSpawnPos;
            }

            if (pursuer != null && fugitive != null)
            {
                pursuer.opponent = fugitive;
                fugitive.opponent = pursuer;
                duelActive = true;
                lastSpawnTime = Time.time;
                battleStartTime = Time.time;
                pursuer.ResetForNewEpisode();
                fugitive.ResetForNewEpisode();
                if (enableEpisodeLogs)
                {
                    Debug.Log($"[ChaseTraining] Episodio iniciado: Pursuer vs Fugitive. Distancia inicial={(Vector3.Distance(pursuerSpawnPos, fugitiveSpawnPos)):F2}");
                }
            }
        }

        private void EnsureRuntimeVisualResources(Material matBase)
        {
            if (pursuerRuntimeMaterial == null)
            {
                Material source = pursuerMaterial != null ? pursuerMaterial : matBase;
                pursuerRuntimeMaterial = CreateRuntimeMaterial(source, pursuerColor, "Pursuer", out pursuerRuntimeTexture);
            }

            if (fugitiveRuntimeMaterial == null)
            {
                Material source = fugitiveMaterial != null ? fugitiveMaterial : matBase;
                fugitiveRuntimeMaterial = CreateRuntimeMaterial(source, fugitiveColor, "Fugitive", out fugitiveRuntimeTexture);
            }
        }

        private Material CreateRuntimeMaterial(Material sourceMaterial, Color color, string label, out Texture2D texture)
        {
            texture = null;
            if (sourceMaterial == null)
                return null;

            Material mat = new Material(sourceMaterial);
            mat.name = $"ChaseMario_{label}";

            texture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            texture.name = $"ChaseTex_{label}";
            texture.SetPixel(0, 0, color);
            texture.SetPixel(0, 1, color);
            texture.SetPixel(1, 0, color);
            texture.SetPixel(1, 1, color);
            texture.Apply();

            mat.mainTexture = texture;
            mat.SetTexture("_MainTex", texture);
            mat.color = Color.white;
            return mat;
        }

        private Material ResolveBaseMaterial()
        {
            Material matBase = baseMarioMaterial;
            if (matBase == null && marioPrefab != null)
            {
                SM64Mario prefabMario = marioPrefab.GetComponent<SM64Mario>();
                if (prefabMario != null)
                {
                    var matField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (matField != null)
                        matBase = matField.GetValue(prefabMario) as Material;
                }
            }
            return matBase;
        }

        private ChaseAgent SpawnMario(string label, ChaseRole role, Vector3 spawnPos, Material runtimeMaterial, string behaviorName)
        {
            GameObject marioObj = new GameObject($"Mario_{label}");
            marioObj.SetActive(false);
            marioObj.transform.position = spawnPos;

            ChaseAgent chaseAgent = marioObj.AddComponent<ChaseAgent>();
            chaseAgent.role = role;
            chaseAgent.chaseEnvironment = this;

            marioObj.AddComponent<MarioInputProvider>();
            SM64Mario sm64Mario = marioObj.AddComponent<SM64Mario>();
            ApplyMarioMaterial(marioObj, sm64Mario, runtimeMaterial);

            SphereCollider combatCollider = marioObj.AddComponent<SphereCollider>();
            combatCollider.radius = 1.5f;
            combatCollider.isTrigger = true;

            var bp = marioObj.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            if (bp == null)
            {
                bp = marioObj.AddComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            }
            bp.BehaviorName = behaviorName;
            bp.BehaviorType = Unity.MLAgents.Policies.BehaviorType.Default;
            // Compute observation size dynamically based on agent raycast count:
            // Base observations: 3 pos + 3 vel + 1 grounded + 7 opponent info + 1 time = 15
            int obsSize = 15;
            var chaseComp = chaseAgent as ChaseAgent;
            if (chaseComp != null)
            {
                obsSize += chaseComp.RaycastCount * 2;
            }
            bp.BrainParameters.VectorObservationSize = obsSize;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionSize);
            bp.TeamId = role == ChaseRole.Pursuer ? 0 : 1;

            var dr = marioObj.GetComponent<Unity.MLAgents.DecisionRequester>();
            if (dr == null)
            {
                dr = marioObj.AddComponent<Unity.MLAgents.DecisionRequester>();
            }
            dr.DecisionPeriod = 8;
            dr.TakeActionsBetweenDecisions = true;

            if (enableEpisodeLogs)
            {
                Debug.Log($"[ChaseTraining] Configurado {label}: behavior={bp.BehaviorName}, cont={bp.BrainParameters.ActionSpec.NumContinuousActions}, disc={bp.BrainParameters.ActionSpec.NumDiscreteActions}, team={bp.TeamId}");
            }

            marioObj.SetActive(true);
            return chaseAgent;
        }

        private void SanitizeBehaviorParameters()
        {
            BehaviorParameters[] allBehaviorParameters = FindObjectsOfType<BehaviorParameters>();
            foreach (BehaviorParameters bp in allBehaviorParameters)
            {
                if (bp == null)
                    continue;

                // Skip already sanitized instances
                int id = bp.GetInstanceID();
                if (sanitizedBPInstanceIDs.Contains(id))
                    continue;

                // Only sanitize BehaviorParameters that belong to our environment-managed agents.
                // Criteria: has a ChaseAgent on the same GameObject OR the GameObject name follows the spawned Mario_* pattern.
                var go = bp.gameObject;
                bool belongsToEnvironment = false;
                if (go != null)
                {
                    if (go.GetComponent<ChaseAgent>() != null)
                        belongsToEnvironment = true;
                    else if (!string.IsNullOrEmpty(go.name) && go.name.StartsWith("Mario_"))
                        belongsToEnvironment = true;
                }

                if (!belongsToEnvironment)
                    continue;

                ActionSpec spec = bp.BrainParameters.ActionSpec;
                if (spec.NumContinuousActions <= 0 && spec.NumDiscreteActions <= 0)
                {
                    bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(ContinuousActionSize);
                    if (string.IsNullOrWhiteSpace(bp.BehaviorName))
                    {
                        bp.BehaviorName = PursuerBehaviorName;
                    }

                    // Log the correction once per behavior name to avoid console spam
                    string bname = string.IsNullOrWhiteSpace(bp.BehaviorName) ? "<unnamed>" : bp.BehaviorName;
                    if (!warnedBehaviorNames.Contains(bname))
                    {
                        Debug.LogWarning($"[ChaseTraining] ActionSpec vazio detectado e corrigido em '{bname}'.");
                        warnedBehaviorNames.Add(bname);
                    }
                }

                sanitizedBPInstanceIDs.Add(id);
            }
        }

        private void ApplyMarioMaterial(GameObject marioObj, SM64Mario sm64Mario, Material runtimeMaterial)
        {
            if (runtimeMaterial == null)
                return;

            sm64Mario.useCustomTexture = true;
            sm64Mario.tintColor = Color.white;

            var materialField = typeof(SM64Mario).GetField("material", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (materialField != null)
                materialField.SetValue(sm64Mario, runtimeMaterial);

            Transform rendererChild = marioObj.transform.Find("MARIO");
            if (rendererChild != null)
            {
                MeshRenderer mr = rendererChild.GetComponent<MeshRenderer>();
                if (mr != null)
                    mr.material = runtimeMaterial;
            }
        }

        private Vector3 GetSpawnPosition(Transform spawnPoint)
        {
            if (spawnPoint != null)
                return spawnPoint.position + Vector3.up;

            return Vector3.up * 2f;
        }

        private void ProcessCombatCollision()
        {
            if (pursuer == null || fugitive == null)
                return;

            // Evita captura instantanea durante o settling do spawn/fisica.
            if (Time.time - lastSpawnTime < catchGraceAfterSpawn)
                return;

            float dist = Vector3.Distance(pursuer.transform.position, fugitive.transform.position);

            // O pursuer só vence se executar um ataque (kick ou stomp) E estiver próximo o suficiente
            bool attackLanded = false;
            if (dist <= attackDistance && pursuer.attackExecutedThisFrame)
            {
                // Kick precisa estar mais próximo que stomp
                if (pursuer.lastKickPressed && dist <= kickHitDistance)
                    attackLanded = true;
                if (pursuer.lastStompPressed && dist <= attackDistance)
                    attackLanded = true;
            }

            if (!attackLanded)
                return;

            // Aplicar knockback no fugitivo para feedback visual/fisico
            Vector3 toFugitive = fugitive.transform.position - pursuer.transform.position;
            Vector3 knockbackDir = toFugitive.sqrMagnitude > 0.001f ? toFugitive.normalized : Vector3.forward;
            ApplyKnockback(fugitive, knockbackDir * knockbackForce);

            pursuer.OnSuccessfulHit(0.8f);
            fugitive.AddReward(-1.2f);
            ResolvePursuerVictory();
        }

        private void ApplyKnockback(ChaseAgent agent, Vector3 force)
        {
            if (agent == null) return;
            SM64Mario sm64 = agent.GetComponent<SM64Mario>();
            if (sm64 != null)
            {
                Vector3 newPos = agent.transform.position + force;
                sm64.Teleport(newPos);
            }
            else
            {
                agent.transform.position += force;
            }
        }

        private void ResolvePursuerVictory()
        {
            if (!duelActive)
                return;

            duelActive = false;
            pursuer?.AddReward(8f);
            fugitive?.AddReward(-8f);
            EndAndReset();
        }

        public void ResolveTimeout()
        {
            if (!duelActive)
                return;

            duelActive = false;
            pursuer?.AddReward(-6f);
            fugitive?.AddReward(6f);
            EndAndReset();
        }

        private void EndAndReset()
        {
            if (pursuer != null && pursuer.isActiveAndEnabled)
                pursuer.EndEpisode();
            if (fugitive != null && fugitive.isActiveAndEnabled)
                fugitive.EndEpisode();

            CancelInvoke(nameof(ResetDuel));
            Invoke(nameof(ResetDuel), 1f);
        }

        private void ResetDuel()
        {
            duelActive = true;
            SpawnDuelAgents();
        }

        private void ApplySpawnJitter(ref Vector3 pursuerPos, ref Vector3 fugitivePos)
        {
            Vector2 pursuerJitter = Random.insideUnitCircle * spawnJitterRadius;
            Vector2 fugitiveJitter = Random.insideUnitCircle * spawnJitterRadius;

            pursuerPos += new Vector3(pursuerJitter.x, 0f, pursuerJitter.y);
            fugitivePos += new Vector3(fugitiveJitter.x, 0f, fugitiveJitter.y);

            float separation = Vector3.Distance(pursuerPos, fugitivePos);
            if (separation < minimumSpawnSeparation)
            {
                Vector3 away = (fugitivePos - pursuerPos);
                away.y = 0f;
                if (away.sqrMagnitude < 0.001f)
                    away = Vector3.right;
                away.Normalize();
                float needed = minimumSpawnSeparation - separation;
                fugitivePos += away * needed;
            }
        }

        public bool IsOutOfBounds(Vector3 position)
        {
            if (arenaCenter == null)
                return false;
            return Vector3.Distance(position, arenaCenter.position) > arenaRadius;
        }

        public float EpisodeTimeoutSeconds => maxBattleTime;
    }
}

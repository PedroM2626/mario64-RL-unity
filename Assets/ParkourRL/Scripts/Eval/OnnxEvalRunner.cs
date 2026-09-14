using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Unity.MLAgents.Sensors;
using Unity.Barracuda;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace ParkourRL
{
    /// <summary>
    /// In-game ONNX evaluation for the PPO benchmark (Barracuda, CPU).
    /// Drives the 3 CompetitiveParkour Marios with exported policies through the
    /// SAME 42-dim observations as training (reuses MarioCompetitiveAgent.
    /// CollectObservations via VectorSensor — zero obs divergence by construction)
    /// and records framework-agnostic metrics: goal success, completion time,
    /// min distance to goal, episode length.
    ///
    /// One eval build serves all frameworks via CLI (no rebake per model).
    /// Models are baked as NNModel assets (Unity's built-in ONNX importer — the
    /// same path ML-Agents itself uses) and selected by KEY:
    ///   CompetitiveParkourEval.exe -nographics -batchmode
    ///     -evalModel sb3 [-evalModel1 ...] [-evalModel2 ...]
    ///       keys: sb3 | cleanrl | rllib | mlagents  (slot default = slot0 key)
    ///     -evalEpisodes 20 -evalOut D:\path\eval_sb3.csv [-evalTimeoutMin 30]
    ///
    /// OBSERVER MODE (the official benchmark-eval protocol):
    ///   ... -evalObserve 1 -evalEpisodes 60 -evalOut D:\path\eval_sb3.csv
    ///   (no -evalModel needed) *while a Python bridge drives the agents* via a
    ///   trainer connection (evaluate.py --env <same exe> --additional-args ...).
    ///   Python = policy inference for ANY framework (proven path, zero Unity
    ///   inference code in the loop); Unity = official metrics (exact goal
    ///   success + completion time). Same underlying episodes on both sides.
    ///
    /// Rationale: runtime ONNXModelConverter proved unreliable for
    /// torch-exported MLPs in Barracuda 3.0 (degenerate 8D shapes with dynamic
    /// batch dims; garbage constants with the optimizer on). Baking via the
    /// Editor importer sidesteps all of it.
    ///
    /// Model format (auto-detected per model):
    ///   1 output of dim 5  -> Box(5) convention (SB3/CleanRL/RLlib exports):
    ///                           [joyX, joyY, jump, kick, stomp], buttons >0.
    ///   2 outputs (2 + 3)  -> ML-Agents hybrid export ([1,2] continuous +
    ///                           [1,3] discrete 0/1 branches).
    /// Anything else -> error in log + zero actions (fail-loud in CSV header).
    /// </summary>
    public class OnnxEvalRunner : MonoBehaviour
    {
        [Header("Baked eval models (keys: sb3 | cleanrl | rllib | mlagents)")]
        [Tooltip("Assign in Inspector: NNModel assets under Assets/ParkourRL/Models/Eval/")]
        public NNModel sb3Model;
        public NNModel cleanrlModel;
        public NNModel rllibModel;
        public NNModel mlagentsModel;
        private class Slot
        {
            public MarioCompetitiveAgent agent;
            public IWorker worker;
            public List<string> outputNames = new List<string>();
            public string inputName;
            public bool singleBox5;
            public bool verifiedFormat;
            public bool loggedFirst;
            public Tensor pendingInput; // previous frame's input, disposed on next call
            public string maskName; // hybrid models only: action_masks input name
            public Tensor maskTensor; // all-ones (all discrete actions enabled)
            public int episodes;
            public bool prevFinished;
            public float lastTime;
            public float minDist;
            public int decisions;
            public float finishTime;
        }

        private List<Slot> slots = new List<Slot>();
        // Cached reflection into ML-Agents 2.0.2 VectorSensor (see FixedUpdate).
        private static readonly FieldInfo obsField =
            typeof(VectorSensor).GetField("m_Observations",
                BindingFlags.NonPublic | BindingFlags.Instance);
        private int targetEpisodes = 20;
        private string outPath;
        private float timeoutSec = 1800f;
        private float startTime;
        private bool done;
        private bool initialized; // set at end of Init; FixedUpdate waits for it
        private int frame; // FixedUpdate counter; inference runs every 5th (decision cadence)
        private List<string> rows = new List<string>();
        // Observer mode (-evalObserve 1, no -evalModel needed): skip ALL inference,
        // only track episode boundaries/metrics. Designed to run WHILE a Python
        // trainer/bridge drives the agents (same player, trainer connection active):
        // Python = policy inference for any framework, Unity = official metrics
        // (exact goal success + completion time, no proxies). Same episodes both sides.
        private bool observeOnly;
        private System.IO.StreamWriter csvWriter; // incremental flush (survives kills)

        void Start()
        {
            string m0 = GetArg("-evalModel", null);
            observeOnly = HasArg("-evalObserve");
            if (string.IsNullOrEmpty(m0) && !observeOnly)
            {
                Debug.Log("[OnnxEval] No -evalModel given; runner disabled (training scene unaffected).");
                enabled = false;
                return;
            }
            // Agents spawn ~0.5s after env Start (Invoke); wait for them.
            StartCoroutine(WaitForAgentsAndInit(m0));
        }

        private System.Collections.IEnumerator WaitForAgentsAndInit(string m0)
        {
            float waited = 0f;
            var agents = new MarioCompetitiveAgent[0];
            while (waited < 30f)
            {
                agents = FindObjectsOfType<MarioCompetitiveAgent>().OrderBy(a => a.teamId).ToArray();
                if (agents.Length > 0) break;
                yield return new WaitForSeconds(0.5f);
                waited += 0.5f;
            }
            if (agents.Length == 0)
            {
                Debug.LogError("[OnnxEval] No MarioCompetitiveAgent found after 30s.");
                enabled = false;
                yield break;
            }
            Init(m0, agents);
        }

        private void Init(string m0, MarioCompetitiveAgent[] agents)
        {
            string m1 = GetArg("-evalModel1", m0);
            string m2 = GetArg("-evalModel2", m0);
            targetEpisodes = GetArgInt("-evalEpisodes", 20);
            timeoutSec = GetArgFloat("-evalTimeoutMin", 30f) * 60f;
            outPath = GetArg("-evalOut", System.IO.Path.Combine(Application.persistentDataPath, "eval_results.csv"));
            startTime = Time.time;

            string[] keys = { m0, m1, m2 };
            for (int i = 0; i < agents.Length && i < 3; i++)
            {
                var slot = new Slot { agent = agents[i], lastTime = 0f, minDist = float.MaxValue };
                if (!observeOnly)
                {
                    try { LoadModel(slot, keys[i]); }
                    catch (System.Exception e)
                    {
                        Debug.LogError($"[OnnxEval] Failed to load '{keys[i]}': {e.Message}");
                        enabled = false;
                        return;
                    }
                    Debug.Log($"[OnnxEval] Slot {i} ({agents[i].name}) <- '{keys[i]}' " +
                              (slot.singleBox5 ? "[Box5]" : $"[hybrid:{string.Join(",", slot.outputNames)}]"));
                }
                slots.Add(slot);
            }
            rows.Add("episode,slot,success,time_s,min_dist,length");
            csvWriter = new System.IO.StreamWriter(outPath, false);
            csvWriter.WriteLine("episode,slot,success,time_s,min_dist,length");
            csvWriter.Flush();
            Debug.Log($"[OnnxEval] {(observeOnly ? "OBSERVER" : "INFERENCE")} mode: " +
                      $"{slots.Count} slots x {targetEpisodes} episodes -> {outPath}");
            initialized = true;
        }

        private NNModel ResolveKey(string key)
        {
            switch (key.Trim().ToLowerInvariant())
            {
                case "sb3": return sb3Model;
                case "cleanrl": return cleanrlModel;
                case "rllib": return rllibModel;
                case "mlagents": return mlagentsModel;
                default: throw new System.ArgumentException(
                    $"unknown model key '{key}' (use sb3|cleanrl|rllib|mlagents)");
            }
        }

        private void LoadModel(Slot slot, string key)
        {
            // Baked NNModel path: Unity's Editor ONNX importer (the same one
            // ML-Agents uses) converts at import time. Runtime ONNXModelConverter
            // proved unreliable for torch graphs in Barracuda 3.0, so we don't
            // convert at runtime at all.
            NNModel nn = ResolveKey(key);
            if (nn == null)
                throw new System.ArgumentException($"NNModel asset for key '{key}' is not assigned in the Inspector.");
            var model = ModelLoader.Load(nn);
            // Type.Auto lets Barracuda pick the backend.
            slot.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, model);
            slot.outputNames = new List<string>(model.outputs);
            slot.inputName = model.inputs.Count > 0 ? model.inputs[0].name : "vector_observation";
            Debug.Log($"[OnnxEval] worker={slot.worker.GetType().Name} " +
                      $"inputs=[{string.Join(";", model.inputs.ConvertAll(i => i.name + ":" + string.Join("x", i.shape)))}] " +
                      $"outputs=[{string.Join(";", slot.outputNames)}] layers={model.layers.Count}");
            // Box5: single output of dim 5. ML-Agents hybrid: outputs dims 2 and 3.
            // PeekOutput before the first Execute may be unallocated, so default by
            // count and re-verify lazily on first inference (see Infer).
            slot.singleBox5 = (slot.outputNames.Count == 1);
            slot.verifiedFormat = false;
            // Hybrid models take an action_masks input (ML-Agents discrete masking).
            // Feed all-ones = every button enabled. Mask size = sum of branch sizes
            // = 2+2+2 = 6 for this project's ActionSpec(2, [2,2,2]).
            if (!slot.singleBox5)
            {
                string mask = null;
                foreach (var inp in model.inputs)
                    if (inp.name.ToLowerInvariant().Contains("mask")) { mask = inp.name; break; }
                if (mask != null)
                {
                    slot.maskName = mask;
                    slot.maskTensor = new Tensor(1, 1, 1, 6);
                    for (int i = 0; i < 6; i++) slot.maskTensor[0, 0, 0, i] = 1f;
                }
                else Debug.LogWarning("[OnnxEval] Hybrid model without action_masks input; discrete may misbehave.");
            }
        }

        void FixedUpdate()
        {
            // Init runs in a coroutine (waits for agent spawn); ignore frames before it.
            // (Without this, slots.All() on the empty list is vacuously true and the
            // run would Finish instantly with an empty CSV.)
            if (done || !initialized) return;
            if (Time.time - startTime > timeoutSec)
            {
                Debug.LogWarning("[OnnxEval] Timeout — writing partial CSV.");
                Finish(2);
                return;
            }
            frame++;
            foreach (var slot in slots)
            {
                if (slot.episodes >= targetEpisodes) continue;
                // Inject at decision cadence (see SetEvalAction docs). Skipped entirely
                // in observer mode (a Python bridge drives via the trainer connection).
                if (!observeOnly && frame % 5 == 0)
                {
                    // 1) observe with the SAME code as training. ML-Agents 2.0.2's
                    // VectorSensor is write-only from the outside, so read back its
                    // private m_Observations via reflection (same pattern this repo
                    // already uses for SM64Mario's private material field).
                    var sensor = new VectorSensor(42);
                    slot.agent.CollectObservations(sensor);
                    var stored = (List<float>)obsField.GetValue(sensor);
                    float[] obs = stored.ToArray();
                    // 2) infer
                    float[] act = Infer(slot, obs);
                    // 3) inject (also refreshes previousPosition bookkeeping)
                    slot.agent.SetEvalAction(act[0], act[1], act[2] > 0f, act[3] > 0f, act[4] > 0f);
                }
                // 4) track episode boundary (time reset) + success
                bool fin = slot.agent.HasFinished;
                float t = slot.agent.EpisodeTimeElapsed;
                float d = slot.agent.GetDistanceToGoal();
                if (d < slot.minDist) slot.minDist = d;
                slot.decisions++;
                if (fin && !slot.prevFinished) slot.finishTime = t;
                if (t < slot.lastTime - 1e-3f)
                {
                    // boundary: previous episode ended. InvariantCulture: the OS
                    // locale may use ',' as decimal separator (pt-BR), which would
                    // corrupt the CSV — format explicitly.
                    var inv = System.Globalization.CultureInfo.InvariantCulture;
                    bool success = slot.prevFinished;
                    string row = string.Format(inv, "{0},{1},{2},{3:F2},{4:F2},{5}",
                        slot.episodes, slot.agent.teamId, (success ? 1 : 0),
                        (success ? slot.finishTime : t), slot.minDist, slot.decisions);
                    rows.Add(row);
                    try { csvWriter.WriteLine(row); csvWriter.Flush(); } catch { }
                    slot.episodes++;
                    slot.minDist = float.MaxValue;
                    slot.decisions = 0;
                    slot.prevFinished = false;
                }
                else
                {
                    slot.prevFinished = fin;
                }
                slot.lastTime = t;
            }
            if (slots.All(s => s.episodes >= targetEpisodes)) Finish(0);
        }

        private float[] Infer(Slot slot, float[] obs)
        {
            // Layout channels-last [1,1,1,42]: the ONNX importer maps torch's
            // [batch, features] to NHWC with features on channels.
            var input = new Tensor(1, 1, 1, 42);
            for (int i = 0; i < 42 && i < obs.Length; i++) input[0, 0, 0, i] = obs[i];
            // Named-dict Execute exactly like ML-Agents' ModelRunner, and keep the
            // input alive until the NEXT call (never dispose a tensor the worker
            // may still reference).
            if (slot.pendingInput != null) { try { slot.pendingInput.Dispose(); } catch { } }
            slot.pendingInput = input;
            if (slot.maskTensor != null)
                slot.worker.Execute(new Dictionary<string, Tensor> { { slot.inputName, input }, { slot.maskName, slot.maskTensor } });
            else
                slot.worker.Execute(new Dictionary<string, Tensor> { { slot.inputName, input } });
            float[] act = new float[5];
            if (!slot.verifiedFormat)
            {
                // Re-verify now that tensors are allocated.
                if (slot.outputNames.Count == 1)
                    slot.singleBox5 = (slot.worker.PeekOutput(slot.outputNames[0]).length == 5);
                slot.verifiedFormat = true;
                Debug.Log($"[OnnxEval] {slot.agent.name} format: " +
                          (slot.singleBox5 ? "Box5" : $"hybrid:{string.Join(",", slot.outputNames)}"));
            }
            if (slot.singleBox5)
            {
                float[] o = slot.worker.PeekOutput(slot.outputNames[0]).ToReadOnlyArray();
                for (int i = 0; i < 5 && i < o.Length; i++) act[i] = o[i];
            }
            else
            {
                // ML-Agents hybrid: find [1,2]-like continuous + [1,3]-like discrete outputs
                foreach (var name in slot.outputNames)
                {
                    float[] o = slot.worker.PeekOutput(name).ToReadOnlyArray();
                    if (o.Length == 2) { act[0] = o[0]; act[1] = o[1]; }
                    else if (o.Length == 3) { act[2] = o[0]; act[3] = o[1]; act[4] = o[2]; }
                }
            }
            if (!slot.loggedFirst)
            {
                // NOTE: log AFTER reading outputs (an earlier version logged the
                // zero-initialized array here and sent us on a false trail).
                slot.loggedFirst = true;
                string obsStr = string.Join(",", obs.Take(8).Select(v => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)));
                string actStr = string.Join(",", act.Select(v => v.ToString("F3", System.Globalization.CultureInfo.InvariantCulture)));
                Debug.Log($"[OnnxEval] {slot.agent.name} first obs[0..7]=[{obsStr}] act=[{actStr}] pos={slot.agent.transform.position}");
            }
            input.Dispose();
            return act;
        }

        private void Finish(int code)
        {
            done = true;
            try { csvWriter.Flush(); csvWriter.Close(); }
            catch (System.Exception e) { Debug.LogError($"[OnnxEval] CSV write failed: {e.Message}"); }
            Debug.Log($"[OnnxEval] Wrote {rows.Count - 1} episode rows to {outPath}");
            foreach (var slot in slots)
            {
                try { slot.worker?.Dispose(); } catch { }
                try { slot.maskTensor?.Dispose(); } catch { }
                try { slot.pendingInput?.Dispose(); } catch { }
            }
#if UNITY_EDITOR
            EditorApplication.Exit(code);
#else
            Application.Quit(code);
#endif
        }

        private static string GetArg(string name, string fallback)
        {
            var args = System.Environment.GetCommandLineArgs();
            int i = System.Array.IndexOf(args, name);
            return (i >= 0 && i + 1 < args.Length) ? args[i + 1] : fallback;
        }
        private static bool HasArg(string name)
        {
            return System.Array.IndexOf(System.Environment.GetCommandLineArgs(), name) >= 0;
        }
        private static int GetArgInt(string name, int fallback)
        {
            int v; return int.TryParse(GetArg(name, null), out v) ? v : fallback;
        }
        private static float GetArgFloat(string name, float fallback)
        {
            float v;
            return float.TryParse(GetArg(name, null),
                System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out v) ? v : fallback;
        }
    }
}

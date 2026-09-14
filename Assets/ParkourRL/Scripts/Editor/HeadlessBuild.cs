using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using System.Linq;
using Unity.Barracuda;
#endif

namespace ParkourRL
{
    /// <summary>
    /// Headless player build for ML-Agents / SB3 training (no Editor UI needed at run time).
    /// Invoked from CLI, e.g.:
    ///   Unity.exe -batchmode -nographics -projectPath D:\libsm64-unity-master
    ///     -executeMethod ParkourRL.HeadlessBuild.BuildCompetitive
    ///     -buildOut Builds/CompetitiveHeadless/CompetitiveParkour.exe -quit -logFile Builds/build.log
    /// Optional CLI args:
    ///   -buildScene Assets/ParkourRL/Scenes/CompetitiveParkour.unity
    ///   -buildOut   <output exe path>
    /// </summary>
    public static class HeadlessBuild
    {
#if UNITY_EDITOR
        private const string DefaultScene = "Assets/ParkourRL/Scenes/CompetitiveParkour.unity";

        public static void BuildCompetitive()
        {
            EnsureEvalRunner();
            string scene = GetArg("-buildScene", DefaultScene);
            string outPath = GetArg("-buildOut", "Builds/CompetitiveHeadless/CompetitiveParkour.exe");

            Debug.Log($"[HeadlessBuild] Scene: {scene}");
            Debug.Log($"[HeadlessBuild] Output: {outPath}");

            var options = new BuildPlayerOptions
            {
                scenes = new[] { scene },
                // Normal Windows64 player (windowsstandalonesupport module).
                // Run it with no_graphics=True from Python for headless speed.
                locationPathName = outPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };

            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"[HeadlessBuild] Result: {report.summary.result} " +
                      $"in {report.summary.totalTime.TotalSeconds:F0}s -> {report.summary.outputPath}");
            if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
            {
                Debug.LogError("[HeadlessBuild] BUILD FAILED — see log above.");
                EditorApplication.Exit(1);
            }

            // libsm64 loads baserom.us.z64 from the folder containing the player
            // binary (SM64Context). Without it Mario spawns frozen: ML-Agents
            // observations/rewards still flow, but actions have no effect and any
            // training run is silently invalid. Copy it automatically.
            try
            {
                string projectRoot = System.IO.Path.GetDirectoryName(Application.dataPath);
                string romSrc = System.IO.Path.Combine(projectRoot, "baserom.us.z64");
                string romDst = System.IO.Path.Combine(
                    System.IO.Path.GetDirectoryName(outPath), "baserom.us.z64");
                if (System.IO.File.Exists(romSrc))
                {
                    System.IO.File.Copy(romSrc, romDst, overwrite: true);
                    Debug.Log($"[HeadlessBuild] ROM copied to {romDst}");
                }
                else
                {
                    Debug.LogWarning($"[HeadlessBuild] ROM not found at {romSrc} — " +
                        "copy baserom.us.z64 next to the .exe or Mario will be frozen.");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[HeadlessBuild] ROM copy failed: {e.Message}");
            }
        }

        private static string GetArg(string name, string fallback)
        {
            var args = System.Environment.GetCommandLineArgs();
            int i = System.Array.IndexOf(args, name);
            if (i >= 0 && i + 1 < args.Length)
                return args[i + 1];
            return fallback;
        }

        /// <summary>
        /// Ensures the CompetitiveParkour scene contains the OnnxEvalRunner (disabled
        /// at runtime unless -evalModel is passed, so training is unaffected).
        /// Idempotent: creates the GameObject only if missing, then saves the scene.
        /// </summary>
        public static void EnsureEvalRunner()
        {
            string scene = GetArg("-buildScene", DefaultScene);
            var unityScene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(scene);
            var existing = Object.FindObjectOfType<OnnxEvalRunner>();
            OnnxEvalRunner runner;
            if (existing == null)
            {
                var go = new GameObject("OnnxEvalRunner");
                runner = go.AddComponent<OnnxEvalRunner>();
                Debug.Log("[HeadlessBuild] Added OnnxEvalRunner to scene.");
            }
            else
            {
                runner = existing;
                Debug.Log("[HeadlessBuild] OnnxEvalRunner already in scene.");
            }
            // Wire baked NNModel assets (imported from Assets/ParkourRL/Models/Eval/*.onnx
            // by Unity's built-in ONNX importer). Missing files stay null and fail
            // loud at eval time only if their key is requested.
            runner.sb3Model = LoadEvalNNModel("sb3_seed0.onnx");
            runner.cleanrlModel = LoadEvalNNModel("cleanrl_seed0.onnx");
            runner.rllibModel = LoadEvalNNModel("rllib_seed0.onnx");
            runner.mlagentsModel = LoadEvalNNModel("mlagents_ppo.onnx");
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(unityScene);
            UnityEditor.SceneManagement.EditorSceneManager.SaveScene(unityScene);
        }

        private static NNModel LoadEvalNNModel(string fileName)
        {
            string path = "Assets/ParkourRL/Models/Eval/" + fileName;
            var nn = UnityEditor.AssetDatabase.LoadAssetAtPath<NNModel>(path);
            if (nn == null)
                Debug.LogWarning($"[HeadlessBuild] Eval NNModel not found (ok until needed): {path}");
            else
                Debug.Log($"[HeadlessBuild] Eval model wired: {path}");
            return nn;
        }
#else
        public static void BuildCompetitive() { }
#endif
    }
}

using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Grava dados para Imitation Learning e Offline RL.
    /// Saves transitions (obs, action, reward, next_obs, done) in JSON and CSV format.
    /// </summary>
    public class HybridDataRecorder : MonoBehaviour
    {
        [Header("Output Settings")]
        [Tooltip("Base directory to save data")]
        [SerializeField] private string outputDirectory = "HybridTrainingData";
        
        [Tooltip("Output format: JSON, CSV, or both")]
        [SerializeField] private OutputFormat outputFormat = OutputFormat.Both;
        
        [Tooltip("File name (without extension)")]
        [SerializeField] private string fileName = "hybrid_episodes";
        
        [Tooltip("Maximum number of episodes per file")]
        [SerializeField] private int maxEpisodesPerFile = 100;

        [Header("Data Quality")]
        [Tooltip("Minimum interval between recordings (seconds)")]
        [SerializeField] private float minRecordInterval = 0.05f;  // 20 FPS
        
        [Tooltip("Maximum number of steps per episode")]
        [SerializeField] private int maxStepsPerEpisode = 2000;

        public enum OutputFormat
        {
            JSON,
            CSV,
            Both
        }

        // Estado
        private string currentFilePath;
        private int episodeCounter = 0;
        private int fileCounter = 0;
        private List<EpisodeData> episodeBatch = new List<EpisodeData>();
        private float lastRecordTime = 0f;
        private bool isRecording = false;
        private EpisodeData currentEpisode;

        // Eventos
        public event Action OnEpisodeStarted;
        public event Action<bool> OnEpisodeEnded;  // bool = success
        public event Action<string> OnDataSaved; // string = file path

        void Start()
        {
            InitializeDirectory();
        }

        private void InitializeDirectory()
        {
            string path = Path.Combine(Application.dataPath, "..", outputDirectory);
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
                Debug.Log($"[HybridRecorder] Directory created: {path}");
            }
            
            currentFilePath = GetNextFilePath();
        }

        private string GetNextFilePath()
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string fileNameWithCounter = $"{fileName}_{timestamp}_batch{fileCounter}";
            return Path.Combine(Application.dataPath, "..", outputDirectory, fileNameWithCounter);
        }

        public void StartEpisode()
        {
            if (Time.time - lastRecordTime < minRecordInterval)
                return;
            
            isRecording = true;
            currentEpisode = new EpisodeData
            {
                episodeId = episodeCounter,
                startTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                startTimestamp = Time.time,
                transitions = new List<TransitionData>()
            };
            
            Debug.Log($"[HybridRecorder] Episode {episodeCounter} started");
            OnEpisodeStarted?.Invoke();
        }

        public void RecordStep(float[] observations, float[] actions, float reward, 
                              float[] nextObservations, bool done, float? extraValue = null)
        {
            if (!isRecording || currentEpisode == null) return;
            
            if (Time.time - lastRecordTime < minRecordInterval)
                return;
            
            if (currentEpisode.transitions.Count >= maxStepsPerEpisode)
                return;
            
            lastRecordTime = Time.time;
            
            var transition = new TransitionData
            {
                step = currentEpisode.transitions.Count,
                timestamp = Time.time,
                observations = observations,
                actions = actions,
                reward = reward,
                nextObservations = nextObservations,
                done = done,
                extraValue = extraValue
            };
            
            currentEpisode.transitions.Add(transition);
        }

        public void SaveEpisode(MarioHybridAgent.HybridTransition[] transitions, bool success)
        {
            if (currentEpisode == null) return;
            
            currentEpisode.endTime = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            currentEpisode.duration = Time.time - currentEpisode.startTimestamp;
            currentEpisode.success = success;
            currentEpisode.totalReward = CalculateTotalReward(transitions);
            currentEpisode.stepCount = transitions.Length;
            
            // Converter transitions
            foreach (var t in transitions)
            {
                var trans = new TransitionData
                {
                    step = currentEpisode.transitions.Count,
                    timestamp = t.timestamp,
                    observations = t.observations,
                    actions = t.actions,
                    reward = t.reward,
                    done = t.done
                };
                currentEpisode.transitions.Add(trans);
            }
            
            episodeBatch.Add(currentEpisode);
            episodeCounter++;
            
            Debug.Log($"[HybridRecorder] Episode {currentEpisode.episodeId} saved. " +
                     $"Success={success}, Steps={currentEpisode.stepCount}, " +
                     $"Reward={currentEpisode.totalReward:F2}");
            
            OnEpisodeEnded?.Invoke(success);
            
            // Salvar batch se atingiu limite
            if (episodeBatch.Count >= maxEpisodesPerFile)
            {
                FlushBatch();
            }
            
            isRecording = false;
            currentEpisode = null;
        }

        private float CalculateTotalReward(MarioHybridAgent.HybridTransition[] transitions)
        {
            float total = 0f;
            foreach (var t in transitions)
            {
                total += t.reward;
            }
            return total;
        }

        public void FlushBatch()
        {
            if (episodeBatch.Count == 0) return;
            
            string jsonPath = currentFilePath + ".json";
            string csvPath = currentFilePath + ".csv";
            
            try
            {
                if (outputFormat == OutputFormat.JSON || outputFormat == OutputFormat.Both)
                {
                    SaveAsJSON(jsonPath);
                }
                
                if (outputFormat == OutputFormat.CSV || outputFormat == OutputFormat.Both)
                {
                    SaveAsCSV(csvPath);
                }
                
                Debug.Log($"[HybridRecorder] Batch saved: {episodeBatch.Count} episodes");
                OnDataSaved?.Invoke(currentFilePath);
                
                // Preparar next batch
                episodeBatch.Clear();
                fileCounter++;
                currentFilePath = GetNextFilePath();
            }
            catch (Exception e)
            {
                Debug.LogError($"[HybridRecorder] Error ao salvar: {e.Message}");
            }
        }

        private void SaveAsJSON(string path)
        {
            var wrapper = new DataWrapper
            {
                metadata = new DatasetMetadata
                {
                    createdAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                    episodeCount = episodeBatch.Count,
                    totalSteps = CalculateTotalSteps(),
                    averageReward = CalculateAverageReward(),
                    successRate = CalculateSuccessRate()
                },
                episodes = episodeBatch.ToArray()
            };
            
            string json = JsonUtility.ToJson(wrapper, true);
            File.WriteAllText(path, json);
        }

        private void SaveAsCSV(string path)
        {
            using (StreamWriter writer = new StreamWriter(path))
            {
                // Header
                writer.WriteLine("episode_id,step,timestamp,obs_0,obs_1,obs_2,obs_3,obs_4,obs_5,obs_6,obs_7,obs_8,obs_9,obs_10,obs_11,obs_12,obs_13,obs_14,obs_15,obs_16,obs_17,obs_18,obs_19,obs_20,obs_21,obs_22,obs_23,obs_24,obs_25,obs_26,obs_27,obs_28,obs_29,action_0,action_1,action_2,reward,done");
                
                // Data
                foreach (var episode in episodeBatch)
                {
                    foreach (var trans in episode.transitions)
                    {
                        var line = new List<string>
                        {
                            episode.episodeId.ToString(),
                            trans.step.ToString(),
                            trans.timestamp.ToString("F3")
                        };
                        
                        // Observations (30 dims)
                        for (int i = 0; i < 30; i++)
                        {
                            line.Add(i < trans.observations.Length ? 
                                trans.observations[i].ToString("F6") : "0");
                        }
                        
                        // Actions
                        for (int i = 0; i < 3; i++)
                        {
                            line.Add(i < trans.actions.Length ? 
                                trans.actions[i].ToString("F6") : "0");
                        }
                        
                        line.Add(trans.reward.ToString("F6"));
                        line.Add(trans.done ? "1" : "0");
                        
                        writer.WriteLine(string.Join(",", line));
                    }
                }
            }
        }

        private int CalculateTotalSteps()
        {
            int total = 0;
            foreach (var ep in episodeBatch)
            {
                total += ep.stepCount;
            }
            return total;
        }

        private float CalculateAverageReward()
        {
            if (episodeBatch.Count == 0) return 0f;
            
            float total = 0f;
            foreach (var ep in episodeBatch)
            {
                total += ep.totalReward;
            }
            return total / episodeBatch.Count;
        }

        private float CalculateSuccessRate()
        {
            if (episodeBatch.Count == 0) return 0f;
            
            int successes = 0;
            foreach (var ep in episodeBatch)
            {
                if (ep.success) successes++;
            }
            return (float)successes / episodeBatch.Count;
        }

        void OnDestroy()
        {
            FlushBatch();
        }

        // Serializable data structures
        [Serializable]
        private class DataWrapper
        {
            public DatasetMetadata metadata;
            public EpisodeData[] episodes;
        }

        [Serializable]
        private class DatasetMetadata
        {
            public string createdAt;
            public int episodeCount;
            public int totalSteps;
            public float averageReward;
            public float successRate;
        }

        [Serializable]
        private class EpisodeData
        {
            public int episodeId;
            public string startTime;
            public float startTimestamp;
            public string endTime;
            public float duration;
            public bool success;
            public int stepCount;
            public float totalReward;
            public List<TransitionData> transitions;
        }

        [Serializable]
        private class TransitionData
        {
            public int step;
            public float timestamp;
            public float[] observations;
            public float[] actions;
            public float reward;
            public float[] nextObservations;
            public bool done;
            public float? extraValue;
        }
    }
}

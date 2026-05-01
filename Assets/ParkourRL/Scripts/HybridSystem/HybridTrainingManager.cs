using UnityEngine;
using UnityEngine.UI;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Hybrid system manager: controls recording mode vs training mode.
    /// Interface para alternar entre Recording (IL/Offline) e Training (RL).
    /// </summary>
    public class HybridTrainingManager : MonoBehaviour
    {
        [Header("Mode Settings")]
        [Tooltip("Modo inicial ao iniciar a cena")]
        [SerializeField] private HybridMode initialMode = HybridMode.Training;
        
        [Tooltip("Permitir troca de modo em runtime (via UI ou teclas)")]
        [SerializeField] private bool allowModeSwitching = true;
        
        [Tooltip("Key to toggle mode")]
        [SerializeField] private KeyCode modeSwitchKey = KeyCode.M;

        [Header("UI References")]
        [SerializeField] private Text modeText;
        [SerializeField] private Text infoText;
        [SerializeField] private Button trainingButton;
        [SerializeField] private Button recordingButton;
        [SerializeField] private Toggle onlyRecordSuccessesToggle;

        [Header("Components")]
        [SerializeField] private HybridParkourEnvironment environment;
        [SerializeField] private HybridDataRecorder dataRecorder;

        // Estado
        private HybridMode currentMode;
        private bool isPaused = false;

        public enum HybridMode
        {
            Training,    // Treino RL normal
            Recording    // Recording for IL/Offline RL
        }

        public HybridMode CurrentMode => currentMode;

        void Start()
        {
            // Check if environment has a different mode set in inspector
            // Environment's choice takes priority
            if (environment != null)
            {
                bool envIsRecording = environment.CurrentMode == HybridParkourEnvironment.HybridMode.Recording;
                bool managerIsRecording = initialMode == HybridMode.Recording;
                
                if (envIsRecording != managerIsRecording)
                {
                    Debug.Log($"[HybridManager] Syncing with Environment mode: {environment.CurrentMode}");
                    currentMode = envIsRecording ? HybridMode.Recording : HybridMode.Training;
                }
                else
                {
                    currentMode = initialMode;
                }
            }
            else
            {
                currentMode = initialMode;
            }
            
            SetupUI();
            ApplyMode();
            
            Debug.Log($"[HybridManager] Modo inicial: {currentMode}");
        }

        void Update()
        {
            // Troca de modo via tecla
            if (allowModeSwitching && Input.GetKeyDown(modeSwitchKey))
            {
                ToggleMode();
            }
            
            // Pause/Resume
            if (Input.GetKeyDown(KeyCode.P))
            {
                TogglePause();
            }
            
            // Update UI
            UpdateUI();
        }

        private void SetupUI()
        {
            if (trainingButton != null)
            {
                trainingButton.onClick.AddListener(() => SetMode(HybridMode.Training));
            }
            
            if (recordingButton != null)
            {
                recordingButton.onClick.AddListener(() => SetMode(HybridMode.Recording));
            }
            
            if (onlyRecordSuccessesToggle != null && dataRecorder != null)
            {
                // Conectar toggle ao recorder (se houver acesso)
                onlyRecordSuccessesToggle.onValueChanged.AddListener(OnRecordSuccessesChanged);
            }
        }

        private void OnRecordSuccessesChanged(bool value)
        {
            // Buscar MarioHybridAgent e configurar
            var agents = FindObjectsOfType<MarioHybridAgent>();
            foreach (var agent in agents)
            {
                // Use reflection or public method if available
            }
        }

        public void SetMode(HybridMode mode)
        {
            if (currentMode == mode) return;
            
            currentMode = mode;
            
            // Propagate mode change to environment (user explicitly changed mode)
            if (environment != null)
            {
                var envMode = (mode == HybridMode.Recording) 
                    ? HybridParkourEnvironment.HybridMode.Recording 
                    : HybridParkourEnvironment.HybridMode.Training;
                environment.SetMode(envMode);
            }
            
            ApplyMode();
            
            Debug.Log($"[HybridManager] Mode changed to: {mode}");
        }

        public void ToggleMode()
        {
            SetMode(currentMode == HybridMode.Training ? HybridMode.Recording : HybridMode.Training);
        }

        private void ApplyMode()
        {
            // Note: We no longer force mode on environment
            // The environment's inspector choice has priority
            // Only sync if environment exists and modes differ (for runtime mode changes)
            if (environment != null)
            {
                bool envIsRecording = environment.CurrentMode == HybridParkourEnvironment.HybridMode.Recording;
                bool managerIsRecording = currentMode == HybridMode.Recording;
                
                if (envIsRecording != managerIsRecording)
                {
                    // Only apply if manager is explicitly changing mode (not on startup)
                    // This is handled by the caller (SetMode/ToggleMode)
                    Debug.Log($"[HybridManager] Mode mismatch with Environment. Env: {environment.CurrentMode}, Manager: {currentMode}");
                }
            }
            
            // Configurar recorder
            if (dataRecorder != null)
            {
                if (currentMode == HybridMode.Recording)
                {
                    // Activate recording
                    Debug.Log("[HybridManager] Recording ACTIVATED");
                }
                else
                {
                    // Finalize pending recording
                    Debug.Log("[HybridManager] Recording DEACTIVATED - Saving data...");
                    dataRecorder.FlushBatch();
                }
            }
            else
            {
                Debug.LogWarning("[HybridManager] DataRecorder is null! Cannot record data.");
            }
            
            // Atualizar UI
            UpdateUI();
        }

        private void UpdateUI()
        {
            if (modeText != null)
            {
                modeText.text = $"Mode: {currentMode}";
                modeText.color = currentMode == HybridMode.Training ? Color.blue : Color.red;
            }
            
            if (infoText != null)
            {
                string info = currentMode == HybridMode.Training 
                    ? "AI training with ML-Agents\nPlayer Mario for comparison"
                    : "RECORDING for IL/Offline RL\nControl green Mario with WASD+Space";
                
                infoText.text = info;
            }
            
            // Update buttons
            if (trainingButton != null)
            {
                trainingButton.interactable = currentMode != HybridMode.Training;
            }
            
            if (recordingButton != null)
            {
                recordingButton.interactable = currentMode != HybridMode.Recording;
            }
        }

        public void TogglePause()
        {
            isPaused = !isPaused;
            Time.timeScale = isPaused ? 0f : (currentMode == HybridMode.Training ? 1f : 1f);
            
            Debug.Log($"[HybridManager] {(isPaused ? "PAUSED" : "RESUMED")}");
        }

        public void ResetScene()
        {
            if (environment != null)
            {
                environment.ResetEnvironment();
            }
        }

        void OnGUI()
        {
            // Debug overlay
            if (GUI.Button(new Rect(Screen.width - 110, Screen.height - 40, 100, 30), "Reset"))
            {
                ResetScene();
            }
            
            if (allowModeSwitching)
            {
                if (GUI.Button(new Rect(Screen.width - 110, Screen.height - 80, 100, 30), "Toggle Mode"))
                {
                    ToggleMode();
                }
            }
            
            // Save button
            if (GUI.Button(new Rect(Screen.width - 110, Screen.height - 120, 100, 30), "Save Data"))
            {
                ForceSaveData();
            }
            
            // Instructions
            GUI.Label(new Rect(10, Screen.height - 30, 400, 20), 
                "M: Toggle Mode | P: Pause | WASD+Space: Control Player | Click 'Save Data' to save");
        }

        void OnDestroy()
        {
            // Ensure data is saved
            if (dataRecorder != null)
            {
                Debug.Log("[HybridManager] OnDestroy - Saving any pending data...");
                dataRecorder.FlushBatch();
            }
        }

        // Manual save method for testing
        public void ForceSaveData()
        {
            if (dataRecorder != null)
            {
                Debug.Log("[HybridManager] Force saving data...");
                dataRecorder.FlushBatch();
            }
            else
            {
                Debug.LogWarning("[HybridManager] Cannot save - DataRecorder is null!");
            }
        }
    }
}

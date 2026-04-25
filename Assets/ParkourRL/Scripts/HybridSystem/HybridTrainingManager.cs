using UnityEngine;
using UnityEngine.UI;

namespace ParkourRL.HybridSystem
{
    /// <summary>
    /// Gerenciador do sistema híbrido: controla modo de gravação vs treino.
    /// Interface para alternar entre Recording (IL/Offline) e Training (RL).
    /// </summary>
    public class HybridTrainingManager : MonoBehaviour
    {
        [Header("Mode Settings")]
        [Tooltip("Modo inicial ao iniciar a cena")]
        [SerializeField] private HybridMode initialMode = HybridMode.Training;
        
        [Tooltip("Permitir troca de modo em runtime (via UI ou teclas)")]
        [SerializeField] private bool allowModeSwitching = true;
        
        [Tooltip("Tecla para alternar modo")]
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
            Recording    // Gravação para IL/Offline RL
        }

        public HybridMode CurrentMode => currentMode;

        void Start()
        {
            currentMode = initialMode;
            
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
                // Usar reflection ou método público se disponível
            }
        }

        public void SetMode(HybridMode mode)
        {
            if (currentMode == mode) return;
            
            currentMode = mode;
            ApplyMode();
            
            Debug.Log($"[HybridManager] Modo alterado para: {mode}");
        }

        public void ToggleMode()
        {
            SetMode(currentMode == HybridMode.Training ? HybridMode.Recording : HybridMode.Training);
        }

        private void ApplyMode()
        {
            // Aplicar ao ambiente
            if (environment != null)
            {
                environment.SetMode(currentMode);
            }
            
            // Configurar recorder
            if (dataRecorder != null)
            {
                if (currentMode == HybridMode.Recording)
                {
                    // Ativar gravação
                    Debug.Log("[HybridManager] Gravação ATIVADA");
                }
                else
                {
                    // Finalizar gravação pendente
                    dataRecorder.FlushBatch();
                }
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
            
            // Atualizar botões
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
            
            // Instruções
            GUI.Label(new Rect(10, Screen.height - 30, 400, 20), 
                "M: Toggle Mode | P: Pause | WASD+Space: Control Player");
        }

        void OnDestroy()
        {
            // Garantir que dados são salvos
            if (dataRecorder != null && currentMode == HybridMode.Recording)
            {
                dataRecorder.FlushBatch();
            }
        }
    }
}

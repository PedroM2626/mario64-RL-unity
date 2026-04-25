# Quick Start - Hybrid Training

## 🚀 Começar em 3 Passos

### 1. Criar a Cena

No Unity Editor:
```
Menu: ParkourRL > Build Hybrid Training Scene
```

Isso cria automaticamente:
- 5 plataformas de parkour
- 2 pontos de spawn (Player e AI)
- Ambiente híbrido com todos os componentes
- Câmeras split-screen
- UI de controle

### 2. Configurar (se necessário)

```
Menu: ParkourRL > Setup Hybrid Scene References
```

Verifique no Inspector do **HybridEnvironment**:
- Goal → GoalPlatform
- Player Spawn Point → PlayerSpawn
- AI Spawn Point → AISpawn

### 3. Executar

Pressione **Play** no Unity!

---

## 🎮 Controles

| Tecla | Ação |
|-------|------|
| **WASD / Setas** | Mover Mario Player (verde) |
| **Espaço** | Pular |
| **M** | Alternar modo (Training/Recording) |
| **P** | Pausar |

---

## 📊 Modos de Operação

### Training Mode (Azul)
- AI Mario treina com ML-Agents
- Player Mario pode ser controlado para comparação
- Execute: `train_hybrid.bat`

### Recording Mode (Vermelho)
- Player Mario grava demonstrações
- Dados salvos em: `HybridTrainingData/`
- Use para: Behavior Cloning, CQL, IQL

---

## 🐍 Treinar Modelos Python

```bash
cd python_trainers

# Analisar dados
python analyze_dataset.py --data ../HybridTrainingData/ --plots

# Behavior Cloning
python train_behavior_cloning.py --data ../HybridTrainingData/ --output models/bc.pth

# Offline RL (CQL)
python train_offline_rl.py --data ../HybridTrainingData/ --algo CQL --output models/cql.pth
```

---

## 📁 Onde Estão os Arquivos?

| Tipo | Localização |
|------|-------------|
| Scripts C# | `Assets/ParkourRL/Scripts/HybridSystem/` |
| Config ML-Agents | `Assets/ParkourRL/Config/mario_parkour_hybrid.yaml` |
| Scripts Python | `python_trainers/` |
| Dados gravados | `HybridTrainingData/` (criado automaticamente) |
| Cena | `Assets/ParkourRL/Scenes/HybridTraining.unity` |

---

## ❓ Troubleshooting

### "Menu ParkourRL não aparece"
- Verifique se os scripts compilaram sem erros
- Console Window: `Ctrl+Shift+C`

### "Câmeras não aparecem"
- Certifique-se de que as câmeras foram criadas
- Check `GameObject > Cameras` na hierarquia

### "Dados não estão sendo gravados"
- Verifique se está em **Recording Mode** (tecla M)
- Check se o diretório `HybridTrainingData/` existe
- Console deve mostrar: "[HybridRecorder] Episódio X iniciado"

### "Player Mario não se move"
- Certifique-se de que a janela do Game está focada
- Verifique se não está em pausa (tecla P)

---

## 📚 Documentação Completa

- [Setup Detalhado](HybridSceneSetup.md)
- [README do Sistema](README_HYBRID.md)
- [Python Trainers](../python_trainers/README.md)

---

**Pronto para começar! Pressione Play e divirta-se! 🎮**

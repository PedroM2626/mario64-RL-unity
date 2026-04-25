# Sistema de Plataformas Dinâmicas

O sistema agora detecta **automaticamente** todas as plataformas na cena e recompensa o Mario por passar por cada uma.

## Como Funciona

### Detecção Automática
1. Encontra todos os objetos com `SM64StaticTerrain` na cena
2. Ignora o DeathFloor (chão de morte) e plataformas muito abaixo
3. Ordena do spawn para o goal (mais próxima → mais distante)
4. Recompensa (+15) ao chegar na área de cada plataforma

### Área da Plataforma (Não Coordenada Exata)
```
Raio de alcance = max(3m, 80% do tamanho da plataforma)
```
- Usa o `bounds` do collider da plataforma
- Verifica altura (se está realmente sobre a plataforma)
- Tolerância maior para plataformas grandes

### Área do Goal (Não Coordenada Exata)
```
Se o goal tiver trigger collider: usa bounds do collider
Se não: raio de 5 metros ao redor do goal
```

## Configuração no Inspector

**MarioRLAgent > Plataformas Dinâmicas:**
```
Auto Detect Platforms = true (marcado)
Platform Reward = 15
Platform Reach Radius = 3
Min Height Above Platform = 0.5
```

## Recompensas

| Ação | Recompensa |
|------|------------|
| Alcançar plataforma N | +15 |
| Alcançar goal | +50 |
| Progresso para goal | +por distância |
| Marco a cada 2m | +2 |
| Queda | -5 |
| Timeout | -5 + (progresso × 3) |

## Debug

No console Unity você verá:
```
[Mario] 4 plataformas detectadas e ordenadas
  Plataforma 0: Platform_1 em (12.0, -8.0, 0.0)
  Plataforma 1: Platform_2 em (27.4, -8.0, 0.0)
  Plataforma 2: GoalPlatform em (64.2, -8.0, 6.9)
[Mario] Plataforma 0 (Platform_1) alcançada! +15 reward | Dist: 1.2m, Raio: 3.0m
```

## Vantagens vs Waypoints Fixos

| Plataformas Dinâmicas | Waypoints Fixos |
|----------------------|-----------------|
| Detecta automaticamente | Precisa configurar manualmente |
| Adapta-se a cenas diferentes | Fixo por cena |
| Usa área real da plataforma | Coordenada específica |
| Tolerância proporcional ao tamanho | Raio fixo |

## Se Quiser Desativar

Desmarque `Auto Detect Platforms` no MarioRLAgent.
O sistema voltará a usar apenas o goal sem recompensas intermediárias.

# Configuração de Waypoints (Checkpoints)

Para corrigir o problema do Mario tentar ir direto ao goal em vez de seguir as plataformas:

## Opção 1: Waypoints Simples (Recomendado)

1. **No MarioRLAgent** (Inspector):
   - Expanda a seção "Waypoints/Checkpoints"
   - No campo `Waypoints`, adicione os transforms das plataformas intermediárias em ordem:
     - Element 0: Posição da primeira plataforma
     - Element 1: Posição da segunda plataforma
     - Element 2: Posição da terceira plataforma
     - etc...
   - Ajuste `Waypoint Reward` (padrão: 15)
   - Ajuste `Waypoint Reached Distance` (padrão: 3)

2. **O que acontece**:
   - Mario recebe +15 reward ao passar por cada waypoint
   - Só ganha se seguir a ordem correta (0 → 1 → 2 → ...)
   - Isso incentiva ele a passar por cada plataforma

## Opção 2: Checkpoints com Triggers (Alternativa)

1. **Na cena**:
   - Crie GameObjects vazios nas plataformas intermediárias
   - Adicione componente `Checkpoint` (script criado)
   - Configure `Checkpoint Index` (0, 1, 2, ... em ordem)
   - Adicione um Collider com `IsTrigger = true`

2. **O que acontece**:
   - Quando Mario entra no trigger, recebe recompensa
   - Mesma lógica de ordem sequencial

## Exemplo de Configuração para a Cena Atual

Se você tem 3 plataformas até o goal:
```
Spawn → Plataforma 1 → Plataforma 2 → Plataforma 3 → Goal
```

Waypoints no Inspector:
- Waypoints[0] = Transform da Plataforma 1 (posição central)
- Waypoints[1] = Transform da Plataforma 2
- Waypoints[2] = Transform da Plataforma 3

## Por que isso funciona?

Sem waypoints:
- Mario vê o goal à frente
- Tenta ir direto (salta para a morte)

Com waypoints:
- Mario precisa passar pelo waypoint 0 para ganhar reward
- Waypoint 0 está na primeira plataforma
- Ele aprende que precisa ir para a plataforma primeiro
- Recompensa acumulativa guia o caminho completo

## Hierarchical RL vs Waypoints

**Waypoints (implementado)**:
- Simples, funciona imediatamente
- Uma política só
- Recompensas intermediárias claras

**Hierarchical RL**:
- Complexo de implementar
- Requer múltiplas políticas (alto nível + baixo nível)
- Mais potente para tarefas muito complexas
- Não necessário para este caso

Para este parkour, waypoints são suficientes e muito mais simples.

"""
Análise de dataset gravado para Imitation Learning.
Mostra estatísticas, distribuições e verifica qualidade dos dados.

Uso:
    python analyze_dataset.py --data ../HybridTrainingData/
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_all_episodes(data_dir):
    """Carrega todos os episódios dos arquivos JSON."""
    episodes = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    print(f"Encontrados {len(json_files)} arquivos JSON")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            file_episodes = data.get('episodes', [])
            
            print(f"  {os.path.basename(json_file)}: {len(file_episodes)} episódios")
            
            for ep in file_episodes:
                ep['source_file'] = os.path.basename(json_file)
                episodes.append(ep)
                
        except Exception as e:
            print(f"  ERRO em {json_file}: {e}")
    
    return episodes, metadata


def analyze_episodes(episodes):
    """Analisa estatísticas dos episódios."""
    
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS GERAIS")
    print("=" * 60)
    
    total_episodes = len(episodes)
    successes = sum(1 for ep in episodes if ep.get('success', False))
    failures = total_episodes - successes
    
    print(f"Total de episódios: {total_episodes}")
    print(f"Sucessos: {successes} ({100*successes/total_episodes:.1f}%)")
    print(f"Falhas: {failures} ({100*failures/total_episodes:.1f}%)")
    
    # Durações
    durations = [ep.get('duration', 0) for ep in episodes]
    step_counts = [ep.get('stepCount', 0) for ep in episodes]
    rewards = [ep.get('totalReward', 0) for ep in episodes]
    
    print(f"\nDURAÇÃO:")
    print(f"  Média: {np.mean(durations):.2f}s")
    print(f"  Mediana: {np.median(durations):.2f}s")
    print(f"  Min: {np.min(durations):.2f}s")
    print(f"  Max: {np.max(durations):.2f}s")
    print(f"  Std: {np.std(durations):.2f}s")
    
    print(f"\nSTEPS POR EPISÓDIO:")
    print(f"  Média: {np.mean(step_counts):.1f}")
    print(f"  Mediana: {np.median(step_counts):.1f}")
    print(f"  Min: {np.min(step_counts)}")
    print(f"  Max: {np.max(step_counts)}")
    
    print(f"\nRECOMPENSA TOTAL:")
    print(f"  Média: {np.mean(rewards):.2f}")
    print(f"  Mediana: {np.median(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")
    print(f"  Std: {np.std(rewards):.2f}")
    
    # Análise por sucesso/falha
    success_eps = [ep for ep in episodes if ep.get('success', False)]
    failure_eps = [ep for ep in episodes if not ep.get('success', False)]
    
    if success_eps and failure_eps:
        print(f"\nCOMPARAÇÃO SUCESSO vs FALHA:")
        print(f"  Duração sucesso: {np.mean([ep.get('duration', 0) for ep in success_eps]):.2f}s")
        print(f"  Duração falha: {np.mean([ep.get('duration', 0) for ep in failure_eps]):.2f}s")
        print(f"  Steps sucesso: {np.mean([ep.get('stepCount', 0) for ep in success_eps]):.1f}")
        print(f"  Steps falha: {np.mean([ep.get('stepCount', 0) for ep in failure_eps]):.1f}")
    
    return {
        'total_episodes': total_episodes,
        'success_rate': successes / total_episodes if total_episodes > 0 else 0,
        'durations': durations,
        'step_counts': step_counts,
        'rewards': rewards,
        'success_eps': success_eps,
        'failure_eps': failure_eps
    }


def analyze_transitions(episodes):
    """Analisa distribuição das transições."""
    
    print("\n" + "=" * 60)
    print("ANÁLISE DAS TRANSIÇÕES")
    print("=" * 60)
    
    all_actions = []
    all_obs = []
    all_rewards = []
    
    for ep in episodes:
        for t in ep.get('transitions', []):
            actions = t.get('actions', [])
            obs = t.get('observations', [])
            reward = t.get('reward', 0)
            
            if len(actions) >= 3 and len(obs) >= 30:
                all_actions.append(actions[:3])
                all_obs.append(obs[:30])
                all_rewards.append(reward)
    
    all_actions = np.array(all_actions)
    all_obs = np.array(all_obs)
    all_rewards = np.array(all_rewards)
    
    print(f"Total de transições: {len(all_actions)}")
    
    # Análise das ações
    print(f"\nDISTRIBUIÇÃO DAS AÇÕES:")
    print(f"  Joystick X: mean={np.mean(all_actions[:, 0]):.3f}, std={np.std(all_actions[:, 0]):.3f}")
    print(f"  Joystick Y: mean={np.mean(all_actions[:, 1]):.3f}, std={np.std(all_actions[:, 1]):.3f}")
    print(f"  Jump rate: {np.mean(all_actions[:, 2] > 0.5):.3f} ({np.sum(all_actions[:, 2] > 0.5)} jumps)")
    
    # Análise das recompensas
    print(f"\nDISTRIBUIÇÃO DAS RECOMPENSAS:")
    print(f"  Mean: {np.mean(all_rewards):.4f}")
    print(f"  Median: {np.median(all_rewards):.4f}")
    print(f"  Min: {np.min(all_rewards):.4f}")
    print(f"  Max: {np.max(all_rewards):.4f}")
    print(f"  Positive %: {100 * np.sum(all_rewards > 0) / len(all_rewards):.1f}%")
    print(f"  Zero %: {100 * np.sum(all_rewards == 0) / len(all_rewards):.1f}%")
    print(f"  Negative %: {100 * np.sum(all_rewards < 0) / len(all_rewards):.1f}%")
    
    # Análise das observações (algumas dimensões importantes)
    print(f"\nOBSERVAÇÕES (30 dimensões):")
    print(f"  Pos X: mean={np.mean(all_obs[:, 0]):.3f}, std={np.std(all_obs[:, 0]):.3f}")
    print(f"  Pos Y: mean={np.mean(all_obs[:, 1]):.3f}, std={np.std(all_obs[:, 1]):.3f}")
    print(f"  Pos Z: mean={np.mean(all_obs[:, 2]):.3f}, std={np.std(all_obs[:, 2]):.3f}")
    print(f"  No ar %: {100 * np.mean(all_obs[:, 7]):.1f}%")
    
    return {
        'actions': all_actions,
        'observations': all_obs,
        'rewards': all_rewards
    }


def plot_analysis(stats, trans_stats, output_dir='analysis_plots'):
    """Gera plots da análise."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribuição de durações
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(stats['durations'], bins=30, edgecolor='black')
    plt.xlabel('Duração (s)')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Durações dos Episódios')
    
    plt.subplot(2, 2, 2)
    plt.hist(stats['step_counts'], bins=30, edgecolor='black')
    plt.xlabel('Steps')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Steps por Episódio')
    
    plt.subplot(2, 2, 3)
    plt.hist(stats['rewards'], bins=30, edgecolor='black')
    plt.xlabel('Recompensa Total')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Recompensas')
    
    plt.subplot(2, 2, 4)
    labels = ['Sucesso', 'Falha']
    sizes = [len(stats['success_eps']), len(stats['failure_eps'])]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Taxa de Sucesso')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_stats.png'))
    print(f"\nPlot salvo: {output_dir}/episode_stats.png")
    
    # 2. Distribuição de ações
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(trans_stats['actions'][:, 0], bins=50, range=(-1, 1), edgecolor='black')
    plt.xlabel('Joystick X')
    plt.ylabel('Frequência')
    plt.title('Distribuição Joystick X')
    
    plt.subplot(1, 3, 2)
    plt.hist(trans_stats['actions'][:, 1], bins=50, range=(-1, 1), edgecolor='black')
    plt.xlabel('Joystick Y')
    plt.ylabel('Frequência')
    plt.title('Distribuição Joystick Y')
    
    plt.subplot(1, 3, 3)
    jumps = trans_stats['actions'][:, 2] > 0.5
    plt.bar(['No Jump', 'Jump'], [np.sum(~jumps), np.sum(jumps)])
    plt.ylabel('Frequência')
    plt.title('Distribuição de Jump')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'))
    print(f"Plot salvo: {output_dir}/action_distribution.png")
    
    # 3. Distribuição de recompensas
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(trans_stats['rewards'], bins=50, edgecolor='black')
    plt.xlabel('Recompensa')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Recompensas (Step)')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    reward_cumsum = np.cumsum(trans_stats['rewards'])
    plt.plot(reward_cumsum / np.arange(1, len(reward_cumsum) + 1))
    plt.xlabel('Step')
    plt.ylabel('Recompensa Média Acumulada')
    plt.title('Evolução da Recompensa Média')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_analysis.png'))
    print(f"Plot salvo: {output_dir}/reward_analysis.png")


def check_data_quality(episodes, trans_stats):
    """Verifica problemas de qualidade nos dados."""
    
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DE QUALIDADE")
    print("=" * 60)
    
    issues = []
    
    # 1. Episódios muito curtos
    short_eps = [ep for ep in episodes if ep.get('stepCount', 0) < 10]
    if short_eps:
        issues.append(f"⚠️  {len(short_eps)} episódios muito curtos (< 10 steps)")
    
    # 2. Episódios muito longos
    long_eps = [ep for ep in episodes if ep.get('stepCount', 0) > 1000]
    if long_eps:
        issues.append(f"⚠️  {len(long_eps)} episódios muito longos (> 1000 steps)")
    
    # 3. Recompensas NaN ou inf
    if np.any(np.isnan(trans_stats['rewards'])):
        issues.append("❌ Recompensas contêm NaN!")
    if np.any(np.isinf(trans_stats['rewards'])):
        issues.append("❌ Recompensas contêm inf!")
    
    # 4. Ações fora do range
    if np.any(np.abs(trans_stats['actions'][:, :2]) > 1.1):
        issues.append("⚠️  Algumas ações de joystick fora do range [-1, 1]")
    
    # 5. Dados faltantes
    missing_obs = np.sum([len(t.get('observations', [])) < 30 
                          for ep in episodes for t in ep.get('transitions', [])])
    if missing_obs > 0:
        issues.append(f"⚠️  {missing_obs} transições com observações incompletas")
    
    # 6. Baixa taxa de sucesso
    success_rate = stats['success_rate']
    if success_rate < 0.3:
        issues.append(f"⚠️  Taxa de sucesso muito baixa ({100*success_rate:.1f}%)")
    elif success_rate > 0.95:
        issues.append(f"ℹ️  Taxa de sucesso muito alta ({100*success_rate:.1f}%) - pode causar overfitting")
    
    if issues:
        print("Problemas encontrados:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Nenhum problema de qualidade detectado!")
    
    # Recomendações
    print(f"\nRECOMENDAÇÕES:")
    if len(episodes) < 50:
        print("  • Considere gravar mais episódios (mínimo recomendado: 50)")
    if np.std(trans_stats['actions'][:, :2]) < 0.1:
        print("  • Pouca variação nas ações - o agente pode não explorar suficientemente")
    print("  • Para Behavior Cloning: dados balanceados funcionam melhor")
    print("  • Para Offline RL: dados diversos (sucesso e falha) são importantes")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset for IL')
    parser.add_argument('--data', type=str, required=True,
                        help='Diretório com dados JSON')
    parser.add_argument('--plots', action='store_true',
                        help='Gerar plots de análise')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                        help='Diretório para salvar plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ANÁLISE DE DATASET - Hybrid Training")
    print("=" * 60)
    
    # Carregar dados
    episodes, metadata = load_all_episodes(args.data)
    
    if not episodes:
        print("\n❌ Nenhum episódio encontrado!")
        return
    
    # Analisar
    global stats
    stats = analyze_episodes(episodes)
    trans_stats = analyze_transitions(episodes)
    
    # Verificar qualidade
    check_data_quality(episodes, trans_stats)
    
    # Gerar plots
    if args.plots:
        plot_analysis(stats, trans_stats, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Análise completa!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Recorded dataset analysis for Imitation Learning.
Shows statistics, distributions, and checks data quality.

Usage:
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
    """Loads all episodes from JSON files."""
    episodes = []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            file_episodes = data.get('episodes', [])
            
            print(f"  {os.path.basename(json_file)}: {len(file_episodes)} episodes")
            
            for ep in file_episodes:
                ep['source_file'] = os.path.basename(json_file)
                episodes.append(ep)
                
        except Exception as e:
            print(f"  ERROR in {json_file}: {e}")
    
    return episodes, metadata


def analyze_episodes(episodes):
    """Analyzes episode statistics."""
    
    print("\n" + "=" * 60)
    print("GENERAL STATISTICS")
    print("=" * 60)
    
    total_episodes = len(episodes)
    successes = sum(1 for ep in episodes if ep.get('success', False))
    failures = total_episodes - successes
    
    print(f"Total episodes: {total_episodes}")
    print(f"Successes: {successes} ({100*successes/total_episodes:.1f}%)")
    print(f"Failures: {failures} ({100*failures/total_episodes:.1f}%)")
    
    # Durations
    durations = [ep.get('duration', 0) for ep in episodes]
    step_counts = [ep.get('stepCount', 0) for ep in episodes]
    rewards = [ep.get('totalReward', 0) for ep in episodes]
    
    print(f"\nDURATION:")
    print(f"  Mean: {np.mean(durations):.2f}s")
    print(f"  Median: {np.median(durations):.2f}s")
    print(f"  Min: {np.min(durations):.2f}s")
    print(f"  Max: {np.max(durations):.2f}s")
    print(f"  Std: {np.std(durations):.2f}s")
    
    print(f"\nSTEPS PER EPISODE:")
    print(f"  Mean: {np.mean(step_counts):.1f}")
    print(f"  Median: {np.median(step_counts):.1f}")
    print(f"  Min: {np.min(step_counts)}")
    print(f"  Max: {np.max(step_counts)}")
    
    print(f"\nTOTAL REWARD:")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Median: {np.median(rewards):.2f}")
    print(f"  Min: {np.min(rewards):.2f}")
    print(f"  Max: {np.max(rewards):.2f}")
    print(f"  Std: {np.std(rewards):.2f}")
    
    # Analysis by success/failure
    success_eps = [ep for ep in episodes if ep.get('success', False)]
    failure_eps = [ep for ep in episodes if not ep.get('success', False)]
    
    if success_eps and failure_eps:
        print(f"\nSUCCESS vs FAILURE COMPARISON:")
        print(f"  Success duration: {np.mean([ep.get('duration', 0) for ep in success_eps]):.2f}s")
        print(f"  Failure duration: {np.mean([ep.get('duration', 0) for ep in failure_eps]):.2f}s")
        print(f"  Success steps: {np.mean([ep.get('stepCount', 0) for ep in success_eps]):.1f}")
        print(f"  Failure steps: {np.mean([ep.get('stepCount', 0) for ep in failure_eps]):.1f}")
    
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
    """Analyzes transition distributions."""
    
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS")
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
    
    print(f"Total transitions: {len(all_actions)}")
    
    # Action analysis
    print(f"\nACTION DISTRIBUTION:")
    print(f"  Joystick X: mean={np.mean(all_actions[:, 0]):.3f}, std={np.std(all_actions[:, 0]):.3f}")
    print(f"  Joystick Y: mean={np.mean(all_actions[:, 1]):.3f}, std={np.std(all_actions[:, 1]):.3f}")
    print(f"  Jump rate: {np.mean(all_actions[:, 2] > 0.5):.3f} ({np.sum(all_actions[:, 2] > 0.5)} jumps)")
    
    # Reward analysis
    print(f"\nREWARD DISTRIBUTION:")
    print(f"  Mean: {np.mean(all_rewards):.4f}")
    print(f"  Median: {np.median(all_rewards):.4f}")
    print(f"  Min: {np.min(all_rewards):.4f}")
    print(f"  Max: {np.max(all_rewards):.4f}")
    print(f"  Positive %: {100 * np.sum(all_rewards > 0) / len(all_rewards):.1f}%")
    print(f"  Zero %: {100 * np.sum(all_rewards == 0) / len(all_rewards):.1f}%")
    print(f"  Negative %: {100 * np.sum(all_rewards < 0) / len(all_rewards):.1f}%")
    
    # Observation analysis (key dimensions)
    print(f"\nOBSERVATIONS (30 dimensions):")
    print(f"  Pos X: mean={np.mean(all_obs[:, 0]):.3f}, std={np.std(all_obs[:, 0]):.3f}")
    print(f"  Pos Y: mean={np.mean(all_obs[:, 1]):.3f}, std={np.std(all_obs[:, 1]):.3f}")
    print(f"  Pos Z: mean={np.mean(all_obs[:, 2]):.3f}, std={np.std(all_obs[:, 2]):.3f}")
    print(f"  Airborne %: {100 * np.mean(all_obs[:, 7]):.1f}%")
    
    return {
        'actions': all_actions,
        'observations': all_obs,
        'rewards': all_rewards
    }


def plot_analysis(stats, trans_stats, output_dir='analysis_plots'):
    """Generates analysis plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Duration distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(stats['durations'], bins=30, edgecolor='black')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Episode Duration Distribution')
    
    plt.subplot(2, 2, 2)
    plt.hist(stats['step_counts'], bins=30, edgecolor='black')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.title('Steps per Episode Distribution')
    
    plt.subplot(2, 2, 3)
    plt.hist(stats['rewards'], bins=30, edgecolor='black')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    
    plt.subplot(2, 2, 4)
    labels = ['Success', 'Failure']
    sizes = [len(stats['success_eps']), len(stats['failure_eps'])]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Success Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_stats.png'))
    print(f"\nPlot saved: {output_dir}/episode_stats.png")
    
    # 2. Action distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(trans_stats['actions'][:, 0], bins=50, range=(-1, 1), edgecolor='black')
    plt.xlabel('Joystick X')
    plt.ylabel('Frequency')
    plt.title('Joystick X Distribution')
    
    plt.subplot(1, 3, 2)
    plt.hist(trans_stats['actions'][:, 1], bins=50, range=(-1, 1), edgecolor='black')
    plt.xlabel('Joystick Y')
    plt.ylabel('Frequency')
    plt.title('Joystick Y Distribution')
    
    plt.subplot(1, 3, 3)
    jumps = trans_stats['actions'][:, 2] > 0.5
    plt.bar(['No Jump', 'Jump'], [np.sum(~jumps), np.sum(jumps)])
    plt.ylabel('Frequency')
    plt.title('Jump Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_distribution.png'))
    print(f"Plot saved: {output_dir}/action_distribution.png")
    
    # 3. Reward distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(trans_stats['rewards'], bins=50, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution (per Step)')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    reward_cumsum = np.cumsum(trans_stats['rewards'])
    plt.plot(reward_cumsum / np.arange(1, len(reward_cumsum) + 1))
    plt.xlabel('Step')
    plt.ylabel('Cumulative Mean Reward')
    plt.title('Mean Reward Evolution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_analysis.png'))
    print(f"Plot saved: {output_dir}/reward_analysis.png")


def check_data_quality(episodes, trans_stats):
    """Checks for data quality issues."""
    
    print("\n" + "=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)
    
    issues = []
    
    # 1. Very short episodes
    short_eps = [ep for ep in episodes if ep.get('stepCount', 0) < 10]
    if short_eps:
        issues.append(f"WARNING: {len(short_eps)} very short episodes (< 10 steps)")
    
    # 2. Very long episodes
    long_eps = [ep for ep in episodes if ep.get('stepCount', 0) > 1000]
    if long_eps:
        issues.append(f"WARNING: {len(long_eps)} very long episodes (> 1000 steps)")
    
    # 3. NaN or inf rewards
    if np.any(np.isnan(trans_stats['rewards'])):
        issues.append("ERROR: Rewards contain NaN!")
    if np.any(np.isinf(trans_stats['rewards'])):
        issues.append("ERROR: Rewards contain inf!")
    
    # 4. Out-of-range actions
    if np.any(np.abs(trans_stats['actions'][:, :2]) > 1.1):
        issues.append("WARNING: Some joystick actions are out of range [-1, 1]")
    
    # 5. Missing data
    missing_obs = np.sum([len(t.get('observations', [])) < 30 
                          for ep in episodes for t in ep.get('transitions', [])])
    if missing_obs > 0:
        issues.append(f"WARNING: {missing_obs} transitions with incomplete observations")
    
    # 6. Low success rate
    success_rate = stats['success_rate']
    if success_rate < 0.3:
        issues.append(f"WARNING: Very low success rate ({100*success_rate:.1f}%)")
    elif success_rate > 0.95:
        issues.append(f"INFO: Very high success rate ({100*success_rate:.1f}%) - may cause overfitting")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("No quality issues detected!")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if len(episodes) < 50:
        print("  - Consider recording more episodes (minimum recommended: 50)")
    if np.std(trans_stats['actions'][:, :2]) < 0.1:
        print("  - Low action variance - agent may not be exploring sufficiently")
    print("  - For Behavior Cloning: balanced data works best")
    print("  - For Offline RL: diverse data (both success and failure) is important")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset for IL')
    parser.add_argument('--data', type=str, required=True,
                        help='Directory with JSON data')
    parser.add_argument('--plots', action='store_true',
                        help='Generate analysis plots')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATASET ANALYSIS - Hybrid Training")
    print("=" * 60)
    
    # Load data
    episodes, metadata = load_all_episodes(args.data)
    
    if not episodes:
        print("\nERROR: No episodes found!")
        return
    
    # Analyze
    global stats
    stats = analyze_episodes(episodes)
    trans_stats = analyze_transitions(episodes)
    
    # Check quality
    check_data_quality(episodes, trans_stats)
    
    # Generate plots
    if args.plots:
        plot_analysis(stats, trans_stats, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

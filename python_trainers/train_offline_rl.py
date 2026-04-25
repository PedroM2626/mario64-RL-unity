"""
Offline RL (CQL/IQL) para Mario Parkour
Treina agente apenas de dados gravados, sem interação com ambiente.

Uso:
    python train_offline_rl.py --data ../HybridTrainingData/ --algo CQL --output models/cql_model.pth
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random

# Configuração
OBSERVATION_DIM = 30
ACTION_DIM = 3
GAMMA = 0.99
TAU = 0.005

class OfflineDataset(Dataset):
    """Dataset para Offline RL."""
    
    def __init__(self, data_dir, min_episode_length=10):
        self.transitions = []
        self.load_data(data_dir, min_episode_length)
        print(f"Dataset carregado: {len(self.transitions)} transições")
        
    def load_data(self, data_dir, min_length):
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for episode in data.get('episodes', []):
                    if episode.get('stepCount', 0) < min_length:
                        continue
                    
                    transitions = episode.get('transitions', [])
                    for i, transition in enumerate(transitions):
                        obs = np.array(transition.get('observations', []), dtype=np.float32)
                        actions = np.array(transition.get('actions', []), dtype=np.float32)
                        reward = float(transition.get('reward', 0.0))
                        done = bool(transition.get('done', False))
                        
                        # Next observation
                        if i + 1 < len(transitions):
                            next_obs = np.array(transitions[i+1].get('observations', []), dtype=np.float32)
                        else:
                            next_obs = obs  # Terminal state
                        
                        if len(obs) >= OBSERVATION_DIM and len(actions) >= ACTION_DIM:
                            self.transitions.append({
                                'obs': obs[:OBSERVATION_DIM],
                                'actions': actions[:ACTION_DIM],
                                'reward': reward,
                                'next_obs': next_obs[:OBSERVATION_DIM],
                                'done': done
                            })
                            
            except Exception as e:
                print(f"Erro ao carregar {json_file}: {e}")
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            'obs': torch.FloatTensor(t['obs']),
            'actions': torch.FloatTensor(t['actions']),
            'reward': torch.FloatTensor([t['reward']]),
            'next_obs': torch.FloatTensor(t['next_obs']),
            'done': torch.FloatTensor([t['done']])
        }


class QNetwork(nn.Module):
    """Q-Network para CQL/IQL."""
    
    def __init__(self, obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, 
                 hidden_size=256, num_layers=2):
        super().__init__()
        
        layers = []
        in_size = obs_dim + action_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.q_value = nn.Linear(hidden_size, 1)
        
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        features = self.network(x)
        return self.q_value(features)


class PolicyNetwork(nn.Module):
    """Política para IQL/actor."""
    
    def __init__(self, obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM,
                 hidden_size=256, num_layers=2):
        super().__init__()
        
        layers = []
        in_size = obs_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)
        
    def forward(self, obs):
        features = self.network(obs)
        mean = torch.tanh(self.mean(features))
        log_std = torch.clamp(self.log_std(features), -20, 2)
        return mean, log_std
    
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob


class CQLAgent:
    """Conservative Q-Learning Agent."""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Q-networks (duplas)
        self.q1 = QNetwork(args.hidden_size, args.num_layers).to(device)
        self.q2 = QNetwork(args.hidden_size, args.num_layers).to(device)
        self.q1_target = QNetwork(args.hidden_size, args.num_layers).to(device)
        self.q2_target = QNetwork(args.hidden_size, args.num_layers).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=args.lr
        )
        
        # Política
        self.policy = PolicyNetwork(args.hidden_size, args.num_layers).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        
    def train_step(self, batch):
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Q-values atuais
        q1_value = self.q1(obs, actions)
        q2_value = self.q2(obs, actions)
        
        # Q-values alvo
        with torch.no_grad():
            next_actions, _ = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = reward + (1 - done) * GAMMA * q_next
        
        # Loss Q padrão
        q1_loss = F.mse_loss(q1_value, q_target)
        q2_loss = F.mse_loss(q2_value, q_target)
        
        # CQL Loss (conservador)
        # Amostrar ações aleatórias
        random_actions = torch.FloatTensor(
            q1_value.shape[0], ACTION_DIM
        ).uniform_(-1, 1).to(self.device)
        
        q1_random = self.q1(obs, random_actions)
        q2_random = self.q2(obs, random_actions)
        
        # CQL penalty
        cql1_loss = torch.logsumexp(q1_random, dim=0).mean() - q1_value.mean()
        cql2_loss = torch.logsumexp(q2_random, dim=0).mean() - q2_value.mean()
        
        # Loss total Q
        q_loss = q1_loss + q2_loss + self.args.cql_alpha * (cql1_loss + cql2_loss)
        
        # Atualizar Q
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Atualizar política
        new_actions, log_prob = self.policy.sample(obs)
        q1_new = self.q1(obs, new_actions)
        q2_new = self.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.args.alpha * log_prob - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'q_value': q1_value.mean().item()
        }
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    def save(self, path):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'policy': self.policy.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }, path)


class IQLAgent:
    """Implicit Q-Learning Agent."""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # Value function
        self.v = self._build_v_network().to(device)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=args.lr)
        
        # Q-networks
        self.q1 = QNetwork(args.hidden_size, args.num_layers).to(device)
        self.q2 = QNetwork(args.hidden_size, args.num_layers).to(device)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=args.lr
        )
        
        # Política
        self.policy = PolicyNetwork(args.hidden_size, args.num_layers).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        
    def _build_v_network(self):
        layers = []
        in_size = OBSERVATION_DIM
        for _ in range(self.args.num_layers):
            layers.append(nn.Linear(in_size, self.args.hidden_size))
            layers.append(nn.ReLU())
            in_size = self.args.hidden_size
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)
    
    def train_step(self, batch):
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)
        
        # Atualizar V
        with torch.no_grad():
            q1_value = self.q1(obs, actions)
            q2_value = self.q2(obs, actions)
            q_value = torch.min(q1_value, q2_value)
        
        v_value = self.v(obs)
        v_loss = ((v_value - q_value) ** 2).mean()
        
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        
        # Atualizar Q
        with torch.no_grad():
            v_next = self.v(next_obs)
            q_target = reward + (1 - done) * GAMMA * v_next
        
        q1_loss = F.mse_loss(self.q1(obs, actions), q_target)
        q2_loss = F.mse_loss(self.q2(obs, actions), q_target)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Atualizar política (AWR)
        with torch.no_grad():
            adv = q_value - v_value
            exp_adv = torch.exp(adv / self.args.temperature)
            exp_adv = torch.clamp(exp_adv, max=100.0)
        
        action_mean, action_log_std = self.policy(obs)
        action_std = torch.exp(action_log_std)
        
        log_prob = -0.5 * ((actions - action_mean) / action_std) ** 2
        log_prob = log_prob.sum(-1, keepdim=True)
        
        policy_loss = -(exp_adv * log_prob).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'v_value': v_value.mean().item()
        }
    
    def save(self, path):
        torch.save({
            'v': self.v.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'policy': self.policy.state_dict()
        }, path)


def train_offline_rl(args):
    """Treina Offline RL."""
    
    print("=" * 60)
    print(f"Offline RL - {args.algo} - Mario Parkour")
    print("=" * 60)
    
    # Dataset
    dataset = OfflineDataset(args.data, min_episode_length=args.min_episode_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Device: {device}")
    print(f"Transições: {len(dataset)}")
    
    # Agente
    if args.algo == 'CQL':
        agent = CQLAgent(args, device)
    elif args.algo == 'IQL':
        agent = IQLAgent(args, device)
    else:
        raise ValueError(f"Algoritmo desconhecido: {args.algo}")
    
    # Treinamento
    losses_history = []
    
    for epoch in range(args.epochs):
        epoch_losses = []
        
        for batch in dataloader:
            metrics = agent.train_step(batch)
            epoch_losses.append(metrics)
        
        # Médias
        avg_metrics = {}
        for key in epoch_losses[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_losses])
        
        losses_history.append(avg_metrics)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - " + 
                  " ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
        
        # Salvar checkpoint
        if (epoch + 1) % 50 == 0:
            checkpoint_path = args.output.replace('.pth', f'_epoch{epoch+1}.pth')
            agent.save(checkpoint_path)
    
    # Salvar modelo final
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    agent.save(args.output)
    
    print("\n" + "=" * 60)
    print(f"Treinamento completo!")
    print(f"Modelo salvo em: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Final metrics: {losses_history[-1]}")


def main():
    parser = argparse.ArgumentParser(description='Train Offline RL for Mario')
    parser.add_argument('--data', type=str, required=True,
                        help='Diretório com dados JSON')
    parser.add_argument('--algo', type=str, default='CQL', choices=['CQL', 'IQL'],
                        help='Algoritmo: CQL ou IQL')
    parser.add_argument('--output', type=str, default='models/offline_rl_model.pth',
                        help='Caminho para salvar modelo')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Número de epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Tamanho da camada oculta')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Número de camadas')
    parser.add_argument('--min-episode-length', type=int, default=10,
                        help='Comprimento mínimo do episódio')
    
    # CQL specific
    parser.add_argument('--cql-alpha', type=float, default=1.0,
                        help='Peso do CQL loss')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Temperatura da política')
    
    # IQL specific
    parser.add_argument('--temperature', type=float, default=3.0,
                        help='Temperatura do AWR')
    
    parser.add_argument('--cuda', action='store_true',
                        help='Usar CUDA se disponível')
    
    args = parser.parse_args()
    
    train_offline_rl(args)


if __name__ == '__main__':
    main()

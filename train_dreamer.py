"""
DreamerV3 Training Script for Mario Parkour Environment
========================================================
This script implements a simplified Dreamer-like world model training approach
for the Mario Parkour environment using PyTorch and ML-Agents.

Features:
- World Model (RSSM - Recurrent State-Space Model)
- Actor-Critic with learned dynamics
- MLOps tracking with MLflow
- TensorBoard logging
- ONNX model export
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import mlflow
import time as time_module
from datetime import datetime
from collections import deque
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# DREAMER WORLD MODEL COMPONENTS
# ============================================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM) - Core of Dreamer
    Predicts: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
              z_t ~ q(z_t | h_t, x_t)  (posterior)
              z_t ~ p(z_t | h_t)       (prior)
    """
    def __init__(self, obs_dim=47, action_dim=5, hidden_dim=256, state_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.recurrent = nn.GRUCell(state_dim + action_dim, hidden_dim)
        
        # Prior: p(z_t | h_t)
        self.prior_mean = nn.Linear(hidden_dim, state_dim)
        self.prior_std = nn.Linear(hidden_dim, state_dim)
        
        # Posterior encoder: q(z_t | h_t, x_t)
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # mean and log_std
        )
        
    def init_state(self, batch_size, device):
        """Initialize hidden and state vectors"""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.state_dim, device=device)
        return h, z
    
    def observe(self, h_prev, z_prev, action, obs):
        """Update with real observation (posterior)"""
        # Recurrent update
        h = self.recurrent(torch.cat([z_prev, action], dim=-1), h_prev)
        
        # Posterior from observation
        posterior_input = torch.cat([h, obs], dim=-1)
        posterior_out = self.posterior(posterior_input)
        mean, log_std = posterior_out.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1
        z = mean + std * torch.randn_like(mean)
        
        return h, z, mean, std
    
    def imagine(self, h_prev, z_prev, action):
        """Predict without observation (prior)"""
        # Recurrent update
        h = self.recurrent(torch.cat([z_prev, action], dim=-1), h_prev)
        
        # Prior prediction
        mean = self.prior_mean(h)
        std = F.softplus(self.prior_std(h)) + 0.1
        z = mean + std * torch.randn_like(mean)
        
        return h, z, mean, std


class ObservationDecoder(nn.Module):
    """Decodes latent state back to observation"""
    def __init__(self, state_dim=32, hidden_dim=256, obs_dim=47):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, h, z):
        return self.decoder(torch.cat([h, z], dim=-1))


class RewardPredictor(nn.Module):
    """Predicts reward from latent state"""
    def __init__(self, state_dim=32, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h, z):
        return self.predictor(torch.cat([h, z], dim=-1))


class ContinuePredictor(nn.Module):
    """Predicts episode termination from latent state"""
    def __init__(self, state_dim=32, hidden_dim=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h, z):
        return self.predictor(torch.cat([h, z], dim=-1))


class Actor(nn.Module):
    """
    Actor network for Dreamer
    Outputs action distribution given imagined state
    """
    def __init__(self, state_dim=32, hidden_dim=256, action_dim=5):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Continuous actions (joystick x, y)
        self.mean_cont = nn.Linear(hidden_dim, 2)
        self.std_cont = nn.Linear(hidden_dim, 2)
        
        # Discrete actions (jump, kick, stomp)
        self.logits_disc = nn.Linear(hidden_dim, 3)  # 3 binary actions
        
    def forward(self, h, z):
        features = self.actor(torch.cat([h, z], dim=-1))
        
        # Continuous actions
        mean_cont = torch.tanh(self.mean_cont(features))
        std_cont = F.softplus(self.std_cont(features)) + 0.01
        
        # Discrete action logits
        logits_disc = self.logits_disc(features)
        
        return mean_cont, std_cont, logits_disc
    
    def sample(self, h, z, deterministic=False):
        mean_cont, std_cont, logits_disc = self.forward(h, z)
        
        if deterministic:
            action_cont = mean_cont
            action_disc = (logits_disc > 0).float()
        else:
            # Sample continuous (non-differentiable, for env interaction only)
            dist_cont = Normal(mean_cont, std_cont)
            action_cont = dist_cont.sample()
            action_cont = torch.clamp(action_cont, -1, 1)
            
            # Sample discrete (non-differentiable, for env interaction only)
            probs_disc = torch.sigmoid(logits_disc)
            action_disc = (torch.rand_like(probs_disc) < probs_disc).float()
        
        return action_cont, action_disc

    def sample_differentiable(self, h, z):
        """Differentiable sampling for imagination / policy gradients.

        - Continuous: reparameterized Normal.rsample() so grads flow to mean/std.
        - Discrete: straight-through Bernoulli (probs + (sample - probs).detach())
          so grads flow to logits via the probs path while keeping 0/1 outputs.
        """
        mean_cont, std_cont, logits_disc = self.forward(h, z)

        dist_cont = Normal(mean_cont, std_cont)
        action_cont = dist_cont.rsample()
        action_cont = torch.clamp(action_cont, -1, 1)

        probs_disc = torch.sigmoid(logits_disc)
        sample_disc = (torch.rand_like(probs_disc) < probs_disc).float()
        action_disc = probs_disc + (sample_disc - probs_disc).detach()

        return action_cont, action_disc


class Critic(nn.Module):
    """
    Critic network for Dreamer
    Predicts value function given state
    """
    def __init__(self, state_dim=32, hidden_dim=256):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.target_critic = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize target with same weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def forward(self, h, z):
        return self.critic(torch.cat([h, z], dim=-1))
    
    def target(self, h, z):
        return self.target_critic(torch.cat([h, z], dim=-1))
    
    def update_target(self, tau=0.01):
        """Soft update of target network"""
        for param, target_param in zip(self.critic.parameters(), 
                                        self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DreamerAgent:
    """
    Complete DreamerV3 Agent
    Combines world model, actor, and critic
    """
    def __init__(self, obs_dim=47, action_dim=5, device="cuda", 
                 learning_rate=1e-4, gamma=0.99, lambda_=0.95):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # World model components
        self.rssm = RSSM(obs_dim, action_dim).to(device)
        self.obs_decoder = ObservationDecoder().to(device)
        self.reward_pred = RewardPredictor().to(device)
        self.continue_pred = ContinuePredictor().to(device)
        
        # Policy components
        self.actor = Actor(action_dim=action_dim).to(device)
        self.critic = Critic().to(device)
        
        # Optimizers
        self.world_optimizer = optim.Adam(
            list(self.rssm.parameters()) +
            list(self.obs_decoder.parameters()) +
            list(self.reward_pred.parameters()) +
            list(self.continue_pred.parameters()),
            lr=learning_rate
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # State
        self.h, self.z = None, None
        self.prev_action = None
        
    def init_state(self, batch_size=1):
        """Initialize agent state at episode start"""
        self.h, self.z = self.rssm.init_state(batch_size, self.device)
        self.prev_action = torch.zeros(batch_size, self.action_dim, device=self.device)
        
    def act(self, obs, deterministic=False):
        """Select action given observation"""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Update state with observation
        self.h, self.z, _, _ = self.rssm.observe(self.h, self.z, self.prev_action, obs_t)
        
        # Sample action
        action_cont, action_disc = self.actor.sample(self.h, self.z, deterministic)
        
        # Combine actions
        action = torch.cat([action_cont, action_disc], dim=-1)
        self.prev_action = action
        
        # Convert to numpy for Unity
        action_np = action.cpu().numpy()[0]
        return self._convert_to_unity_action(action_np), action_np
    
    def _convert_to_unity_action(self, action):
        """Convert Dreamer action to Unity ActionTuple"""
        # action: [joy_x, joy_y, jump, kick, stomp]
        cont = np.array([[action[0], action[1]]], dtype=np.float32)
        disc = np.array([[1 if action[2] > 0.5 else 0,
                          1 if action[3] > 0.5 else 0,
                          1 if action[4] > 0.5 else 0]], dtype=np.int32)
        return ActionTuple(continuous=cont, discrete=disc)
    
    def train_world_model(self, observations, actions, rewards, dones):
        """
        Train world model on collected experience
        observations: [B, T, obs_dim]
        actions: [B, T, action_dim]
        rewards: [B, T]
        dones: [B, T]
        """
        batch_size, seq_len = observations.shape[:2]
        
        # Initialize states
        h, z = self.rssm.init_state(batch_size, self.device)
        
        # Collect predictions
        obs_preds = []
        reward_preds = []
        kl_divs = []
        
        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t]
            
            # Posterior update
            h_new, z_new, post_mean, post_std = self.rssm.observe(h, z, action_t, obs_t)
            
            # Prior prediction
            _, _, prior_mean, prior_std = self.rssm.imagine(h, z, action_t)
            
            # Decode
            obs_pred = self.obs_decoder(h_new, z_new)
            reward_pred = self.reward_pred(h_new, z_new)
            
            # KL divergence
            kl = self._kl_divergence(post_mean, post_std, prior_mean, prior_std)
            
            obs_preds.append(obs_pred)
            reward_preds.append(reward_pred)
            kl_divs.append(kl)
            
            h, z = h_new, z_new
        
        # Compute losses
        obs_preds = torch.stack(obs_preds, dim=1)
        reward_preds = torch.stack(reward_preds, dim=1).squeeze(-1)
        kl_divs = torch.stack(kl_divs, dim=1)
        
        obs_loss = F.mse_loss(obs_preds, observations)
        reward_loss = F.mse_loss(reward_preds, rewards)
        kl_loss = kl_divs.mean()
        
        total_loss = obs_loss + reward_loss + 0.1 * kl_loss
        
        self.world_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.rssm.parameters()) +
            list(self.obs_decoder.parameters()) +
            list(self.reward_pred.parameters()),
            100.0
        )
        self.world_optimizer.step()
        
        return {
            'world_model/obs_loss': obs_loss.item(),
            'world_model/reward_loss': reward_loss.item(),
            'world_model/kl_loss': kl_loss.item(),
            'world_model/total_loss': total_loss.item()
        }
    
    def _kl_divergence(self, mean1, std1, mean2, std2):
        """Compute KL divergence between two Gaussians"""
        var1 = std1 ** 2
        var2 = std2 ** 2
        kl = torch.log(std2 / std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
        return kl.sum(dim=-1)
    
    def train_actor_critic(self, batch_size=32, horizon=15):
        """
        Train actor and critic using imagined trajectories.

        Single differentiable rollout:
        - actions via Actor.sample_differentiable (rsample + straight-through)
        - next states via RSSM.imagine (reparameterized, grads flow)
        - rewards/values with grad enabled
        Critic regresses detached lambda-returns; actor maximizes mean
        predicted reward (grads flow through dynamics to the policy).
        World-model weights are frozen in the optimizer sense (only
        actor/critic optimizers step), but grads flow through them.
        """
        # Initialize state if not already done
        if self.h is None or self.z is None:
            self.init_state(batch_size)

        # Start from current state, detached to avoid backprop into history.
        # self.h/z are [1, D]; expand to [batch, D].
        if self.h.shape[0] == 1 and batch_size > 1:
            h = self.h.detach().repeat(batch_size, 1)
            z = self.z.detach().repeat(batch_size, 1)
        elif self.h.shape[0] == batch_size:
            h = self.h.detach()
            z = self.z.detach()
        else:
            # Fallback: average/trim to requested batch
            h = self.h.detach()[:1].repeat(batch_size, 1)
            z = self.z.detach()[:1].repeat(batch_size, 1)

        rewards_pred = []
        values = []

        for _ in range(horizon):
            # Differentiable action sample (grads flow to actor)
            action_cont, action_disc = self.actor.sample_differentiable(h, z)
            action = torch.cat([action_cont, action_disc], dim=-1)

            # Imagine next state (differentiable dynamics)
            h, z, _, _ = self.rssm.imagine(h, z, action)

            # Predict reward and value WITH grad (needed for both losses)
            reward = self.reward_pred(h, z)
            value = self.critic(h, z)

            rewards_pred.append(reward)
            values.append(value)

        # Compute detached lambda-returns as critic targets
        with torch.no_grad():
            rewards_det = [r.detach() for r in rewards_pred]
            values_det = [v.detach() for v in values]
            returns_detached = self._compute_returns(rewards_det, values_det)

        # ========== Treinar Critic ==========
        values_stacked = torch.stack(values, dim=1).squeeze(-1)  # [B, H] with grad
        critic_loss = F.mse_loss(values_stacked, returns_detached)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # ========== Treinar Actor ==========
        # Maximize mean predicted reward (grads flow: reward_pred -> rssm -> actor).
        rewards_stacked = torch.stack(rewards_pred, dim=1).squeeze(-1)  # [B, H] with grad
        actor_loss = -rewards_stacked.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        # Update target critic
        self.critic.update_target()

        return {
            'policy/actor_loss': actor_loss.item(),
            'policy/critic_loss': critic_loss.item(),
            'policy/returns_mean': returns_detached.mean().item()
        }
    
    def _compute_returns(self, rewards, values, horizon=15):
        """Compute TD-lambda returns"""
        returns = []
        gae = 0
        
        # Usar o tamanho real dos tensores, não o parâmetro horizon
        actual_horizon = len(rewards)
        
        for t in reversed(range(actual_horizon)):
            if t == actual_horizon - 1:
                next_value = 0
            else:
                next_value = values[t + 1].squeeze(-1)
            
            delta = rewards[t].squeeze(-1) + self.gamma * next_value - values[t].squeeze(-1)
            gae = delta + self.gamma * self.lambda_ * gae
            returns.insert(0, gae + values[t].squeeze(-1))
        
        return torch.stack(returns, dim=1)
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'rssm': self.rssm.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'reward_pred': self.reward_pred.state_dict(),
            'continue_pred': self.continue_pred.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        self.reward_pred.load_state_dict(checkpoint['reward_pred'])
        self.continue_pred.load_state_dict(checkpoint['continue_pred'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class MultiAgentDreamer:
    """
    Wrapper para treinar múltiplos agentes em paralelo com um modelo compartilhado.
    Cada agente tem seu próprio estado (h, z, prev_action) mas compartilha os pesos do modelo.
    """
    def __init__(self, base_agent, num_agents, device):
        self.agent = base_agent
        self.num_agents = num_agents
        self.device = device
        
        # Estados separados para cada agente
        self.states = {}
        
    def init_agent_state(self, agent_name):
        """Inicializa o estado de um agente específico"""
        h, z = self.agent.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.agent.action_dim, device=self.device)
        self.states[agent_name] = {
            'h': h,
            'z': z,
            'prev_action': prev_action
        }
        
    def act(self, agent_name, obs, deterministic=False):
        """Seleciona ação para um agente específico"""
        if agent_name not in self.states:
            self.init_agent_state(agent_name)
            
        state = self.states[agent_name]
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Update state with observation
        state['h'], state['z'], _, _ = self.agent.rssm.observe(
            state['h'], state['z'], state['prev_action'], obs_t
        )
        
        # Sample action
        action_cont, action_disc = self.agent.actor.sample(state['h'], state['z'], deterministic)
        
        # Combine actions
        action = torch.cat([action_cont, action_disc], dim=-1)
        state['prev_action'] = action
        
        # Convert to numpy for Unity
        action_np = action.cpu().numpy()[0]
        return self._convert_to_unity_action(action_np), action_np
    
    def _convert_to_unity_action(self, action):
        """Convert Dreamer action to Unity ActionTuple"""
        cont = np.array([[action[0], action[1]]], dtype=np.float32)
        disc = np.array([[1 if action[2] > 0.5 else 0,
                          1 if action[3] > 0.5 else 0,
                          1 if action[4] > 0.5 else 0]], dtype=np.int32)
        return ActionTuple(continuous=cont, discrete=disc)
    
    def get_prev_action_numpy(self, agent_name):
        """Retorna a ação anterior do agente como numpy array"""
        if agent_name in self.states:
            return self.states[agent_name]['prev_action'].cpu().numpy()[0]
        return np.zeros(5)
        
    def train_world_model(self, observations, actions, rewards, dones):
        """Delega para o agente base"""
        return self.agent.train_world_model(observations, actions, rewards, dones)
        
    def train_actor_critic(self, batch_size, horizon):
        """Delega para o agente base"""
        return self.agent.train_actor_critic(batch_size, horizon)
        
    def save(self, path):
        """Delega para o agente base"""
        self.agent.save(path)
        
    def load(self, path):
        """Delega para o agente base"""
        self.agent.load(path)


class ExperienceBuffer:
    """Buffer for collecting experience sequences"""
    def __init__(self, capacity=200000, seq_len=50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))
        
    def sample(self, batch_size):
        """Sample sequences from buffer (sampling with replacement)."""
        if len(self.buffer) < self.seq_len + 1:
            return None
        
        sequences = []
        for _ in range(batch_size):
            start_idx = np.random.randint(0, len(self.buffer) - self.seq_len)
            seq = [self.buffer[start_idx + i] for i in range(self.seq_len)]
            sequences.append(seq)
        
        # Unpack sequences
        obs_seq = torch.tensor([[s[0] for s in seq] for seq in sequences], dtype=torch.float32)
        action_seq = torch.tensor([[s[1] for s in seq] for seq in sequences], dtype=torch.float32)
        reward_seq = torch.tensor([[s[2] for s in seq] for seq in sequences], dtype=torch.float32)
        done_seq = torch.tensor([[s[3] for s in seq] for seq in sequences], dtype=torch.float32)
        
        return obs_seq, action_seq, reward_seq, done_seq
    
    def __len__(self):
        return len(self.buffer)


def export_onnx(agent, path):
    """Export Dreamer recurrent policy to ONNX for EXTERNAL (Python) inference.

    LIMITATION: NOT Unity Barracuda / ML-Agents compatible. The Dreamer policy
    is recurrent (needs h, z, prev_action); Unity ML-Agents expects a stateless
    obs->action model. Old export took a single 288-dim (h+z) tensor which
    Unity never provides. This export takes explicit (obs 47-dim, h 256-dim,
    z 32-dim, prev_action 5-dim) and returns (action 5-dim, next_h, next_z)
    for Python-side rollout. For in-Unity inference, retrain the behavior
    with PPO/SAC.
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    class RecurrentPolicyWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.rssm = base.rssm
            self.actor = base.actor

        def forward(self, obs, h_prev, z_prev, prev_action):
            h, z, _, _ = self.rssm.observe(h_prev, z_prev, prev_action, obs)
            mean_cont, _, logits_disc = self.actor(h, z)
            # Deterministic action: mean + thresholded discretes
            action_disc = (logits_disc > 0).float()
            action = torch.cat([mean_cont, action_disc], dim=-1)
            return action, h, z

    base_agent = agent.agent if hasattr(agent, 'agent') else agent

    wrapper = RecurrentPolicyWrapper(base_agent).to(base_agent.device)
    wrapper.eval()

    dummy_obs = torch.randn(1, 47, device=base_agent.device)
    dummy_h = torch.zeros(1, 256, device=base_agent.device)
    dummy_z = torch.zeros(1, 32, device=base_agent.device)
    dummy_a = torch.zeros(1, 5, device=base_agent.device)

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_h, dummy_z, dummy_a),
        path,
        opset_version=11,
        input_names=["obs", "h_prev", "z_prev", "prev_action"],
        output_names=["action", "next_h", "next_z"],
    )
    print(f"[OK] Dreamer ONNX (external inference only, not Barracuda) -> {path}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 Training for Mario Parkour")
    parser.add_argument("--env", type=str, default=None, help="Path to Unity executable")
    parser.add_argument("--run-id", type=str, default=f"DreamerParkour_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="Overwrite previous models")
    parser.add_argument("--tb-logdir", type=str, default="./tensorboard_logs", help="TensorBoard log directory")
    parser.add_argument("--time-scale", type=float, default=3.0, help="Unity time scale")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training (256 CPU-safe; use 2048 only on large GPU)")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length for world model")
    parser.add_argument("--horizon", type=int, default=15, help="Imagination horizon")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of parallel agents in Unity")
    parser.add_argument("--capacity", type=int, default=200000, help="Experience buffer capacity (must exceed seq_len*8; 200k default)")
    parser.add_argument("--no-graphics", action="store_true", help="Run Unity without graphics (headless mode)")
    parser.add_argument("--timeout", type=int, default=300, help="Unity connection timeout in seconds (default 300)")
    args = parser.parse_args()

    print(f"[*] Starting DreamerV3 Training")
    print(f"[*] Device: {args.device}")
    print(f"[*] Run ID: {args.run_id}")

    # MLOps: Initialize Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Mario_DreamerParkour")

    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, args.run_id))
    print(f"[*] TensorBoard logs at: {os.path.join(args.tb_logdir, args.run_id)}")

    with mlflow.start_run(run_name=args.run_id) as run:
        print(f"[*] Started MLflow run: {run.info.run_id}")
        mlflow.log_param("Framework", "DreamerV3 + Unity ML-Agents")
        mlflow.log_param("Scene", "DreamerParkour")
        mlflow.log_param("time_scale", args.time_scale)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("seq_len", args.seq_len)
        mlflow.log_param("horizon", args.horizon)
        mlflow.log_param("device", args.device)
        mlflow.log_param("num_agents", args.num_agents)
        mlflow.log_param("buffer_capacity", args.capacity)
        
        # Initialize Dreamer Agent (shared model for multi-agent)
        base_agent = DreamerAgent(obs_dim=47, action_dim=5, device=args.device)
        agent = MultiAgentDreamer(base_agent, num_agents=args.num_agents, device=args.device)
        
        # Experience buffer (larger for parallel agents)
        buffer = ExperienceBuffer(capacity=args.capacity, seq_len=args.seq_len)
        
        print(f"[*] Configured for {args.num_agents} parallel agents")
        print(f"[*] Buffer capacity: {args.capacity} transitions")
        print(f"[*] Graphics enabled: {not args.no_graphics}")
        print(f"[*] Connection timeout: {args.timeout}s")
        
        # Load checkpoint if resuming
        model_path = f"models/{args.run_id}/dreamer_model.pt"
        if args.resume and os.path.exists(model_path) and not args.force:
            print("[*] Loading checkpoint...")
            agent.load(model_path)

        # Connect to Unity
        print("[*] Waiting for Unity connection...")
        if args.env is None:
            print("[!] IMPORTANT: Press PLAY in the Unity Editor NOW!")
            print("[!] Waiting 10 seconds for Unity to start...")
            import time
            time.sleep(10)
        
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=args.time_scale)
        
        # Configurar timeout maior
        from mlagents_envs.environment import UnityEnvironment
        env = UnityEnvironment(
            file_name=args.env, 
            side_channels=[channel], 
            no_graphics=args.no_graphics,
            timeout_wait=args.timeout
        )
        env.reset()
        
        behavior_names = list(env.behavior_specs.keys())
        print(f"[*] Behaviors detected: {behavior_names}")
        
        # Training metrics
        step = 0
        episode = 0
        episode_rewards = []
        episode_lengths = []
        
        # Per-agent tracking - use agent keys (behavior_name_agent_id)
        obs_dict = {}
        episode_reward = {}
        episode_length = {}
        
        try:
            print("[*] Dreamer training started!")
            
            while True:
                actions_to_send = {}
                
                for name in behavior_names:
                    dec, term = env.get_steps(name)
                    
                    # Collect actions for all agents in this behavior
                    behavior_continuous_actions = []
                    behavior_discrete_actions = []
                    
                    # Handle terminal steps (multiple agents may finish)
                    for i in range(len(term)):
                        agent_id = term.agent_id[i]
                        reward = term.reward[i]
                        obs = term.obs[0][i]
                        agent_key = f"{name}_{agent_id}"
                        
                        # Initialize if first time seeing this agent
                        if agent_key not in episode_reward:
                            episode_reward[agent_key] = 0.0
                            episode_length[agent_key] = 0
                        
                        episode_reward[agent_key] += reward
                        episode_length[agent_key] += 1
                        
                        # Add terminal experience
                        if agent_key in obs_dict:
                            prev_action = agent.get_prev_action_numpy(agent_key)
                            buffer.add(obs_dict[agent_key], prev_action, reward, True)
                        
                        # Log episode
                        episode += 1
                        episode_rewards.append(episode_reward[agent_key])
                        episode_lengths.append(episode_length[agent_key])
                        
                        print(f"[{agent_key}] Ep {episode} | Reward: {episode_reward[agent_key]:.2f} | "
                              f"Steps: {episode_length[agent_key]} | Total steps: {step}")
                        
                        # Limpar nome para MLflow (remove ? = e outros caracteres especiais)
                        safe_name = name.replace("?", "_").replace("=", "_").replace("&", "_")
                        safe_agent_key = agent_key.replace("?", "_").replace("=", "_").replace("&", "_")
                        
                        mlflow.log_metric(f"{safe_name}_reward", episode_reward[agent_key], step=episode)
                        mlflow.log_metric(f"{safe_name}_length", episode_length[agent_key], step=episode)
                        tb_writer.add_scalar(f"Rewards/{safe_agent_key}", episode_reward[agent_key], episode)
                        tb_writer.add_scalar(f"EpisodeLength/{safe_agent_key}", episode_length[agent_key], episode)
                        
                        # Reset
                        episode_reward[agent_key] = 0.0
                        episode_length[agent_key] = 0
                        agent.init_agent_state(agent_key)
                    
                    # Handle decision steps (all active agents)
                    for i in range(len(dec)):
                        agent_id = dec.agent_id[i]
                        obs = dec.obs[0][i]
                        reward = dec.reward[i]
                        agent_key = f"{name}_{agent_id}"
                        
                        # Initialize if first time seeing this agent
                        if agent_key not in episode_reward:
                            episode_reward[agent_key] = 0.0
                            episode_length[agent_key] = 0
                        
                        # Store experience
                        if agent_key in obs_dict:
                            prev_action = agent.get_prev_action_numpy(agent_key)
                            buffer.add(obs_dict[agent_key], prev_action, reward, False)
                        
                        obs_dict[agent_key] = obs
                        episode_reward[agent_key] += reward
                        episode_length[agent_key] += 1
                        
                        # Initialize agent state if not exists
                        if agent_key not in agent.states:
                            agent.init_agent_state(agent_key)
                            print(f"[DEBUG] Initialized agent state for {agent_key}")
                        
                        # Get action from Dreamer for this specific agent
                        unity_action, action_np = agent.act(agent_key, obs, deterministic=False)
                        
                        # Debug: print first action for each agent
                        if step < 5:
                            print(f"[DEBUG] {agent_key} action: continuous={unity_action.continuous}, discrete={unity_action.discrete}")
                        
                        # Unity action: (1, continuous_dim) or (1, discrete_dim)
                        behavior_continuous_actions.append(unity_action.continuous[0])
                        if len(unity_action.discrete) > 0:
                            behavior_discrete_actions.append(unity_action.discrete[0])
                    
                    # Stack actions for all agents in this behavior
                    if len(behavior_continuous_actions) > 0:
                        continuous_batch = np.stack(behavior_continuous_actions, axis=0)
                        if len(behavior_discrete_actions) > 0:
                            discrete_batch = np.stack(behavior_discrete_actions, axis=0)
                            actions_to_send[name] = ActionTuple(continuous=continuous_batch, discrete=discrete_batch)
                        else:
                            actions_to_send[name] = ActionTuple(continuous=continuous_batch)
                
                # Send actions and step
                for name in actions_to_send:
                    env.set_actions(name, actions_to_send[name])
                
                env.step()
                step += 1
                
                # Train world model.
                # NOTE: Buffer holds single transitions; sample() builds [B, T] sequences
                # with replacement, so we only need len >= seq_len + margin, NOT
                # seq_len * batch_size (old gate was impossible with defaults).
                if step % 50 == 0 and len(buffer) >= args.seq_len * 8:
                    batch = buffer.sample(min(args.batch_size, max(8, len(buffer) // args.seq_len)))
                    if batch is not None:
                        obs_seq, action_seq, reward_seq, done_seq = batch
                        obs_seq = obs_seq.to(args.device)
                        action_seq = action_seq.to(args.device)
                        reward_seq = reward_seq.to(args.device)
                        
                        world_metrics = agent.train_world_model(obs_seq, action_seq, reward_seq, done_seq)
                        
                        for key, value in world_metrics.items():
                            tb_writer.add_scalar(key, value, step)
                
                # Train policy (somente se tiver dados suficientes no buffer)
                if step % 100 == 0 and len(buffer) >= args.seq_len * 8:
                    try:
                        # Imagination uses current RSSM state, not the full batch size.
                        # Clamp to avoid OOM on large --batch-size values.
                        imagine_batch = min(32, args.batch_size)
                        policy_metrics = agent.train_actor_critic(batch_size=imagine_batch, horizon=args.horizon)
                        
                        for key, value in policy_metrics.items():
                            tb_writer.add_scalar(key, value, step)
                    except Exception as e:
                        print(f"[!] Policy training error: {e}")
                
                # Save checkpoint
                if episode % 50 == 0 and episode > 0:
                    os.makedirs(f"models/{args.run_id}", exist_ok=True)
                    agent.save(model_path)
                    
                    # Export ONNX
                    try:
                        export_onnx(agent, f"models/{args.run_id}/dreamer_actor.onnx")
                    except Exception as e:
                        print(f"[!] ONNX export failed: {e}")
                
                # Progress log
                if step % 1000 == 0:
                    avg_reward = np.mean(episode_rewards[-50:]) if episode_rewards else 0
                    avg_length = np.mean(episode_lengths[-50:]) if episode_lengths else 0
                    print(f"[Progress] Step: {step} | Episodes: {episode} | "
                          f"Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.1f}")

        except KeyboardInterrupt:
            print("[!] Training interrupted by user. Saving final model...")
            os.makedirs(f"models/{args.run_id}", exist_ok=True)
            agent.save(f"models/{args.run_id}/dreamer_final.pt")
            
            try:
                export_onnx(agent, f"models/{args.run_id}/dreamer_actor_final.onnx")
            except Exception as e:
                print(f"[!] ONNX export failed: {e}")
            
            mlflow.log_artifact(f"models/{args.run_id}", "models_final")
            
        finally:
            env.close()
            tb_writer.close()
            print("[*] Training complete!")


if __name__ == "__main__":
    main()

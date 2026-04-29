"""
Behavior Cloning (BC) for Mario Parkour
Trains a neural network to imitate human demonstrations.

Usage:
    python train_behavior_cloning.py --data ../HybridTrainingData/ --epochs 100 --output models/bc_model.pth
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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
OBSERVATION_DIM = 30
ACTION_DIM = 3  # [joystick_x, joystick_y, jump]

class MarioDataset(Dataset):
    """Mario transition dataset."""
    
    def __init__(self, data_dir, min_episode_length=10):
        self.transitions = []
        self.load_data(data_dir, min_episode_length)
        print(f"Dataset loaded: {len(self.transitions)} transitions")
        
    def load_data(self, data_dir, min_length):
        """Loads all JSON files from the directory."""
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {data_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for episode in data.get('episodes', []):
                    if episode.get('stepCount', 0) < min_length:
                        continue
                    
                    for transition in episode.get('transitions', []):
                        obs = np.array(transition.get('observations', []), dtype=np.float32)
                        actions = np.array(transition.get('actions', []), dtype=np.float32)
                        
                        if len(obs) >= OBSERVATION_DIM and len(actions) >= ACTION_DIM:
                            self.transitions.append({
                                'obs': obs[:OBSERVATION_DIM],
                                'actions': actions[:ACTION_DIM]
                            })
                            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        t = self.transitions[idx]
        return torch.FloatTensor(t['obs']), torch.FloatTensor(t['actions'])


class MarioPolicy(nn.Module):
    """Behavior policy for Mario."""
    
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
        
        # Outputs: 2 continuous (joystick) + 1 discrete (jump)
        self.continuous_head = nn.Linear(hidden_size, 2)
        self.jump_head = nn.Linear(hidden_size, 1)  # Jump probability
        
    def forward(self, obs):
        features = self.network(obs)
        
        continuous_actions = torch.tanh(self.continuous_head(features))
        jump_prob = torch.sigmoid(self.jump_head(features))
        
        return continuous_actions, jump_prob
    
    def predict(self, obs):
        """Inference (no gradient)."""
        with torch.no_grad():
            cont, jump = self.forward(obs)
            jump_action = (jump > 0.5).float()
            return torch.cat([cont, jump_action], dim=-1)


def train_behavior_cloning(args):
    """Trains Behavior Cloning."""
    
    print("=" * 60)
    print("Behavior Cloning - Mario Parkour")
    print("=" * 60)
    
    # Load data
    dataset = MarioDataset(args.data, min_episode_length=args.min_episode_length)
    
    if len(dataset) == 0:
        print("ERROR: No transitions loaded!")
        return
    
    # Train/validation split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    model = MarioPolicy(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Device: {device}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # History
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Training
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for obs, actions in train_loader:
            obs, actions = obs.to(device), actions.to(device)
            
            optimizer.zero_grad()
            
            cont_pred, jump_pred = model(obs)
            
            # Loss for continuous actions (joystick)
            loss_cont = mse_loss(cont_pred, actions[:, :2])
            
            # Loss for jump (BCE)
            loss_jump = bce_loss(jump_pred.squeeze(), actions[:, 2])
            
            # Total loss
            loss = loss_cont + loss_jump
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for obs, actions in val_loader:
                obs, actions = obs.to(device), actions.to(device)
                
                cont_pred, jump_pred = model(obs)
                
                loss_cont = mse_loss(cont_pred, actions[:, :2])
                loss_jump = bce_loss(jump_pred.squeeze(), actions[:, 2])
                loss = loss_cont + loss_jump
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': vars(args)
            }, args.output)
            print(f"  -> Model saved (val loss: {avg_val_loss:.6f})")
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Model saved at: {args.output}")
    print(f"Best val loss: {best_val_loss:.6f}")
    
    # Plot curves
    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Behavior Cloning Training')
        plt.grid(True)
        plt.savefig(args.output.replace('.pth', '_training.png'))
        print(f"Plot saved at: {args.output.replace('.pth', '_training.png')}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Behavior Cloning for Mario')
    parser.add_argument('--data', type=str, required=True, 
                        help='Directory with JSON data')
    parser.add_argument('--output', type=str, default='models/bc_model.pth',
                        help='Path to save model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--min-episode-length', type=int, default=10,
                        help='Minimum episode length')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training curves')
    
    args = parser.parse_args()
    
    train_behavior_cloning(args)


if __name__ == '__main__':
    main()

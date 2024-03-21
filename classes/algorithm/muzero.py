### WIP ###
# MuZero Algorithm
# MuZero is a model-based reinforcement learning algorithm that learns a model of the environment and uses it to plan.
# The algorithm is based on the AlphaZero algorithm, which is a model-free reinforcement learning algorithm.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

class MuZeroNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(MuZeroNetwork, self).__init__()
        # Representation Function
        self.representation = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Dynamics Function (predicts next hidden state and reward)
        self.dynamics_state = nn.Sequential(
            nn.Linear(128 + action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.dynamics_reward = nn.Sequential(
            nn.Linear(128, 1),
            nn.ReLU()
        )
        
        # Prediction Function (policy and value)
        self.prediction_policy = nn.Sequential(
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        self.prediction_value = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
    def forward(self, state, action=None, hidden_state=None):
        if hidden_state is None:
            hidden_state = self.representation(state)
        if action is not None:
            hidden_state = self.dynamics_state(torch.cat([hidden_state, action], dim=-1))
            reward = self.dynamics_reward(hidden_state)
        else:
            reward = None
        policy = self.prediction_policy(hidden_state)
        value = self.prediction_value(hidden_state)
        return policy, value, hidden_state, reward
    

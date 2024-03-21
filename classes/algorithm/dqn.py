# Deep Q-learning Network
# Antonio van Dyck
# 21-03-2024
# 

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from classes.game.player import Player

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# state_size = 3 (player's sum, dealer's sum, usable_ace)
state_size = 3  

# action_size = 2 (hit, stand)
action_size = 2

# Deep Q-learning Network
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Deep Q-learning Agent
class DQNAgent(Player):
    def __init__(self):
        super().__init__("DQN Agent")
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.model = DQNNetwork(self.state_size, self.action_size).to(device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.batch_size = 32 

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.LongTensor([action]).to(device)
            reward = torch.FloatTensor([reward]).to(device)
            done = torch.BoolTensor([done]).to(device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state).detach())
            predicted_target = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            loss = self.loss_fn(predicted_target, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

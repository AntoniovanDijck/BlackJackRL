### WIP ###
### CUDA ONLY ###

# MuZero Algorithm
# MuZero is a model-based reinforcement learning algorithm that learns a model of the environment and uses it to plan.
# The algorithm is based on the AlphaZero algorithm, which is a model-free reinforcement learning algorithm.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import math

class BlackjackEnvironment:
    def __init__(self):
        self.deck = Deck()
        self.player = Player("Player")
        self.dealer = Dealer("Dealer")
    
    def reset(self):
        self.deck = Deck()  # Reset the deck
        self.player.reset()
        self.dealer.reset()
        
        # Deal initial cards
        for _ in range(2):
            self.player.add_card(self.deck.deal())
            self.dealer.add_card(self.deck.deal())
        
        # Here you would return the initial state of the game
        # The state could be a representation of the player's and dealer's hands
        return self.get_state()

    def step(self, action):
        # Execute the action (hit or stand) and return the new state, reward, and done status
        done = False
        reward = 0
        
        if action == 0:  # Assuming 0 is 'hit' and 1 is 'stand'
            self.player.add_card(self.deck.deal())
            if self.player.value > 21:
                done = True
                reward = -1  # Assuming the reward for busting is -1
                
        # Dealer's turn to play if player stands or after hitting
        if action == 1 or self.player.value > 21:
            while self.dealer.value < 17:
                self.dealer.add_card(self.deck.deal())
            done = True
            # Determine the reward based on the game outcome
            
        # Update this method to determine the reward based on the final outcomes
        return self.get_state(), reward, done, {}
    
    def get_state(self):
        # Implement a method to retrieve the current state
        # Could be as simple as the player's and dealer's current values and whether the player has an ace
        state = [self.player.value, self.dealer.show_first_card().value if self.dealer.show_first_card() else 0, int(any(card.rank == 'Ace' for card in self.player.hand))]
        return np.array(state, dtype=np.float32)



class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = self.determine_value()

    def determine_value(self):
        if self.rank in ['Jack', 'Queen', 'King']:
            return 10
        elif self.rank == 'Ace':
            return 11
        return int(self.rank)

class Deck:
    def __init__(self):
        ranks = [str(n) for n in range(2, 11)] + ['Jack', 'Queen', 'King', 'Ace']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]
        np.random.shuffle(self.cards)

    def deal(self):
        return self.cards.pop()

class Player:
    def __init__(self, name):
        self.name = name
        self.reset()

    def add_card(self, card):
        self.hand.append(card)
        self.adjust_for_ace()

    def adjust_for_ace(self):
        self.value = sum(card.value for card in self.hand)
        aces = sum(card.rank == 'Ace' for card in self.hand)
        while self.value > 21 and aces:
            self.value -= 10
            aces -= 1

    def reset(self):
        self.hand = []
        self.value = 0

class Dealer(Player):
    def show_first_card(self):
        return self.hand[0] if self.hand else None
    
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
            nn.Linear(128 + 1, 128),
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
            # Assuming action is a 1D tensor of shape [batch_size], we add an extra dimension to match hidden_state
            action = action.view(-1, 1)  # Reshape action to [batch_size, 1] to match hidden_state's [batch_size, hidden_state_size]
            hidden_state = self.dynamics_state(torch.cat([hidden_state, action], dim=-1))
            reward = self.dynamics_reward(hidden_state)
        else:
            reward = None
        policy = self.prediction_policy(hidden_state)
        value = self.prediction_value(hidden_state)
        return policy, value, hidden_state, reward


class MuZeroAgent(Player):
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.network = MuZeroNetwork(state_size, action_size).to(self.config.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=1000)
        self.scaler = GradScaler()  # For mixed precision training
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def sample_batch(self):
        # Directly use the sample method of ReplayBuffer to get the batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.config.batch_size)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def select_action(self, state, deterministic=False):
        # Simplified policy selection, not using MCTS for this example
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        with torch.no_grad():
            policy, _, _, _ = self.network(state_tensor)
        if deterministic:
            return policy.argmax(dim=-1).item()
        else:
            return np.random.choice(len(policy[0]), p=policy[0].cpu().numpy())

    def train_step(self, batch):
        # A simplified training step, assuming `batch` is a tuple of (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = batch
        actions = actions.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        
        with autocast():  # Mixed precision context
            _, _, hidden_states, _ = self.network(states)
            _, _, next_hidden_states, _ = self.network(next_states, actions)
            _, value, _, _ = self.network(states)
            _, next_value, _, _ = self.network(next_states)

            # Simplified loss calculation
            value_loss = ((rewards + self.config.gamma * next_value * (1 - dones) - value) ** 2).mean()
            reward_loss = torch.tensor(0)  # Placeholder for reward prediction loss
            policy_loss = torch.tensor(0)  # Placeholder for policy loss

            loss = value_loss + reward_loss + policy_loss

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()


class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.learning_rate = 1e-3
        self.gamma = 0.99  # Discount factor for future rewards
        self.num_simulations = 10  # Number of MCTS simulations
        self.update_frequency = 10 # How often to update the network
        self.batch_size = 32 # Batch size for training
        self.hidden_state_size = 128  # Size of the hidden state in the network
        self.update_frequency = 10  # How often to update the network

# Test
#config = Config()
#agent = MuZeroAgent(state_size=10, action_size=2, config=config)

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def expand_node(node, action_size, actions_priors):
    for i in range(action_size):
        node.children[i] = Node(actions_priors[i])

def ucb_score(parent, child):
    pb_c = 1.25
    pb_c_base = 19652
    pb_c_init = 1.25
    pb_c = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0
    return prior_score + value_score

def select_child(node):
    _, action, child = max((ucb_score(node, child), action, child) for action, child in node.children.items())
    return action, child

def simulate(agent, state, config):
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.config.device)
    
    # Initial prediction from the representation
    policy, value, hidden_state, _ = agent.network(state_tensor)
    
    root = Node(0)
    # You should adjust this based on how your Node class and expand_node function are designed
    actions_priors = policy.squeeze().tolist()  # Assuming policy is a tensor of action probabilities
    expand_node(root, agent.action_size, actions_priors)

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(node)
            search_path.append(node)
        
        # Here, you would simulate the dynamics function but let's fix the basics first.

    return root


def select_action(agent, state, config, deterministic=False):
    root = simulate(agent, state, config)
    visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
    if deterministic:
        action = max(visit_counts, key=lambda x: x[0])[1]
    else:
        visit_count_sum = sum(visit_count for visit_count, _ in visit_counts)
        probs = [visit_count / visit_count_sum for visit_count, _ in visit_counts]
        action = np.random.choice([action for _, action in visit_counts], p=probs)
    return action

            
def train_network(agent, config, episodes):
    env = BlackjackEnvironment()
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
        
        # Check if the replay buffer has enough samples
        if len(agent.replay_buffer) >= config.batch_size:
            # Only then perform the training step
            if episode % config.update_frequency == 0:
                states, actions, rewards, next_states, dones = agent.sample_batch()
                states = torch.FloatTensor(states).to(config.device)
                actions = torch.LongTensor(actions).unsqueeze(-1).to(config.device)
                rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(config.device)
                next_states = torch.FloatTensor(next_states).to(config.device)
                dones = torch.FloatTensor(dones).unsqueeze(-1).to(config.device)
                agent.train_step((states, actions, rewards, next_states, dones))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        state, action, reward, next_state, done = zip(*[self.buffer[idx] for idx in indices])
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

# train the agent
config = Config()
agent = MuZeroAgent(state_size=3, action_size=2, config=config)
train_network(agent, config, episodes=100)
            

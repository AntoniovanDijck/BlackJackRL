# MuZero Algorithm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from collections import deque
import math
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import os 

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
        done = False
        reward = 0
        
        if action == 0:  # hit
            self.player.add_card(self.deck.deal())
            if self.player.value > 21:
                done = True
                reward = -1  # Lose by going over
            
        if action == 1 or self.player.value > 21:  # stand or bust
            while self.dealer.value < 17:
                self.dealer.add_card(self.deck.deal())
            done = True
            if self.player.value > 21:
                reward = -1  # Player busts
            elif self.dealer.value > 21 or self.player.value > self.dealer.value:
                reward = 1  # Win by dealer busting or having a higher value
            elif self.player.value < self.dealer.value:
                reward = -1  # Lose by having a lower value
            else:
                reward = 0  # Draw

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

class Config:
    def __init__(self):
        self.device = torch.device("mps" if torch.cuda.is_available() else "cpu")
        self.learning_rate = 1e-3
        self.gamma = 0.99  # Discount factor for future rewards
        self.num_simulations = 10  # Number of MCTS simulations
        self.update_frequency = 10 # How often to update the network
        self.batch_size = 32 # Batch size for training
        self.hidden_state_size = 128  # Size of the hidden state in the network
        self.update_frequency = 10  # How often to update the network

class MuZeroAgent(Player):

    def __init__(self, state_size=3, action_size=2, config=Config()):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.network = MuZeroNetwork(state_size, action_size).to(self.config.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=1000)
        #self.scaler = GradScaler()  # For mixed precision training
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
        
    def load(self, name):
        self.network.load_state_dict(torch.load(name))

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.config.device)
        actions = torch.LongTensor(actions).unsqueeze(-1).to(self.config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.config.device)
        
        # Forward pass
        _, _, hidden_states, predicted_rewards = self.network(states, actions)
        _, next_value, _, _ = self.network(next_states)
        _, value, _, _ = self.network(states)
        
        # Compute losses
        value_loss = ((rewards + self.config.gamma * next_value.detach() * (1 - dones) - value) ** 2).mean()
        reward_loss = ((predicted_rewards - rewards) ** 2).mean()  # Reward prediction loss

        loss = value_loss + reward_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

    # Initial prediction from the representation function
    policy, value, hidden_state, _ = agent.network(state_tensor)

    root = Node(0)
    actions_priors = policy.squeeze().tolist()  # Assuming policy is a tensor of action probabilities
    expand_node(root, agent.action_size, actions_priors)

    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        done = False
        virtual_reward = 0
        virtual_state = hidden_state

        while node.expanded():
            action, node = select_child(node)
            search_path.append(node)
            
            # Predict next state and reward using the dynamics function
            action_tensor = torch.tensor([[action]], dtype=torch.float32).to(agent.config.device)
            with torch.no_grad():
                _, _, virtual_state, virtual_reward_tensor = agent.network(None, action_tensor, virtual_state)
                virtual_reward += virtual_reward_tensor.item()

            if done:
                break

        # Update tree with the simulation results
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += virtual_reward  # Update with the cumulative reward from the simulation
        
        # If not done, expand the new node
        if not done:
            # Use the policy from the last predicted state to expand the node
            policy, _, _, _ = agent.network(None, None, virtual_state)
            actions_priors = policy.squeeze().tolist()
            expand_node(node, agent.action_size, actions_priors)

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
          
def train_mu0_network(episodes):
    config = Config()
    agent = MuZeroAgent(state_size=3, action_size=2, config=config)
    env = BlackjackEnvironment()
    #load trained weights
    #check if the file exists
    if os.path.exists("weights/muzero.pth"):
        agent.load("weights/muzero.pth")
        print("Agent loaded successfully")
    else:
        print("No weights found, training from scratch")
    

    for episode in tqdm(range(episodes)):
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
                #print(f"Episode {episode}: Network updated")

    # Save the trained agent
    torch.save(agent.network.state_dict(), "weights/muzero.pth")
    print("Agent saved successfully")

#Test the agent
def test_m0_agent(weight_filename="weights/muzero.pth", env=BlackjackEnvironment(), episodes=100):
    #load the trained agent
    agent = MuZeroAgent(state_size=3, action_size=2, config=Config())

    outcomes = Counter()

    #load trained weights
    agent.load(weight_filename)

    for _ in tqdm(range(episodes)):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _ = env.step(action)
            state = next_state

        # Determine the game outcome based on final states and rewards
        if reward == 1:
            if env.player.value <= 21 and env.dealer.value < env.player.value or env.dealer.value > 21:
                outcomes['Player wins'] += 1
            else:
                outcomes['Dealer wins'] += 1
        elif reward == -1:
            if env.player.value > 21:
                outcomes['Player busts'] += 1
            else:
                outcomes['Dealer wins'] += 1
        else:
            if env.player.value == env.dealer.value:
                outcomes['Push'] += 1
            elif env.dealer.value > 21:
                outcomes['Dealer busts'] += 1
                

    # Displaying the outcomes
    print("Game outcomes:")
    for outcome, count in outcomes.items():
        print(f"{outcome}: {count}")

    # Optional: Plotting the outcomes if needed
    labels, values = zip(*outcomes.items())
    indexes = np.arange(len(labels))
    width = 1

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels, rotation='vertical')
    plt.show()

    return outcomes

# train the agent
train_mu0_network(episodes=100000000)

# test the agent
test_m0_agent(episodes=100)

#test the agent with kk game
def play_against_agent():
    env = BlackjackEnvironment()
    agent = MuZeroAgent(state_size=3, action_size=2, config=Config())

    # Load the trained agent
    weight_filename = "weights/muzero.pth"
    agent.load(weight_filename)
    
    state = env.reset()
    done = False
    while not done:
        # Display current player and dealer cards
        print("Player's hand:", [f"{card.rank} of {card.suit}" for card in env.player.hand], f"Value: {env.player.value}")
        print("Dealer's hand: [Hidden], {0} of {1}".format(env.dealer.hand[1].rank, env.dealer.hand[1].suit))
        
        # Player's turn
        action = input("Choose your action (Hit = 0 / Stand = 1): ")
        while action not in ["0", "1"]:
            action = input("Invalid input. Choose your action (Hit = 0 / Stand = 1): ")
        action = int(action)
        
        if action == 0:
            print("Player hits.")
        else:
            print("Player stands.")
            done = True

        state, reward, done, _ = env.step(action)

        if done:
            # It's the dealer's (agent's) turn. The agent plays until the game ends.
            print("Dealer's turn.")
            while done:
                action = agent.select_action(state, deterministic=True)  # Agent makes a decision
                state, reward, done, _ = env.step(action)  # Update the environment based on the agent's action
                print("Dealer's hand:", [f"{card.rank} of {card.suit}" for card in env.dealer.hand], f"Value: {env.dealer.value}")
                if action == 0:
                    print("Dealer hits.")
                else:
                    print("Dealer stands.")
                    break

    # Final outcome
    print("\nFinal Hands:")
    print("Player's hand:", [f"{card.rank} of {card.suit}" for card in env.player.hand], f"Value: {env.player.value}")
    print("Dealer's hand:", [f"{card.rank} of {card.suit}" for card in env.dealer.hand], f"Value: {env.dealer.value}")
    if reward == 1:
        print("Player wins!")
    elif reward == -1:
        print("Dealer wins!")
    else:
        print("It's a draw.")

# Now, let's play!
play_against_agent()


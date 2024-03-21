# Simple Q_table based learning algorithm for Blackjack game

import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os 

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

class QLearningAgent(Player):
    def __init__(self):
        super().__init__("QLearning Agent")
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_rate = 0.95
        self.exploration_rate = 1.0
        self.exploration_decay = 0.99

    def decide_action(self, dealer_card_value):
        state = (self.value, dealer_card_value)
        if np.random.rand() < self.exploration_rate or state not in self.q_table:
            return np.random.choice(['hit', 'stand'])
        return 'hit' if np.argmax(self.q_table[state]) == 0 else 'stand'

    def update_q_table(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2)
        action_index = 0 if action == 'hit' else 1
        next_max = np.max(self.q_table[next_state]) if next_state in self.q_table else 0
        self.q_table[state][action_index] += self.learning_rate * (reward + self.discount_rate * next_max - self.q_table[state][action_index])
    
    def save_q_table(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename='q_table.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("File not found. Starting with a new Q-table.")

class BlackjackGame:
    def __init__(self):
        self.player = QLearningAgent()
        self.dealer = Dealer("Dealer")
        self.deck = Deck()

    def reset_deck(self):
        self.deck = Deck()

    def play_round(self, learn=False):
        self.reset_deck()
        self.player.reset()
        self.dealer.reset()

        for _ in range(2):
            self.player.add_card(self.deck.deal())
            self.dealer.add_card(self.deck.deal())

        dealer_first_card_value = self.dealer.show_first_card().value
        player_state = (self.player.value, dealer_first_card_value)
        action = None  # Initialize action to None

        while self.player.value < 21:
            action = self.player.decide_action(dealer_first_card_value)
            if action == 'hit':
                self.player.add_card(self.deck.deal())
            next_state = (self.player.value, dealer_first_card_value)
            if learn and action == 'hit':  # Update Q-table only on 'hit'
                reward = 0  # Intermediate reward is 0; final reward will be determined after the dealer's turn.
                self.player.update_q_table(player_state, action, reward, next_state)
            player_state = next_state  # Update player_state to next_state
            if self.player.value > 21 or action == 'stand':
                break  # Break if player busts or decides to stand

        # Process dealer's actions outside of the player's action loop
        while self.dealer.value < 17:
            self.dealer.add_card(self.deck.deal())

        outcome = self.determine_outcome()
        if learn:
            # Final reward based on game outcome
            reward = {'Player wins': 1, 'Dealer wins': -1, 'Player busts': -1, 'Dealer busts': 1, 'Push': 0}[outcome]
            self.player.update_q_table(player_state, action, reward, (None, None))  # End of game state is None
            self.player.exploration_rate *= self.player.exploration_decay

        return outcome


    def determine_outcome(self):
        if self.player.value > 21:
            return 'Player busts'
        elif self.dealer.value > 21:
            return 'Dealer busts'
        elif self.player.value > self.dealer.value:
            return 'Player wins'
        elif self.player.value < self.dealer.value:
            return 'Dealer wins'
        else:
            return 'Push'
        
def train_Q_agent(rounds=1000000, save_filename='weights/q_table.pkl',
                learning_rate=0.1, discount_rate=0.95,
                exploration_rate=1.0, exploration_decay=0.99,
                early_stopping_rounds=10000, improvement_threshold=0.01):
    game = BlackjackGame()
    game.player.load_q_table(save_filename)  # Load the Q-table if it exists

    # Set the hyperparameters
    game.player.learning_rate = learning_rate
    game.player.discount_rate = discount_rate
    game.player.exploration_rate = exploration_rate
    game.player.exploration_decay = exploration_decay

    # Initialize variables for early stopping
    best_win_rate = 0
    rounds_without_improvement = 0

    outcomes = []
    
    for round_number in tqdm(range(rounds), desc="Training Progress"):
        outcome = game.play_round(learn=True)
        outcomes.append(outcome)

        # Early stopping check
        if (round_number + 1) % early_stopping_rounds == 0:
            current_win_rate = outcomes.count('Player wins') / early_stopping_rounds
            if current_win_rate - best_win_rate < improvement_threshold:
                rounds_without_improvement += 1
            else:
                rounds_without_improvement = 0
                best_win_rate = current_win_rate

            outcomes = []  # Reset outcomes for the next batch

            if rounds_without_improvement >= 2:  # Stop if no improvement in two consecutive checks
                print(f"Early stopping triggered after {round_number + 1} rounds.")
                break

    game.player.save_q_table(save_filename)  # Save the Q-table after training
    print("Training completed.")


def test_Q_agent(rounds=1000, save_filename='weights/q_table.pkl', report_interval=100):
    game = BlackjackGame()
    game.player.load_q_table(save_filename)

    # Initialize metrics
    results = {'Player wins': 0, 'Dealer wins': 0, 'Player busts': 0, 'Dealer busts': 0, 'Push': 0}
    win_rates = []
    loss_rates = []
    push_rates = []
    intervals = []

    for i in tqdm(range(1, rounds + 1)):
        result = game.play_round(learn=False)
        results[result] += 1

        # Report and collect metrics at each interval
        if i % report_interval == 0 or i == rounds:
            total_games = sum(results.values())
            win_rate = results['Player wins'] / total_games
            loss_rate = (results['Dealer wins'] + results['Player busts']) / total_games
            push_rate = results['Push'] / total_games
            win_rates.append(win_rate)
            loss_rates.append(loss_rate)
            push_rates.append(push_rate)
            intervals.append(i)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(intervals, win_rates, label='Win Rate')
    plt.plot(intervals, loss_rates, label='Loss Rate')
    plt.plot(intervals, push_rates, label='Push Rate')
    plt.xlabel('Rounds')
    plt.ylabel('Rate')
    plt.title('Agent Performance Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print final metrics
    for outcome, count in results.items():
        print(f"{outcome}: {count}")

train_Q_agent(rounds=100000, learning_rate=0.1, discount_rate=0.95, exploration_rate=1.0, exploration_decay=0.99, early_stopping_rounds=10000)
print("\nTesting the trained agent with detailed metrics and plotting performance:")
test_Q_agent(rounds=1000, report_interval=1)
# BlackJack game class
# Antonio van Dyck
# 21-03-2024
# 

import numpy as np
from classes.game.dealer import Dealer
from classes.game.deck import Deck
from classes.algorithm.dqn import DQNAgent

class BlackjackGame:
    def __init__(self):
        self.player = DQNAgent()
        self.dealer = Dealer("Dealer")
        self.deck = Deck()

    def reset_deck(self):
        self.deck = Deck()

    def play_round(self, learn=False):
        self.reset_deck()
        self.player.reset()
        self.dealer.reset()
        done = False

        # Deal two cards each
        for _ in range(2):
            self.player.add_card(self.deck.deal())
            self.dealer.add_card(self.deck.deal())

        # Player's turn
        while self.player.value < 21:
            state = self.get_state(self.player, self.dealer)
            action = self.player.act(state)  # DQN decides
            if action == 0:
                self.player.add_card(self.deck.deal())
            else:
                break # Stand
            
            next_state = self.get_state(self.player, self.dealer)
            done = self.player.value > 21 or action == 1
            reward = 0 if not done else self.calculate_reward(self.player, self.dealer)
        
            # Store the experience in replay memory
            if learn:
                self.player.remember(state, action, reward, next_state, done)
            
            if done:
                break

        # Dealer's turn
        while not done and self.dealer.value < 17:
            self.dealer.add_card(self.deck.deal())
    
        outcome = self.determine_outcome()  # Determine game outcome
    
        # Learning part: replay to train neural network
        if learn:
            self.player.replay()
            self.player.update_target_model()

        return outcome
    def get_state(self, player, dealer):
        dealer_first_card_value = dealer.show_first_card().value if dealer.show_first_card() else 0
        has_ace = 1 if any(card.rank == 'Ace' for card in player.hand) else 0
        return np.array([player.value, dealer_first_card_value, has_ace]).reshape(1, -1)

    def calculate_reward(self, player, dealer):
        if player.value > 21:
            return -1
        elif dealer.value > 21 or player.value > dealer.value:
            return 1
        elif player.value < dealer.value:
            return -1
        return 0


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
                   

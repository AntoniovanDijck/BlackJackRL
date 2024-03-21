# Deck Class
# Antonio van Dyck
# 21-03-2024
# 

import random
from classes.game.card import Card

class Deck:
    def __init__(self):
        self.reset()

    def reset(self):
        ranks = [str(n) for n in range(2, 11)] + ['Jack', 'Queen', 'King', 'Ace']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]
        self.reshuffle()

    def reshuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        if len(self.cards) < 15:  
            self.reset()
        return self.cards.pop()
# Player class
# Antonio van Dyck
# 21-03-2024
# 

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
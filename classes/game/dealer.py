from classes.game.player import Player

class Dealer(Player):
    def show_first_card(self):
        return self.hand[0] if self.hand else None
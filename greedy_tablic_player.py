from tablic_player import TablicPlayer
from tablic import Tablic

class GreedyTablicPlayer(TablicPlayer):
    def __init__(self):
        pass

    @classmethod
    def get_take_value(cls, played_card, take):
        if len(take) == 0:
            return (0, 0)
        tricks = (Tablic.card_to_tricks(played_card) 
                 + sum(Tablic.card_to_tricks(card) for card in take))
        return (tricks, len(take) + 1)

    @classmethod
    def find_best_play(cls, game):
        hand = game.get_hand(game.current_player)
        return max(Tablic.get_valid_takes(game.table, hand), 
                    key=lambda x: cls.get_take_value(*x))

from tablic_player import TablicPlayer
from tablic import Tablic

class HumanTablicPlayer(TablicPlayer):
    def __init__(self):
        pass

    @classmethod
    def find_best_play(cls, game):
        hand = game.get_hand(game.current_player)
        valid_takes = list(Tablic.get_valid_takes(game.table, hand))
        print()
        print(game.table)
        print(hand)
        for ind, val in enumerate(valid_takes):
            print(ind, val)
        take_to_play = int(input())
        return valid_takes[take_to_play]

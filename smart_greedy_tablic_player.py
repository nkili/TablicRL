from tablic_player import TablicPlayer
from tablic import Tablic
from greedy_tablic_player import GreedyTablicPlayer


class SmartGreedyTablicPlayer(TablicPlayer):
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
    def solve_hand(cls, game, player, path=[]):
        greedy_player = GreedyTablicPlayer()
        hand = sorted(game._hands[player])
        max_points = 0
        if game._is_terminal or (len(hand) == game._hand_size and len(path) > 0):
            return game._rewards[player], path

        best_path = []
        table = game._table

        valid_takes = list(Tablic.get_valid_takes(table, hand))
        for card, take in valid_takes:
            game_copy = game.copy()
            game_copy.play_card(card, take)
            if not game_copy._is_terminal:
                greedy_card, greedy_take = greedy_player.play_turn(game_copy)
                game_copy.play_card(greedy_card, greedy_take)
            points, new_path = cls.solve_hand(game_copy, player, path + [(card, take)])
            if points > max_points or max_points == 0:
                max_points = points
                best_path = new_path
        return max_points, best_path

    @classmethod
    def find_best_play(cls, game):
        if (game._deck_pointer >= len(game._deck)):
            points, move_list = cls.solve_hand(game.copy(), game.current_player)
            return move_list[0]

        hand = game.get_hand(game.current_player)
        return max(Tablic.get_valid_takes(game.table, hand), 
                    key=lambda x: cls.get_take_value(*x))


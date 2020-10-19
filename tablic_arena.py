from tablic import Tablic
from greedy_tablic_player import GreedyTablicPlayer
from reinforced_tablic_player import ReinforcedTablicPlayer
import random

class TablicArena:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
    
    def simulate_games(self, num_of_games, seed=None):
        random.seed(seed)
        p1_wins, draws, p2_wins = (0, 0, 0)
        total_results = [0, 0]
        for game in range(1, num_of_games+1):
            deck = Tablic.get_shuffled_deck()
            battle_results = [0, 0]
            results = self.start_and_play_game(deck)
            battle_results[0] += results[0]
            battle_results[1] += results[1]
            self.switch_players()
            results = self.start_and_play_game(deck)
            battle_results[1] += results[0]
            battle_results[0] += results[1]
            self.switch_players()
            total_results[0] += battle_results[0]
            total_results[1] += battle_results[1]
            if (battle_results[0] > battle_results[1]): p1_wins += 1
            if (battle_results[0] == battle_results[1]): draws += 1
            if (battle_results[0] < battle_results[1]): p2_wins += 1
        return (p1_wins, draws, p2_wins, total_results[0], total_results[1])

    def start_and_play_game(self, deck=None, print_plays=False):
        game = Tablic(deck)
        while not game.is_terminal:
            card, take = self.player1.play_turn(game)
            if (print_plays): print(f"Playing {card} and taking {take}.")
            game.play_card(card, take)
            card, take = self.player2.play_turn(game)
            if (print_plays): print(f"Playing {card} and taking {take}.")
            game.play_card(card, take)
        return game.rewards
    
    def switch_players(self):
        self.player1, self.player2 = self.player2, self.player1



if __name__ == '__main__':
    BATTLES = 50
    player0 = GreedyTablicPlayer()
    # player0 = ReinforcedTablicPlayer(None)
    # player0.load_model(f"Model")
    player1 = GreedyTablicPlayer()
    #player1 = ReinforcedTablicPlayer(None)
    # player1.load_model(f"Models/TD-cleared/g{GAMMAS[j]}e{50000}")
    arena = TablicArena(player0, player1)
    wins, draws, loses, pts, opp_pts = arena.simulate_games(BATTLES, 0)
    print(f"W:{wins}, D:{draws}, L:{loses}")
    print(pts / BATTLES, opp_pts / BATTLES)
    print()

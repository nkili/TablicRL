from tablic import Tablic
from smart_greedy_tablic_player import SmartGreedyTablicPlayer
from greedy_tablic_player import GreedyTablicPlayer
from reinforced_tablic_player import ReinforcedTablicPlayer
from human_tablic_player import HumanTablicPlayer
import random
import torch


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
            print(p1_wins, draws, p2_wins, end=" \r")
        return (p1_wins, draws, p2_wins, total_results[0], total_results[1])

    def start_and_play_game(self, deck=None, print_plays=False):
        game = Tablic(deck)
        while not game.is_terminal:
            card, take = self.player1.play_turn(game)
            if (print_plays): print(f"1- Playing {card} and taking {take}.")
            game.play_card(card, take)
            card, take = self.player2.play_turn(game)
            if (print_plays): print(f"2- Playing {card} and taking {take}.")
            game.play_card(card, take)
        return game.rewards
    
    def switch_players(self):
        self.player1, self.player2 = self.player2, self.player1



def getPlayer(name, device = "cpu"):
    if (name == "G"):
        return GreedyTablicPlayer()
    if (name == "SG"):
        return SmartGreedyTablicPlayer()
    if (name == "H"):
        return HumanTablicPlayer()
    player = ReinforcedTablicPlayer(None)
    player.load_model(f"Models/{name}")
    player.agent.device = torch.device(device)
    return player

if __name__ == '__main__':
    BATTLES = 1

    players = [
        #(1, getPlayer("g99e100000-ZS")),
        #(1, getPlayer("g99e100000-ZS")),
        #(1, getPlayer("g99e200000-ZS")),
        #(1, getPlayer("g99e300000-ZS")),
        #(2, getPlayer("g99e400000-ZS")),
        #(92, getPlayer("g99e1000000", "cuda")),
        (91, getPlayer("9/g99e1000000-ZS", "cuda")),
        #(63, getPlayer("6-MyBoy/g99e500000-ZS")),
        #(73, getPlayer("7-JamesBond/g99e500000-ZS", "cuda")),
        #(-3, getPlayer("g99e100000-ZS")),
        #(1, getPlayer("5-Two/Lg99e100000-ZS"))
        #(2, getPlayer("Lg99e100000-ZS")),
        #(0, getPlayer("G")),
        #(-1, getPlayer("H")),
        #(0, getPlayer("G")),
        #(100, getPlayer("SG")),
        
    ]

    for i in range(len(players)):
        for j in range(i+1, len(players)):
            arena = TablicArena(players[i][1], players[j][1])
            wins, draws, loses, pts, opp_pts = arena.simulate_games(BATTLES, 20)
            print(f"{players[i][0]}\t{players[j][0]}\t{wins}\t{draws}\t{loses}\t{pts / BATTLES}\t{opp_pts/BATTLES}")
            # print(f"Player {players[i][0]} vs Player Greedy")
            # print(f"W:{wins}, D:{draws}, L:{loses}")
            # print(pts / BATTLES, opp_pts / BATTLES)
            # print()

    # player0 = getPlayer("2-Epsilon010/g99e5000ZS", "cpu")
    # player1 = getPlayer("SG", "cuda")
    # arena = TablicArena(player0, player1)
    # wins, draws, loses, pts, opp_pts = arena.simulate_games(BATTLES, 0)
    # print(f"W:{wins}, D:{draws}, L:{loses}")
    # print(pts / BATTLES, opp_pts / BATTLES)
    # print()
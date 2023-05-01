from tablic import Tablic
from smart_greedy_tablic_player import SmartGreedyTablicPlayer
from greedy_tablic_player import GreedyTablicPlayer
from reinforced_tablic_player import ReinforcedTablicPlayer
from human_tablic_player import HumanTablicPlayer
import random
import torch

class TablicPlayerStats:
    def __init__(self, playerName):
        self.player = playerName
        self.games = 0
        self.wins = 0
        self.draws = 0
        self.loses = 0
        self.points = 0
        self.opp_pts = 0
        self.moves = 0
        self.takingPlays = 0
        self.greedyPointOpportunity = 0
        self.greedyCardOpportunity = 0
        self.sameGreedy = 0
        self.greedyPoints = 0
        self.greedyPointsAndCards = 0

    def print_player_stats(self):
        print(f"{self.player}, {self.games}, {self.wins}, {self.draws}, {self.loses}, {self.points/self.games}, {self.opp_pts/self.games}, {self.moves}, {self.takingPlays}, {self.greedyPointOpportunity}, {self.greedyCardOpportunity}, {self.sameGreedy}, {self.greedyPoints}, {self.greedyPointsAndCards}")

class TablicPlayerAnalysis:
    def __init__(self, players):
        self.players = players
        self.analysis = dict()
        for player in self.players:
            self.analysis[player[0]] = TablicPlayerStats(player[0])
        self.globalGames = 0
    
    def analyze_players(self, num_of_games, seed=None):
        for i in range(len(self.players)):
            player1 = self.players[i]
            for j in range(i, len(self.players)-1):
                player2 = self.players[-1]
                self.simulate_games_with_analysis(player1, player2, num_of_games, seed)
                break;
        return self.analysis

    def simulate_games_with_analysis(self, player1, player2, num_of_games, seed=None):
        player1_name, player1_player = player1
        player2_name, player2_player = player2
        random.seed(seed)
        for game in range(1, num_of_games+1):
            deck = Tablic.get_shuffled_deck()
            battle_results = [0, 0]
            results = self.start_and_play_game(player1, player2, deck)
            battle_results[0] += results[0]
            battle_results[1] += results[1]
            results = self.start_and_play_game(player2, player1, deck)
            battle_results[1] += results[0]
            battle_results[0] += results[1]
            self.globalGames  += 1
            print(self.globalGames, end="\r")
            self.analysis[player1_name].games += 1
            self.analysis[player2_name].games += 1
            self.analysis[player1_name].points += battle_results[0]
            self.analysis[player1_name].opp_pts += battle_results[1]
            self.analysis[player2_name].points += battle_results[1]
            self.analysis[player2_name].opp_pts += battle_results[0]
            if (battle_results[0] > battle_results[1]): 
                self.analysis[player1_name].wins += 1
                self.analysis[player2_name].loses += 1
            if (battle_results[0] == battle_results[1]):
                self.analysis[player1_name].draws += 1
                self.analysis[player2_name].draws += 1
            if (battle_results[0] < battle_results[1]):
                self.analysis[player2_name].wins += 1
                self.analysis[player1_name].loses += 1

    def analyze_play(self, playerName, card, take, game):
        greedyCard, greedyTake = GreedyTablicPlayer.find_best_play(game)
        greedyTakePoints, greedyTakeCards = GreedyTablicPlayer.get_take_value(greedyCard, greedyTake)
        playerTakePoints, playerTakeCards = GreedyTablicPlayer.get_take_value(card, take)
        self.analysis[playerName].moves += 1
        self.analysis[playerName].takingPlays += 1 if (playerTakeCards > 0) else 0
        self.analysis[playerName].greedyPointOpportunity += 1 if (greedyTakePoints > 0) else 0
        self.analysis[playerName].greedyCardOpportunity += 1 if (greedyTakeCards > 0) else 0
        self.analysis[playerName].sameGreedy += 1 if (greedyCard == card and greedyTake == take) else 0
        self.analysis[playerName].greedyPoints += 1 if (greedyTakePoints > 0 and greedyTakePoints == playerTakePoints) else 0
        self.analysis[playerName].greedyPointsAndCards += 1 if (greedyTakeCards > 0 and greedyTakePoints == playerTakePoints and greedyTakeCards == playerTakeCards) else 0


    def start_and_play_game(self, player1, player2, deck=None):
        player1_name, player1_player = player1
        player2_name, player2_player = player2
        game = Tablic(deck)
        while not game.is_terminal:
            # Player 1 turn
            card, take = player1_player.play_turn(game)
            self.analyze_play(player1_name, card, take, game)
            game.play_card(card, take)
            # Player 2 turn
            card, take = player2_player.play_turn(game)
            self.analyze_play(player2_name, card, take, game)
            game.play_card(card, take)
        return game.rewards


def getPlayer(name, device = "cuda"):
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
    BATTLES = 1000

    players = [
        # (921, getPlayer("9/g99e100000")),
        # (922, getPlayer("9/g99e200000")),
        # (923, getPlayer("9/g99e300000")),
        # (924, getPlayer("9/g99e400000")),
        # (925, getPlayer("9/g99e500000")),
        # (926, getPlayer("9/g99e600000")),
        # (927, getPlayer("9/g99e700000")),
        # (928, getPlayer("9/g99e800000")),
        (929, getPlayer("9/g99e900000")),
        (92, getPlayer("9/g99e1000000")),
        #(63, getPlayer("6-MyBoy/g99e500000-ZS")),
        #(73, getPlayer("7-JamesBond/g99e500000-ZS", "cuda")),
        #(-3, getPlayer("g99e100000-ZS")),
        #(1, getPlayer("5-Two/Lg99e100000-ZS"))
        #(2, getPlayer("Lg99e100000-ZS")),
        #(0, getPlayer("G")),
        #(-1, getPlayer("H")),
        (0, getPlayer("G")),
       # (100, getPlayer("SG")),
    ]
    
    tablicAnalysis = TablicPlayerAnalysis(players)
    results = tablicAnalysis.analyze_players(BATTLES)
    for i in results:
        results[i].print_player_stats()
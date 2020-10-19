from tablic_player import TablicPlayer
from tablic import Tablic
from DQNAgent import DQNAgent
import random
import numpy as np
import torch

class ReinforcedTablicPlayer(TablicPlayer):
    def __init__(self, gamma):
        self.agent = DQNAgent(gamma)
    
    def load_model(self, model_path):
        self.agent = torch.load(model_path)
    
    def save_model(self, model_path):
        torch.save(self.agent, model_path)

    @classmethod
    def take_to_state_action(cls, state_vector, played_card, take):
        take_vector = Tablic.get_take_vector(played_card, take)
        result = np.concatenate((take_vector, state_vector)) 
        return torch.from_numpy(result).type(torch.cuda.FloatTensor)

    @classmethod
    def get_valid_state_actions(cls, game):
        hand = game.get_hand(game.current_player)
        observation = game.get_observation_vector(game.current_player)
        valid_takes = list(Tablic.get_valid_takes(game.table, hand))
        valid_state_actions = torch.zeros([len(valid_takes), 80]).type(torch.cuda.FloatTensor)
        for ind, (played_card, take) in enumerate(valid_takes):
            valid_state_actions[ind] = cls.take_to_state_action(observation, played_card, take)
        return valid_takes, valid_state_actions

    def find_best_play_from_state_actions(self, takes, state_actions):
        with torch.no_grad():
            takes_value = self.agent.forward(state_actions)
        best_take_ind = torch.argmax(takes_value)
        return takes[best_take_ind]

    def get_random_play_from_state_actions(self, valid_takes, valid_state_actions):
        return random.choice(valid_takes)

    def find_best_play(self, game):
        return self.find_best_play_from_state_actions(
                            *self.get_valid_state_actions(game))

    def get_random_play(self, game):
        return random.choice(list(Tablic.get_all_valid_takes(game.table, game.get_hand(game.current_player))))

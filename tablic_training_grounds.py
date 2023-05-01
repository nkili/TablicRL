import random
import time
import torch
from tablic import Tablic
from reinforced_tablic_player import ReinforcedTablicPlayer
from DQN import DQN, DQN_Dropout, DQN_Long, DQN_LongDropout

# Number of episodes
EPISODES = 1000000
SAVE_FREQ = 100000
NOTIFY_FREQ = 5000

# Update frequency
UPDATE_FREQ = 25
# Switch nets frequency
SWITCH_FREQ = 250

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / EPISODES

# Gamma
GAMMA = 0.99

# Zero sum model
IS_ZERO_SUM = True
IS_NOISY_PLAY = True

def getModelName(gamma, isZeroSum, isNoisyPlay, networkName, episode):
    suffix = "-ZS" if isZeroSum else ""
    suffix += "-NP" if isNoisyPlay else ""
    return f"Models/{networkName}g{int(gamma*100)}e{episode}{suffix}"

def trainAModel(gamma, isZeroSum, isNoisyPlay, network):
    player = ReinforcedTablicPlayer(gamma, isZeroSum, network[1])
    epsilon = EPSILON_START
    start = time.time()
    for episode in range(1, EPISODES+1):
        game = Tablic()
        game_actions = [[],[]]
        game_rewards = [[],[]]
        game_valid_actions = [[],[]]

        if (episode > SWITCH_FREQ and episode % SWITCH_FREQ == 0):
            player.agent.update_target_model()

        while not game.is_terminal:
            current_player = game.current_player

            all_takes, all_state_actions = player.get_valid_state_actions(game)
            game_valid_actions[current_player].append(all_state_actions)
            game_rewards[current_player].append(game.rewards[current_player])

            if random.random() < epsilon:
                played_card, played_take = player.get_random_play_from_state_actions(all_takes, all_state_actions)
            elif isNoisyPlay:
                played_card, played_take = player.find_best_play_from_state_actions_noisy(all_takes, all_state_actions)
            else:
                played_card, played_take = player.find_best_play_from_state_actions(all_takes, all_state_actions)

            observation_vector = game.get_observation_vector(current_player)
            state_action = player.take_to_state_action(observation_vector, played_card, played_take)
            game.play_card(played_card, played_take)
            game_actions[current_player].append(state_action)

        for current_player in range(2):
            game_rewards[current_player].append(game.rewards[current_player])
            game_valid_actions[current_player].append(None)
            for i in range(len(game_actions[current_player])):
                action = game_actions[current_player][i]
                reward = game_rewards[current_player][i+1] - game_rewards[current_player][i]
                valid_actions = game_valid_actions[current_player][i+1]
                assert(game.is_terminal)
                if player.agent.isZeroSum: reward = 0 if valid_actions != None else game.rewards[current_player]- game.rewards[1-current_player]
                player.agent.remember(action, reward, valid_actions)
        
        if episode % NOTIFY_FREQ == NOTIFY_FREQ / 2:
            print(f"Episode {episode} completed after {int(time.time() - start)} seconds.")
        if episode % UPDATE_FREQ == 0:
            player.agent.backward()
        if episode % SAVE_FREQ == 0 or episode == EPISODES:
            modelName = getModelName(gamma, isZeroSum, isNoisyPlay, network[0], episode)
            print(f"Model {modelName} saved after {int(time.time() - start)} seconds.")
            print()
            player.save_model(modelName)
        epsilon = max(epsilon - EPSILON_DECAY, EPSILON_END)
    end = time.time()
    print(f"Training with GAMMA={gamma} lasted {int(end-start)} seconds.")



if __name__ == '__main__':
    for gamma in [0.99]:
        for isZeroSum in [False]:
            for isNoisyPlay in [False]:
                for network in [("", DQN)]:
                    print(f"Training model with GAMMA = {gamma}, IS_ZERO_SUM={isZeroSum}, IS_NOISY_PLAY={isNoisyPlay}, network={network[0]}")
                    trainAModel(gamma, isZeroSum, isNoisyPlay, network)

# for network in [("", DQN), ("D", DQN_Dropout), ("L", DQN_Long), ("LD", DQN_LongDropout)]:
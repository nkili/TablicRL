import random
import time
import torch
from tablic import Tablic
from reinforced_tablic_player import ReinforcedTablicPlayer

# Number of episodes
EPISODES = 50000
SAVE_FREQ = 5000

# Update frequency
UPDATE_FREQ = 5
# Switch nets frequency
SWITCH_FREQ = 50

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / EPISODES

# Gamma
GAMMA = 0

if __name__ == '__main__':
    player = ReinforcedTablicPlayer(GAMMA)
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
                player.agent.remember(action, reward, valid_actions)
        
        if episode % UPDATE_FREQ == 0:
            player.agent.backward()
        if episode % SAVE_FREQ == 0:
            print(f"Episode {episode} saved.")
            player.save_model(f"Models/g{int(GAMMA*100)}e{episode}")
        epsilon = max(epsilon - EPSILON_DECAY, EPSILON_END)
    player.save_model(f"Models/g{int(GAMMA*100)}e{episode}")
    end = time.time()
    print(f"Training with GAMMA={GAMMA} lasted {end-start} seconds.")
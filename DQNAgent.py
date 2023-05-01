import random
import time
import torch
import torch.nn as nn
from DQN import DQN
from collections import deque

class DQNAgent():
    def __init__(self, gamma=0, isZeroSum=False, memory_len=2048*48, batch_size=2048,
                 learning_rate=0.001, l2=1e-5, device="cuda", dqnClass=DQN):
        self.device = torch.device(device)
        self.target = dqnClass().to(self.device)
        self.current = dqnClass().to(self.device)

        self.isZeroSum = isZeroSum
        self.gamma = gamma
        self.memory = deque(maxlen = memory_len)
        self.batch_size = batch_size
        self.since_last_backward = 0

        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.current.parameters(),
                                            lr=learning_rate, weight_decay=l2)

    def forward(self, x):
        return self.current(x.to(self.device))

    def update_target_model(self):
        self.target.load_state_dict(self.current.state_dict())

    def remember(self, state_action, reward, valid_actions):
        self.since_last_backward += 1
        if (valid_actions == None):
            self.memory.append((state_action, reward, valid_actions))
        else:
            self.memory.append((state_action, reward, valid_actions.to(self.device)))

    def backward(self):
        batch_len = min(self.batch_size, len(self.memory))
        batch = list(self.memory)[(-self.since_last_backward):-1] + random.sample(self.memory, batch_len-self.since_last_backward)
        batch_nn_input = torch.zeros([batch_len, 80]).to(self.device)
        for ind, (state_action, _, _) in enumerate(batch):
            batch_nn_input[ind] = state_action
        q_values = self.current.forward(batch_nn_input)
        with torch.no_grad():
            q_values_target = q_values.clone().detach()
            for ind, (_, reward, valid_actions) in enumerate(batch):
                if valid_actions == None:
                    q_values_target[ind] = reward
                else:
                    q_values_target[ind] = reward + self.gamma * torch.max(self.target(valid_actions))
        loss = self.loss_fn(q_values, q_values_target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.since_last_backward = 0
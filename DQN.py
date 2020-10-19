import torch
import torch.nn as nn

device = torch.device("cuda")

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        input_size = 80
        output_size = 1
        self.neuralnet = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2, input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2, output_size))

    def forward(self, x):
        return self.neuralnet(x)

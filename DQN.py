import torch
import torch.nn as nn

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
            nn.Linear(input_size*2, input_size*2),
            nn.ReLU(),
            nn.Linear(input_size*2, output_size))

    def forward(self, x):
        return self.neuralnet(x)

class DQN_Dropout(nn.Module):
    def __init__(self):
        super(DQN_Dropout, self).__init__()
        input_size = 80
        output_size = 1
        self.neuralnet = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size*2, input_size*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size*2, output_size))

    def forward(self, x):
        return self.neuralnet(x)

class DQN_Long(nn.Module):
    def __init__(self):
        super(DQN_Long, self).__init__()
        input_size = 80
        output_size = 1
        self.neuralnet = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.neuralnet(x)

class DQN_LongDropout(nn.Module):
    def __init__(self):
        super(DQN_LongDropout, self).__init__()
        input_size = 80
        output_size = 1
        self.neuralnet = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.neuralnet(x)

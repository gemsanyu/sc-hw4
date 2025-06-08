import torch as T
from torch.nn import Module, Sequential, Linear, ReLU, Tanh

class Policy(Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.layer = Sequential(Linear(num_inputs, 64),
                                ReLU(),
                                Linear(64, 32),
                                ReLU(),
                                Linear(32, 16),
                                ReLU(),
                                Linear(16, 8),
                                ReLU(),
                                Linear(8, 4),
                                ReLU(),
                                Linear(4, 2),
                                Tanh())
    
    def forward(self, x):
        out = self.layer(x)
        return out
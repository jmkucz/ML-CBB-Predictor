"""
Modified model from University of Michigan EECS 445 material
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(70, 28)
        self.fc2 = nn.Linear(28, 14)

        self.init_weights()

    def init_weights(self):
        for fc in [self.fc1, self.fc2]:
            f_size = fc.in_features
            nn.init.normal_(fc.weight, 0.0, 1 / f_size)
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        print(x.shape)
        N, H = x.shape
        w = N * H
        x = x.view(-1, w)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

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

        

        self.init_weights()

    def init_weights(self):
        


    def forward(self, x):
        N, C, H, W = x.shape



        return z

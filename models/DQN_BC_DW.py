import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from utils.train import ReplayBuffer, RLTrainer
from config import device
import matplotlib.pyplot as plt

class DQN_BC_DW(nn.Module):
    def __init__(self, state_size, action_range):
        super(DQN_BC_DW, self).__init__()
        self.action_dim = action_range
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim * self.action_dim)  # Output for all bid-ask combinations
        )
    
    def forward(self, state):
        return self.network(state)
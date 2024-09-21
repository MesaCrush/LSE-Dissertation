import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class Actor(nn.Module):
    def __init__(self, state_size, action_range):
        super(Actor, self).__init__()
        self.action_dim = action_range
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.bid_head = nn.Linear(64, self.action_dim)
        self.ask_head = nn.Linear(64, self.action_dim)
    
    def forward(self, state):
        shared_features = self.shared(state)
        bid_logits = self.bid_head(shared_features)
        ask_logits = self.ask_head(shared_features)
        return bid_logits, ask_logits

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from itertools import count


class Actor(nn.Module):
    def __init__(self, state_size, action_range):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_range = action_range
        self.action_dim = action_range * 2 + 1

        self.shared = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.bid_head = nn.Linear(64, self.action_dim)
        self.ask_head = nn.Linear(64, self.action_dim)
        
    def forward(self, state):
        shared_features = self.shared(state)
        bid_probs = F.softmax(self.bid_head(shared_features), dim=-1)
        ask_probs = F.softmax(self.ask_head(shared_features), dim=-1)
        return bid_probs, ask_probs

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value



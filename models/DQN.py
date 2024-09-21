import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_range):
        super(DQN, self).__init__()
        self.action_dim = action_range * 2 + 1
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim * self.action_dim)  # For all bid-ask combinations
        )
    
    def forward(self, state):
        return self.network(state)

def select_action(state, dqn_model, epsilon):
    if random.random() < epsilon:
        return random.randint(0, dqn_model.action_dim - 1), random.randint(0, dqn_model.action_dim - 1)
    else:
        with torch.no_grad():
            q_values = dqn_model(state).view(dqn_model.action_dim, dqn_model.action_dim)
            bid_action, ask_action = divmod(q_values.argmax().item(), dqn_model.action_dim)
            return bid_action, ask_action

# Training loop would involve:
# 1. Selecting actions using epsilon-greedy
# 2. Storing experiences in a replay buffer
# 3. Sampling from the replay buffer and updating the Q-network
# 4. Periodically updating a target network
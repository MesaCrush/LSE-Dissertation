import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from utils.evaluate import evaluate
from config import device
import numpy as np
from collections import deque
import random
from abc import ABC, abstractmethod

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLTrainer(ABC):
    def __init__(self, env, model, n_iters, eval_interval, save_path, lr=0.01):
        self.env = env
        self.model = model
        self.n_iters = n_iters
        self.eval_interval = eval_interval
        self.save_path = save_path
        self.lr = lr
        self.episode_rewards = []
        self.episode_pnls = []
        self.best_reward = float('-inf')

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update_model(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def post_episode_update(self, episode):
        pass

    def train(self):
        for episode in range(self.n_iters):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_pnl = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_model(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_pnl += info['step_pnl']
                state = next_state

            self.episode_rewards.append(episode_reward)
            self.episode_pnls.append(episode_pnl)
            self.post_episode_update(episode)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{self.n_iters}, Reward: {episode_reward:.2f}, PnL: {episode_pnl:.2f}")

            if (episode + 1) % self.eval_interval == 0:
                self.evaluate_and_save()

        self.plot_training_progress()
        return self.model

    def evaluate_and_save(self):
        eval_results = evaluate(self.env, self.model, num_episodes=10, is_rl_agent=True)
        mean_reward = eval_results['mean_pnl']
        
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.save_model()
            print(f"New best model saved with mean reward: {self.best_reward:.2f}")

    @abstractmethod
    def save_model(self):
        pass

    def plot_training_progress(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_pnls)
        plt.title('Episode PnLs')
        plt.xlabel('Episode')
        plt.ylabel('PnL')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
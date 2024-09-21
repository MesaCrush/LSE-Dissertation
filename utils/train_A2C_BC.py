import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from utils.train import RLTrainer
from config import device
from typing import Optional
from matplotlib import pyplot as plt

class A2C_BC_Trainer(RLTrainer):
    def __init__(self, env, actor, critic, n_iters, eval_interval, save_path, 
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, entropy_coef=0.01,
                 bc_weight=0.5, save_interval=100):
        super().__init__(env, (actor, critic), n_iters, eval_interval, save_path, lr_actor)
        self.actor, self.critic = actor, critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.bc_weight = bc_weight
        self.save_interval = save_interval
        
        self.actor_losses = []
        self.critic_losses = []
        self.bc_losses = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        bid_logits, ask_logits = self.actor(state_tensor)
        
        bid_probs = F.softmax(bid_logits, dim=-1)
        ask_probs = F.softmax(ask_logits, dim=-1)
        
        bid_dist = Categorical(bid_probs)
        ask_dist = Categorical(ask_probs)
        
        bid_action = bid_dist.sample().item()
        ask_action = ask_dist.sample().item()
        
        return bid_action, ask_action

    def update_model(self, state, bid_action, ask_action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Compute value and advantage
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        advantage = reward + (1 - done) * self.gamma * next_value.detach() - value
        
        # Compute actor loss
        bid_logits, ask_logits = self.actor(state_tensor)
        bid_probs = F.softmax(bid_logits, dim=-1)
        ask_probs = F.softmax(ask_logits, dim=-1)
        bid_log_prob = torch.log(bid_probs[0, bid_action] + 1e-8)
        ask_log_prob = torch.log(ask_probs[0, ask_action] + 1e-8)
        actor_loss = -(bid_log_prob + ask_log_prob) * advantage.detach()
        
        # Add entropy regularization
        entropy = -(bid_probs * torch.log(bid_probs + 1e-8)).sum() - (ask_probs * torch.log(ask_probs + 1e-8)).sum()
        actor_loss -= self.entropy_coef * entropy
        
        # Compute behavior cloning loss
        expert_action = self.env.generate_expert_action()
        expert_bid, expert_ask = expert_action
        bc_loss = F.cross_entropy(bid_logits, torch.tensor([expert_bid]).to(device)) 
                  #F.cross_entropy(ask_logits, torch.tensor([expert_ask]).to(device))
        
        
        # Combine actor loss and behavior cloning loss
        total_actor_loss = (1 - self.bc_weight) * actor_loss + self.bc_weight * bc_loss
        
        # Compute critic loss
        critic_loss = F.mse_loss(value, reward + (1 - done) * self.gamma * next_value.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_losses.append(total_actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.bc_losses.append(bc_loss.item())

    def train(self):
        for episode in range(self.n_iters):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_pnl = 0

            while not done:
                bid_action, ask_action = self.select_action(state)
                next_state, reward, done, info = self.env.step((bid_action, ask_action))
                self.update_model(state, bid_action, ask_action, reward, next_state, done)
                
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
        return self.actor, self.critic

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        
        # Compute advantage
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor)
        advantage = reward + (1 - done) * self.gamma * next_value.detach() - value
        
        # Compute actor loss
        action, log_probs = self.select_action(state)
        actor_loss = -(log_probs[0] + log_probs[1]) * advantage.detach()
        
        # Add entropy regularization
        bid_probs, ask_probs = self.actor(state_tensor)
        entropy = -(bid_probs * torch.log(bid_probs + 1e-8)).sum() - (ask_probs * torch.log(ask_probs + 1e-8)).sum()
        actor_loss -= self.entropy_coef * entropy
        
        # Compute behavior cloning loss
        expert_action = self.env.generate_expert_action()
        expert_action_tensor = torch.LongTensor(expert_action).to(device)
        bc_loss = F.cross_entropy(bid_probs, expert_action_tensor[0].unsqueeze(0)) + \
                  F.cross_entropy(ask_probs, expert_action_tensor[1].unsqueeze(0))
        
        # Combine actor loss and behavior cloning loss
        total_actor_loss = (1 - self.bc_weight) * actor_loss + self.bc_weight * bc_loss
        
        # Compute critic loss
        critic_loss = F.mse_loss(value, reward + (1 - done) * self.gamma * next_value.detach())
        
        # Update actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_losses.append(total_actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.bc_losses.append(bc_loss.item())

    def post_episode_update(self, episode):
        # Implement any post-episode logic here
        pass

    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, self.save_path)
        print(f"Model saved to {self.save_path}")

    def plot_training_progress(self):
        super().plot_training_progress()  # Call the parent class method
        
        # Add additional plots for actor, critic, and BC losses
        plt.figure(figsize=(12, 9))
        plt.subplot(3, 1, 1)
        plt.plot(self.actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.bc_losses)
        plt.title('Behavior Cloning Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('a2c_bc_losses.png')
        plt.close()
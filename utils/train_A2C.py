import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class ActorCriticTrainer:
    def __init__(self, env, actor, critic, n_iters, eval_interval, save_path, 
                 lr_actor=0.0003, lr_critic=0.001, gamma=0.99, entropy_coef=0.01,
                 max_grad_norm=0.5, gae_lambda=0.95):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.n_iters = n_iters
        self.eval_interval = eval_interval
        self.save_path = save_path
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        self.episode_rewards = []
        self.episode_pnls = []
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        bid_probs, ask_probs = self.actor(state_tensor)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        bid_probs = bid_probs + epsilon
        ask_probs = ask_probs + epsilon
        
        # Renormalize
        bid_probs = bid_probs / bid_probs.sum()
        ask_probs = ask_probs / ask_probs.sum()
        
        bid_dist = Categorical(bid_probs)
        ask_dist = Categorical(ask_probs)
        
        bid_action = bid_dist.sample()
        ask_action = ask_dist.sample()
        
        return [bid_action.item(), ask_action.item()], (bid_dist.log_prob(bid_action), ask_dist.log_prob(ask_action))

    def compute_gae(self, rewards, values, next_value, done):
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - done) - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - done)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        return returns

    def update_model(self, states, actions, log_probs, rewards, next_state, done):
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_state_tensor = torch.FloatTensor([next_state]).to(self.device)
        
        # Compute values
        values = self.critic(states_tensor).squeeze()
        next_value = self.critic(next_state_tensor).squeeze()
        
        # Compute returns and advantages
        returns = self.compute_gae(rewards, values.detach().cpu().numpy(), next_value.detach().cpu().numpy(), done)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute actor loss
        actor_loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            actor_loss -= sum(log_prob) * advantage.detach()
        
        # Add entropy regularization
        bid_probs, ask_probs = self.actor(states_tensor)
        entropy = -(bid_probs * torch.log(bid_probs + 1e-8)).sum(dim=1).mean() - (ask_probs * torch.log(ask_probs + 1e-8)).sum(dim=1).mean()
        actor_loss -= self.entropy_coef * entropy

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(values, returns)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_grad_norm)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_grad_norm)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train(self):
        for episode in range(self.n_iters):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_pnl = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            
            states, actions, log_probs, rewards = [], [], [], []

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                
                episode_reward += reward
                episode_pnl += info['step_pnl']
                state = next_state

                if len(states) >= 32 or done:  # Update every 32 steps or at episode end
                    actor_loss, critic_loss = self.update_model(states, actions, log_probs, rewards, next_state, done)
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                    states, actions, log_probs, rewards = [], [], [], []

            self.episode_rewards.append(episode_reward)
            self.episode_pnls.append(episode_pnl)
            self.actor_losses.append(episode_actor_loss)
            self.critic_losses.append(episode_critic_loss)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_pnl = np.mean(self.episode_pnls[-10:])
                avg_actor_loss = np.mean(self.actor_losses[-10:])
                avg_critic_loss = np.mean(self.critic_losses[-10:])
                print(f"Episode {episode+1}/{self.n_iters}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg PnL: {avg_pnl:.2f}, "
                      f"Avg Actor Loss: {avg_actor_loss:.4f}, "
                      f"Avg Critic Loss: {avg_critic_loss:.4f}")

            if (episode + 1) % self.eval_interval == 0:
                self.save_model()
                self.plot_training_progress()

        self.save_model()
        self.plot_training_progress()
        return self.actor, self.critic

    def save_model(self):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, self.save_path)
        print(f"Model saved to {self.save_path}")

    def plot_training_progress(self):
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))
        
        axs[0].plot(self.episode_rewards)
        axs[0].set_title('Episode Rewards')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Reward')
        
        axs[1].plot(self.episode_pnls)
        axs[1].set_title('Episode PnLs')
        axs[1].set_xlabel('Episode')
        axs[1].set_ylabel('PnL')
        
        axs[2].plot(self.actor_losses)
        axs[2].set_title('Actor Loss')
        axs[2].set_xlabel('Episode')
        axs[2].set_ylabel('Loss')
        
        axs[3].plot(self.critic_losses)
        axs[3].set_title('Critic Loss')
        axs[3].set_xlabel('Episode')
        axs[3].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('a2c_training_progress.png')
        plt.close()
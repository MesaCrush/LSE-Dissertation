import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.train import ReplayBuffer, RLTrainer
from config import device
import matplotlib.pyplot as plt
from typing import Optional

class DQNTrainer(RLTrainer):
    def __init__(self, env, model, n_iters, eval_interval, save_path, lr=0.001, batch_size=64, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.98, target_update=10,
                 patience=300, min_delta=0.001):
        super().__init__(env, model, n_iters, eval_interval, save_path, lr)
        self.target_model = type(model)(env.observation_space.shape[0], env.action_space.nvec[0]).to(device)
        self.target_model.load_state_dict(model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(100000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.no_improve_count = 0
        
        # Tracking metrics
        self.losses = []
        self.episode_rewards = []
        self.episode_pnls = []
        self.epsilon_values = []
        self.eval_rewards = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze()
                bid_action, ask_action = divmod(q_values.view(-1).argmax().item(), self.model.action_dim)
                return [bid_action, ask_action]

    def update_model(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) > self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(np.array(actions)).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            dones = torch.FloatTensor(dones).to(device)

            current_q_values = self.model(states)
            action_indices = actions[:, 0] * self.model.action_dim + actions[:, 1]
            current_q_values = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            loss = F.mse_loss(current_q_values, target_q_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())

    def post_episode_update(self, episode):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if episode % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'episode_rewards': self.episode_rewards,
            'episode_pnls': self.episode_pnls,
            'losses': self.losses,
            'epsilon_values': self.epsilon_values,
            'eval_rewards': self.eval_rewards,
        }, self.save_path)
        print(f"Model saved. Current episode: {len(self.episode_rewards)}")

    def plot_training_progress(self):
        plt.figure(figsize=(15, 20))
        
        plt.subplot(5, 1, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(5, 1, 2)
        plt.plot(self.episode_pnls)
        plt.title('Episode PnLs')
        plt.xlabel('Episode')
        plt.ylabel('PnL')
        
        plt.subplot(5, 1, 3)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.subplot(5, 1, 4)
        plt.plot(self.epsilon_values)
        plt.title('Epsilon Value')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        
        plt.tight_layout()
        plt.savefig('dqn_training_progress.png')
        plt.close()

    def train(self) -> Optional[torch.nn.Module]:
        for episode in range(self.n_iters):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_pnl = 0
            episode_loss = 0
            update_count = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update_model(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_pnl += info['step_pnl']
                episode_loss += self.losses[-1] if self.losses else 0
                update_count += 1
                state = next_state

            avg_episode_loss = episode_loss / update_count if update_count > 0 else float('inf')
            
            self.episode_rewards.append(episode_reward)
            self.episode_pnls.append(episode_pnl)
            self.epsilon_values.append(self.epsilon)
            self.post_episode_update(episode)

           
            self.save_model()
            

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{self.n_iters}, Reward: {episode_reward:.2f}, PnL: {episode_pnl:.2f}, "
                      f"Avg Loss: {avg_episode_loss:.4f}, Epsilon: {self.epsilon:.4f}")

            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.run_evaluation_episodes()
                self.eval_rewards.append(eval_reward)
                print(f"Evaluation Reward: {eval_reward:.2f}")
                self.save_model()

        print("Training completed. Final model saved.")
        self.plot_training_progress()
        return self.model

    def run_evaluation_episodes(self, num_episodes=5):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
        return total_reward / num_episodes

# Example usage
# env = YourEnvironment()
# state_size = env.observation_space.shape[0]
# action_range = env.action_space.nvec[0]
# model = YourModelClass(state_size, action_range).to(device)
# trainer = DQNTrainer(
#     env=env, 
#     model=model, 
#     n_iters=1000, 
#     eval_interval=100, 
#     save_path='dqn_model.pth',
#     patience=50,
#     min_delta=0.001
# )
# trained_model = trainer.train()
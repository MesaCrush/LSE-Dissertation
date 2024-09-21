import torch
import torch.nn.functional as F
import numpy as np
import random
from utils.train import ReplayBuffer, RLTrainer
from config import device
import matplotlib.pyplot as plt
from typing import Optional

class DQN_BC_Trainer(RLTrainer):
    def __init__(self, env, model, n_iters, eval_interval, save_path, lr=0.001, batch_size=64, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
                 bc_weight=0.5, save_interval=100):
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
        self.bc_weight = bc_weight
        self.losses = []
        self.q_losses = []
        self.bc_losses = []
        self.save_interval = save_interval
        self.current_episode = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.model(state_tensor).view(self.model.action_dim, self.model.action_dim)
                bid_action, ask_action = divmod(q_values.argmax().item(), self.model.action_dim)
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

            # Q-learning loss
            current_q_values = self.model(states).view(self.batch_size, self.model.action_dim, self.model.action_dim)
            action_indices = actions[:, 0] * self.model.action_dim + actions[:, 1]
            current_q_values = current_q_values.view(self.batch_size, -1).gather(1, action_indices.unsqueeze(1)).squeeze(1)

            next_q_values = self.target_model(next_states).view(self.batch_size, -1).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            q_loss = F.mse_loss(current_q_values, target_q_values)

            # Behavior cloning loss
            expert_actions = torch.LongTensor([self.env.generate_expert_action() for _ in range(self.batch_size)]).to(device)
            expert_actions_reshaped = expert_actions.view(-1)  # Reshape to [batch_size * 2]
    
            bc_loss = F.cross_entropy(self.model(states).view(self.batch_size * 2, -1), expert_actions_reshaped)

            # Combined loss
            loss = (1 - self.bc_weight) * q_loss + self.bc_weight * bc_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())
            self.q_losses.append(q_loss.item())
            self.bc_losses.append(bc_loss.item())

    def post_episode_update(self, episode):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if episode % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': self.current_episode,
            'epsilon': self.epsilon,
        }, self.save_path)
        print(f"Model saved at episode {self.current_episode}")

    def plot_training_progress(self):
        plt.figure(figsize=(12, 15))
        
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
        plt.title('Total Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.subplot(5, 1, 4)
        plt.plot(self.q_losses)
        plt.title('Q-Learning Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.subplot(5, 1, 5)
        plt.plot(self.bc_losses)
        plt.title('Behavior Cloning Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

    def train(self) -> Optional[torch.nn.Module]:
        for episode in range(self.n_iters):
            self.current_episode = episode
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
            self.post_episode_update(episode)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{self.n_iters}, Reward: {episode_reward:.2f}, PnL: {episode_pnl:.2f}, Avg Loss: {avg_episode_loss:.4f}")

            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.run_evaluation_episodes()
                print(f"Evaluation Reward: {eval_reward:.2f}")

            if (episode + 1) % self.save_interval == 0:
                self.save_model()

        # Save the final model at the end of training
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
# trainer = DQN_BC_Trainer(
#     env=env, 
#     model=model, 
#     n_iters=1000, 
#     eval_interval=100, 
#     save_path='dqn_bc_model.pth',
#     save_interval=100  # Save every 100 episodes
# )
# trained_model = trainer.train()
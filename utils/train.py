import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.evaluate import evaluate
from config import device

def train(env, actor, critic, n_iters, eval_interval, save_path='best_model.pth'):
    optimizerA = torch.optim.Adam(actor.parameters())
    optimizerC = torch.optim.Adam(critic.parameters())
    
    episode_rewards = []
    episode_pnls = []
    best_reward = float('-inf')
    
    for iter in range(n_iters):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_pnl = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).to(device)
            bid_probs, ask_probs = actor(state_tensor)
            value = critic(state_tensor)
            
            bid_dist = torch.distributions.Categorical(bid_probs)
            ask_dist = torch.distributions.Categorical(ask_probs)
            
            bid_action = bid_dist.sample()
            ask_action = ask_dist.sample()
            
            action = [bid_action.item(), ask_action.item()]
            next_state, reward, done, info = env.step(action)
            
            # Update episode metrics
            episode_reward += reward
            episode_pnl += info['step_pnl']
            
            # Compute advantage and update networks
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            next_value = critic(next_state_tensor).detach()
            advantage = torch.tensor(reward + (1 - done) * 0.99 * next_value.item() - value.item(), dtype=torch.float32, device=device)
            
            # Actor loss
            log_prob = bid_dist.log_prob(bid_action) + ask_dist.log_prob(ask_action)
            actor_loss = -log_prob * advantage.detach()
            
            # Critic loss
            target_value = torch.tensor([reward + (1 - done) * 0.99 * next_value.item()], dtype=torch.float32, device=device)
            critic_loss = torch.nn.functional.mse_loss(value, target_value)
            
            # Backpropagation
            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()
            
            state = next_state
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_pnls.append(episode_pnl)
        
        # Print progress
        if (iter + 1) % 10 == 0:
            print(f"Episode {iter+1}/{n_iters}, Reward: {episode_reward:.2f}, PnL: {episode_pnl:.2f}")
        
        # Evaluate and save best model
        if (iter + 1) % eval_interval == 0:
            eval_results = evaluate(env, actor, num_episodes=10, is_rl_agent=True)
            mean_reward = eval_results['mean_pnl']
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'optimizerA_state_dict': optimizerA.state_dict(),
                    'optimizerC_state_dict': optimizerC.state_dict(),
                }, save_path)
                print(f"New best model saved with mean reward: {best_reward:.2f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(episode_pnls)
    plt.title('Episode PnLs')
    plt.xlabel('Episode')
    plt.ylabel('PnL')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()
    
    return actor, critic

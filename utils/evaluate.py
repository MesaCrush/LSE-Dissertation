import torch
import numpy as np
from config import device, MAX_STEPS

def evaluate(env, agent, num_episodes=100, is_rl_agent=True, alg='dqn'):
    average_pnls = []
    episode_inventories = []
    episode_lengths = []
    total_trades = []
    sharpe_ratios = []
    
    visualize_episode = np.random.randint(0, num_episodes)
    episode_data = None
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_pnl = 0
        episode_inventory = []
        episode_returns = []
        num_trades = 0
        
        episode_actions = []
        episode_market_data = []
        
        for t in range(env.max_steps):
            if is_rl_agent:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    if alg == 'a2c':  # A2C model
                       
                        bid_probs, ask_probs = agent[0](state_tensor)
                        bid_action = bid_probs.argmax().item()
                        ask_action = ask_probs.argmax().item()
                    elif hasattr(agent, 'action_dim'):  # DQN model
                        q_values = agent(state_tensor).squeeze()
                        bid_action, ask_action = divmod(q_values.argmax().item(), agent.action_dim)
                    else:
                        raise ValueError("Unknown agent type")
                action = [bid_action, ask_action]
            else:
                action = agent.get_action(state, env)
            
            next_state, _, done, info = env.step(action)
            
            episode_pnl += info['step_pnl']
            episode_inventory.append(info['inventory'])
            episode_returns.append(info['step_pnl'])
            num_trades += info['filled_bid'] + info['filled_ask']
            
            if episode == visualize_episode:
                episode_actions.append(action)
                episode_market_data.append({
                    'step': t,
                    'mid_price': info['mid_price'],
                    'best_bid': info['best_bid'],
                    'best_ask': info['best_ask'],
                    'agent_bid_price': info['agent_bid_price'],
                    'agent_ask_price': info['agent_ask_price'],
                    'inventory': info['inventory'],
                    'step_pnl': info['step_pnl']
                })
            
            state = next_state
            
            if done:
                break
        
        steps = t + 1
        average_pnls.append(episode_pnl / steps)
        episode_inventories.append(np.mean(np.abs(episode_inventory)) / steps)
        episode_lengths.append(steps)
        total_trades.append(num_trades)
        
        if len(episode_returns) > 1:
            returns_mean = np.mean(episode_returns)
            returns_std = np.std(episode_returns)
            if returns_std != 0:
                sharpe_ratio = np.sqrt(252) * returns_mean / returns_std
                sharpe_ratios.append(sharpe_ratio)
        
        if episode == visualize_episode:
            episode_data = {
                'actions': episode_actions,
                'market_data': episode_market_data
            }
    
    results = {
        'mean_pnl': np.mean(average_pnls),
        'std_pnl': np.std(average_pnls),
        'mean_inventory': np.mean(episode_inventories),
        'std_inventory': np.std(episode_inventories),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_trades_per_episode': np.mean(total_trades),
        'mean_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
        'episode_data': episode_data
    }
    
    return results
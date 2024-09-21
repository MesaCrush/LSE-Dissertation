import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.market_env_historical import HistoricalMarketEnv
from config import HISTORICAL_DATA_PATH, ACTION_RANGE, QUANTITY_EACH_TRADE, MAX_STEPS

def run_episode(env, use_expert=False):
    state, info = env.reset()
    total_reward = 0
    total_pnl = 0
    episode_data = []

    for step in range(MAX_STEPS):
        if use_expert:
            action = info['expert_action']
        else:
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        total_pnl += info['step_pnl']

        # Calculate expert bid and ask prices
        expert_bid_price = info['mid_price'] - info['expert_action'][0] * info['spread'] * 0.2
        expert_ask_price = info['mid_price'] + info['expert_action'][1] * info['spread'] * 0.2

        episode_data.append({
            'step': step,
            'action': action,
            'reward': reward,
            'pnl': info['step_pnl'],
            'mid_price': info['mid_price'],
            'best_bid': info['best_bid'],
            'best_ask': info['best_ask'],
            'inventory': info['inventory'],
            'expert_bid_price': expert_bid_price,
            'expert_ask_price': expert_ask_price,
            'agent_bid_price': info['agent_bid_price'],
            'agent_ask_price': info['agent_ask_price']
        })

        if done:
            break

    return total_reward, total_pnl, episode_data

def compare_strategies(env, num_episodes=10):
    expert_rewards = []
    expert_pnls = []
    random_rewards = []
    random_pnls = []

    for episode in range(num_episodes):
        expert_reward, expert_pnl, _ = run_episode(env, use_expert=True)
        random_reward, random_pnl, _ = run_episode(env, use_expert=False)

        expert_rewards.append(expert_reward)
        expert_pnls.append(expert_pnl)
        random_rewards.append(random_reward)
        random_pnls.append(random_pnl)

        if episode % 10 == 0:
            print(f"Completed episode {episode}/{num_episodes}")

    return expert_rewards, expert_pnls, random_rewards, random_pnls

def plot_comparison(expert_rewards, expert_pnls, random_rewards, random_pnls):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Reward comparison
    ax1.hist(expert_rewards, alpha=0.5, label='Expert Strategy')
    ax1.hist(random_rewards, alpha=0.5, label='Random Strategy')
    ax1.set_xlabel('Total Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Total Rewards')
    ax1.legend()

    # PnL comparison
    ax2.hist(expert_pnls, alpha=0.5, label='Expert Strategy')
    ax2.hist(random_pnls, alpha=0.5, label='Random Strategy')
    ax2.set_xlabel('Total PnL')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Total PnL')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('strategy_comparison.png')
    plt.close()

    print("Comparison plot saved as 'strategy_comparison.png'")

def print_statistics(expert_rewards, expert_pnls, random_rewards, random_pnls):
    print("\nExpert Strategy Statistics:")
    print(f"Average Reward: {np.mean(expert_rewards):.2f} ± {np.std(expert_rewards):.2f}")
    print(f"Average PnL: {np.mean(expert_pnls):.2f} ± {np.std(expert_pnls):.2f}")

    print("\nRandom Strategy Statistics:")
    print(f"Average Reward: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    print(f"Average PnL: {np.mean(random_pnls):.2f} ± {np.std(random_pnls):.2f}")

def plot_single_episode(episode_data):
    steps = [data['step'] for data in episode_data]
    mid_prices = [data['mid_price'] for data in episode_data]
    best_bids = [data['best_bid'] for data in episode_data]
    best_asks = [data['best_ask'] for data in episode_data]
    expert_bid_prices = [data['expert_bid_price'] for data in episode_data]
    expert_ask_prices = [data['expert_ask_price'] for data in episode_data]
    inventories = [data['inventory'] for data in episode_data]
    rewards = [data['reward'] for data in episode_data]
    pnls = [data['pnl'] for data in episode_data]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    # Price plot
    ax1.plot(steps, mid_prices, label='Mid Price', color='green')
    ax1.plot(steps, best_bids, label='Best Bid', color='blue', alpha=0.5)
    ax1.plot(steps, best_asks, label='Best Ask', color='red', alpha=0.5)
    ax1.plot(steps, expert_bid_prices, label='Expert Bid', color='cyan', linestyle='--')
    ax1.plot(steps, expert_ask_prices, label='Expert Ask', color='magenta', linestyle='--')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title('Market Prices and Expert Actions')

    # Inventory plot
    ax2.plot(steps, inventories, label='Inventory', color='orange')
    ax2.set_ylabel('Inventory')
    ax2.legend()
    ax2.set_title('Inventory over Time')

    # Reward plot
    ax3.plot(steps, rewards, label='Reward', color='purple')
    ax3.set_ylabel('Reward')
    ax3.legend()
    ax3.set_title('Rewards over Time')

    # PnL plot
    ax4.plot(steps, pnls, label='PnL', color='brown')
    ax4.set_ylabel('PnL')
    ax4.set_xlabel('Step')
    ax4.legend()
    ax4.set_title('Profit and Loss over Time')

    plt.tight_layout()
    plt.savefig('single_episode_analysis.png')
    plt.close()

    print("Single episode analysis plot saved as 'single_episode_analysis.png'")

    # Print some statistics about the expert actions
    bid_distances = np.array(expert_bid_prices) - np.array(best_bids)
    ask_distances = np.array(best_asks) - np.array(expert_ask_prices)
    
    print("\nExpert Action Statistics:")
    print(f"Average distance of expert bid from best bid: {np.mean(bid_distances):.4f}")
    print(f"Average distance of expert ask from best ask: {np.mean(ask_distances):.4f}")
    print(f"Percentage of expert bids better than best bid: {np.mean(bid_distances > 0) * 100:.2f}%")
    print(f"Percentage of expert asks better than best ask: {np.mean(ask_distances > 0) * 100:.2f}%")

def test_historical_market_env():
    env = HistoricalMarketEnv(
        data_path=HISTORICAL_DATA_PATH,
        action_range=ACTION_RANGE,
        quantity_each_trade=QUANTITY_EACH_TRADE,
        max_steps=MAX_STEPS
    )

    print("Running comparison between expert and random strategies...")
    expert_rewards, expert_pnls, random_rewards, random_pnls = compare_strategies(env)

    print_statistics(expert_rewards, expert_pnls, random_rewards, random_pnls)
    plot_comparison(expert_rewards, expert_pnls, random_rewards, random_pnls)

    # Run a single episode with expert strategy for detailed analysis
    _, _, expert_episode_data = run_episode(env, use_expert=True)
    plot_single_episode(expert_episode_data)

if __name__ == "__main__":
    test_historical_market_env()
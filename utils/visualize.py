import matplotlib.pyplot as plt
import os

def visualize_episode_data(episode_data, agent_type):
    if not episode_data:
        print(f"No episode data available for {agent_type} agent.")
        return

    market_data = episode_data['market_data']
    actions = episode_data['actions']

    steps = [data['step'] for data in market_data]
    mid_prices = [data['mid_price'] for data in market_data]
    best_bids = [data['best_bid'] for data in market_data]
    best_asks = [data['best_ask'] for data in market_data]
    agent_bid_prices = [data['agent_bid_price'] for data in market_data]
    agent_ask_prices = [data['agent_ask_price'] for data in market_data]
    inventories = [data['inventory'] for data in market_data]
    pnls = [data['step_pnl'] for data in market_data]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    
    # Plot prices
    ax1.plot(steps, mid_prices, label='Mid Price', color='green', linewidth=2)
    ax1.plot(steps, best_bids, label='Market Best Bid', color='blue', linestyle='--', alpha=0.7)
    ax1.plot(steps, best_asks, label='Market Best Ask', color='red', linestyle='--', alpha=0.7)
    ax1.scatter(steps, agent_bid_prices, label='Agent Bid', color='cyan', marker='o', s=30)
    ax1.scatter(steps, agent_ask_prices, label='Agent Ask', color='magenta', marker='o', s=30)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.set_title(f'{agent_type} Agent: Market Prices and Agent Actions')
    
    # Plot inventory
    ax2.plot(steps, inventories, label='Inventory', color='purple')
    ax2.set_ylabel('Inventory')
    ax2.legend()
    
    # Plot step PnL
    ax3.bar(steps, pnls, label='Step PnL', color='orange', alpha=0.7)
    ax3.set_ylabel('Step PnL')
    ax3.set_xlabel('Step')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', f'{agent_type.lower()}_agent_episode.png'))
    plt.close()

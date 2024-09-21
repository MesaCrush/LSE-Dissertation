import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.market_env_simulated import MarketEnv

def test_market_env():
    # Initialize the environment
    env = MarketEnv(
        init_price=100.0,
        action_range=5,
        quantity_each_trade=10,
        max_steps=100
    )

    # Test reset
    state, info = env.reset()
    assert len(state) == 6, f"State should have 6 elements, but has {len(state)}"
    assert info['inventory'] == 0, f"Initial inventory should be 0, but is {info['inventory']}"
    assert info['mid_price'] == 100.0, f"Initial mid price should be 100.0, but is {info['mid_price']}"

    # Run for 100 steps
    for step in range(100):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)

        assert len(next_state) == 6, f"Next state should have 6 elements, but has {len(next_state)}"
        assert info['agent_bid_price'] <= info['agent_ask_price'], "Agent's bid price should be <= ask price"

        # Check reward
        assert isinstance(reward, (int, float)), f"Reward should be numeric, but is {type(reward)}"

        print(f"Step {step+1}/{100}, Reward: {reward:.2f}, PnL: {info['step_pnl']:.2f}")
        print(f"Best bid: {info['best_bid']}, Mid price: {info['mid_price']}, Best ask: {info['best_ask']}")


        assert info['best_bid'] <= info['mid_price'] <= info['best_ask'], "Price ordering is incorrect"
       
        # Check inventory
        assert isinstance(info['inventory'], (int, float)), f"Inventory should be numeric, but is {type(info['inventory'])}"

        # Check PnL
        assert 'step_pnl' in info, "step_pnl should be in info dictionary"
        assert 'realized_pnl' in info, "realized_pnl should be in info dictionary"

        # Check best bid and ask in state
        assert next_state[4] == info['best_bid'], "Best bid in state doesn't match info"
        assert next_state[5] == info['best_ask'], "Best ask in state doesn't match info"

        if done:
            assert step == 99, f"Environment ended at step {step}, expected 99"
            break

    print("All basic checks passed!")

    # Plotting
    plot_market_data(env)

def plot_market_data(env):
    # Run another episode to collect data for plotting
    state, _ = env.reset()
    data = []
    for _ in range(100):
        action = env.action_space.sample()
        next_state, _, done, info = env.step(action)
        data.append((next_state, info))
        if done:
            break

    # Extract data
    timesteps = range(len(data))
    inventories = [state[0] for state, _ in data]
    order_imbalances = [state[1] for state, _ in data]
    spreads = [state[2] for state, _ in data]
    mid_price_moves = [state[3] for state, _ in data]
    best_bids = [state[4] for state, _ in data]
    best_asks = [state[5] for state, _ in data]
    mid_prices = [(bid + ask) / 2 for bid, ask in zip(best_bids, best_asks)]
    agent_bid_prices = [info['agent_bid_price'] for _, info in data]
    agent_ask_prices = [info['agent_ask_price'] for _, info in data]

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

    # Price plot
    ax1.plot(timesteps, mid_prices, label='Mid Price', color='green', linewidth=2)
    ax1.plot(timesteps, best_bids, label='Best Bid', color='blue', alpha=0.5)
    ax1.plot(timesteps, best_asks, label='Best Ask', color='red', alpha=0.5)
    ax1.plot(timesteps, agent_bid_prices, label='Agent Bid', color='cyan', alpha=0.5, linestyle='--')
    ax1.plot(timesteps, agent_ask_prices, label='Agent Ask', color='magenta', alpha=0.5, linestyle='--')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Inventory plot
    ax2.plot(timesteps, inventories, label='Inventory', color='purple')
    ax2.set_ylabel('Inventory')
    ax2.legend()
    ax2.grid(True)

    # Spread and Order Imbalance plot
    ax3.plot(timesteps, spreads, label='Spread', color='orange')
    ax3.plot(timesteps, order_imbalances, label='Order Imbalance', color='brown')
    ax3.set_ylabel('Spread / Order Imbalance')
    ax3.set_xlabel('Timestep')
    ax3.legend()
    ax3.grid(True)

    plt.suptitle('Market Environment Simulation')
    plt.tight_layout()
    plt.savefig('market_simulation_test.png')
    plt.close()

    print("Plot saved as 'market_simulation_test.png'")

if __name__ == "__main__":
    test_market_env()
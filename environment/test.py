
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from environment.market_env import MarketEnv

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
    assert len(state) == 4, f"State should have 4 elements, but has {len(state)}"
    assert info['inventory'] == 0, f"Initial inventory should be 0, but is {info['inventory']}"
    assert info['mid_price'] == 100.0, f"Initial mid price should be 100.0, but is {info['mid_price']}"

    # Run for 100 steps
    for step in range(100):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)

    
        #assert info['action'] == action.tolist(), f"Action in info {info['action']} doesn't match taken action {action}"
        assert info['agent_bid_price'] <= info['agent_ask_price'], "Agent's bid price should be <= ask price"

        # Check reward
        assert isinstance(reward, (int, float)), f"Reward should be numeric, but is {type(reward)}"

        assert info['best_bid'] <= info['mid_price'] <= info['best_ask'], "Price ordering is incorrect"

        # Check inventory
        assert isinstance(info['inventory'], (int, float)), f"Inventory should be numeric, but is {type(info['inventory'])}"

        # Check PnL
        assert 'step_pnl' in info, "step_pnl should be in info dictionary"
        assert 'realized_pnl' in info, "realized_pnl should be in info dictionary"

        if done:
            assert step == 99, f"Environment ended at step {step}, expected 99"
            break

    print("All basic checks passed!")

    # Plotting (optional, you can keep or remove this part)
    plot_market_data(env)

def plot_market_data(env):
    # Run another episode to collect data for plotting
    state, _ = env.reset()
    data = []
    for _ in range(100):
        action = env.action_space.sample()
        _, _, done,info = env.step(action)
        data.append(info)
        if done:
            break

    # Extract data
    timesteps = range(len(data))
    mid_prices = [info['mid_price'] for info in data]
    best_bids = [info['best_bid'] for info in data]
    best_asks = [info['best_ask'] for info in data]
    agent_bid_prices = [info['agent_bid_price'] for info in data]
    agent_ask_prices = [info['agent_ask_price'] for info in data]

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.plot(timesteps, mid_prices, label='Mid Price', color='green', linewidth=2)
    plt.plot(timesteps, best_bids, label='Best Bid', color='blue', alpha=0.5)
    plt.plot(timesteps, best_asks, label='Best Ask', color='red', alpha=0.5)
    plt.plot(timesteps, agent_bid_prices, label='Agent Bid', color='cyan', alpha=0.5, linestyle='--')
    plt.plot(timesteps, agent_ask_prices, label='Agent Ask', color='magenta', alpha=0.5, linestyle='--')

    plt.title('Market Environment Simulation')
    plt.xlabel('Timestep')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('market_simulation_test.png')
    plt.close()

    print("Plot saved as 'market_simulation_test.png'")

if __name__ == "__main__":
    test_market_env()

import numpy as np
from market_env import MarketEnv

def test_market_env():
    # Initialize the environment
    env = MarketEnv(
        init_price=100.0,
        action_range=5,
        quantity_each_trade=10,
        max_steps=100
    )

    # Reset the environment and get the initial state
    initial_state, _ = env.reset()
    print(f"Initial State: {initial_state}")

    # Run a few steps with random actions
    for step in range(5):
        action = env.action_space.sample()  # Random action
        new_state, reward, done, _, info = env.step(action)
        
        print(f"\nStep {step + 1}")
        print(f"Action taken: {action}")
        print(f"New State: {new_state}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

        if done:
            print("Episode finished early.")
            break

    # Print final exchange state
    print("\nFinal Exchange State:")
    print(f"Best Bid: {env.exchange.get_best_bid()}")
    print(f"Best Ask: {env.exchange.get_best_ask()}")
    print(f"Mid Price: {env.exchange.get_mid_price()}")
    print(f"Spread: {env.exchange.get_spread()}")
    bim_best, bim_rest = env.exchange.get_book_imbalance()
    print(f"Book Imbalance (Best): {bim_best}")
    print(f"Book Imbalance (Rest): {bim_rest}")

if __name__ == "__main__":
    test_market_env()
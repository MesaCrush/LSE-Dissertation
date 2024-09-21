import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import math

class SimplifiedMarketEnv(gym.Env):
    def __init__(self, init_price=100, action_range=10, quantity_each_trade=1, max_steps=1000):
        super(SimplifiedMarketEnv, self).__init__()
        self.init_price = init_price
        self.inventory = 0
        self.quantity_each_trade = quantity_each_trade
        self.max_steps = max_steps
        self.current_step = 0
        self.action_range = action_range
    
        self.action_space = MultiDiscrete([action_range, action_range])
       
        low = np.array([-100, -1, 0, -1, 0, 0], dtype=np.float32)
        high = np.array([100, 1, 10, 1, 200, 200], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        
    def step(self, action):
        bid_action, ask_action = action
        
        # Generate current price and next step's price
        current_price = self.generate_price(self.current_step)
        next_price = self.generate_price(self.current_step + 1)
        spread = 0.4  # Fixed spread for simplicity
        
        best_bid = current_price - spread / 2
        best_ask = current_price + spread / 2
    
        bid_price = round(next_price - spread / 2 ,2)
        ask_price = round(next_price + spread / 2, 2)

        # More lenient order execution
        filled_ask = self.quantity_each_trade if ask_price <=  next_price + spread / 2 else 0
        filled_bid = self.quantity_each_trade if bid_price >=  next_price - spread / 2 else 0

        # Update inventory
        self.inventory += filled_bid - filled_ask

        # Calculate reward (simple PnL)
        step_pnl = (ask_price - bid_price) * min(filled_ask, filled_bid)
        
        # Adjusted reward function
        reward = step_pnl - abs(self.inventory) * 0.01  # Reduced inventory penalty

        # Get new state
        new_state = self.get_state(current_price, spread, filled_bid, filled_ask)

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'current_price': current_price,
            'next_price': next_price,
            'spread': spread,
            'inventory': self.inventory,
            'filled_ask': filled_ask,
            'filled_bid': filled_bid,
            'step_pnl': step_pnl,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'agent_bid_price': bid_price,
            'agent_ask_price': ask_price,
        }
        
        # Generate expert action using next step's price
        expert_action = self.generate_expert_action(next_price, spread)
        
        # Calculate expert bid and ask prices
        expert_bid_price = round(next_price - spread / 2, 2)
        expert_ask_price = round(next_price + spread / 2, 2)

        # Add expert action and prices to info dictionary
        info['expert_action'] = expert_action
        info['expert_bid_price'] = expert_bid_price
        info['expert_ask_price'] = expert_ask_price

        return new_state, reward, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = 0
        self.current_step = 0
        
        initial_price = self.generate_price(0)
        initial_state = self.get_state(initial_price, 0.2, 0, 0)
        info = {
            'step_pnl': 0,
            'best_bid': initial_price - 0.1,
            'best_ask': initial_price + 0.1,
            'current_price': initial_price,
            'next_price': self.generate_price(1),
            'spread': 0.2,
            'inventory': 0,
            'filled_ask': 0,
            'filled_bid': 0,
            'agent_bid_price': initial_price - 0.1,
            'agent_ask_price': initial_price + 0.1,
        }
        return initial_state, info
            
    def get_state(self, current_price, spread, filled_bid, filled_ask):
        order_imbalance = np.sin(2 * math.pi * self.current_step / 100)  # Simulated order imbalance
        price_move = current_price - self.init_price
        
        return np.array([
            self.inventory,
            order_imbalance,
            spread,
            price_move,
            current_price - spread / 2,  # best_bid
            current_price + spread / 2   # best_ask
        ], dtype=np.float32)

    def generate_price(self, step):
        noise = np.random.normal(0, 0.1)
        price = self.init_price + 2 * math.sin(2 * math.pi * step / 100) + noise
        return round(price, 2)

    def generate_expert_action(self, next_price, spread):
        # The expert uses the next step's price to make decisions
        best_bid = next_price - spread / 2
        best_ask = next_price + spread / 2
        
        # Calculate the actions that would place orders at the prophet's best bid and ask
        bid_action = int((best_bid - (next_price - spread / 2)) / (spread * 0.1))
        ask_action = int((best_ask - (next_price + spread / 2)) / (spread * 0.1))
        
        # Ensure actions are within the valid range
        bid_action = max(0, min(bid_action, self.action_range - 1))
        ask_action = max(0, min(ask_action, self.action_range - 1))
        
        return np.array([bid_action, ask_action])
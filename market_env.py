import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from exchange import Exchange
from market_maker import Market_Maker
from market_order import Market_Orders
from rewardFuns import RewardFuns
from order import Order


class MarketEnv(gym.Env):
    def __init__(self, init_price, action_range, quantity_each_trade, max_steps):
        super(MarketEnv, self).__init__()
        self.init_price = init_price
        self.inventory = 0
        self.last_mid_price = init_price
        self.quantity_each_trade = quantity_each_trade
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space
        self.action_space = gym.spaces.Discrete(action_range * 2 + 1)  # Buy, sell, or do nothing
        
        # Observation space
        low = np.array([-10000, -1000000, 0, -100], dtype=np.float32)  # Adjusted for potential negative values
        high = np.array([10000, 1000000, 1000, 100], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        
        # Initialize other components
        self.exchange = Exchange(initial_mid=self.init_price, tick_size=0.01)
        self.mm = Market_Maker(exchange=self.exchange)
        self.mo = Market_Orders(exchange=self.exchange)
        self.rewardfuns = RewardFuns()
        
        # Episode tracking
        self.episode = 0
        self.action_records = []
        self.best_bid_records = []
        self.best_ask_records = []
        
    def step(self, action):
        # Update exchange state
        agent_ask_qty, agent_bid_qty = self.update_exchange_state(action=action)
            
        # Get new state
        new_state = self.get_state()
        
        # Calculate reward
        reward = self.rewardfuns.plain_reward(agent_bid_qty=agent_bid_qty, agent_ask_qty=agent_ask_qty)
      
        # Update inventory
        self.inventory += agent_bid_qty - agent_ask_qty
        
        # Increment step counter
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        info = {
            'agent_ask_qty': agent_ask_qty,
            'agent_bid_qty': agent_bid_qty,
            'inventory': self.inventory,
            'current_step': self.current_step
        }
        
        return new_state, reward, done, False, info
        
    def update_exchange_state(self, action):
        self.mm.new_simulated_limit_order_per_step()
        self.mo.new_simulated_market_order_per_step()

        # Keep quantity same, only adjusting ask price and bid price
        spread = self.exchange.get_spread()
        mid_price = self.exchange.get_mid_price()
        ask_price = mid_price + action * spread
        bid_price = mid_price - action * spread

        # Intercate with exchange
        ask_order = Order(quantity=self.quantity_each_trade, direction='sell', type="limit", price=ask_price)
        self.exchange.limit_order_placement(ask_order)
        bid_order = Order(quantity=self.quantity_each_trade, direction='sell', type="limit", price=bid_price)
        self.exchange.limit_order_placement(bid_order)
        agent_ask_qty, agent_bid_qty = self.exchange.check_agent_order_filled(ask_price=ask_price, ask_qty=self.quantity_each_trade, bid_price=bid_price, bid_qty=self.quantity_each_trade)
        return agent_ask_qty, agent_bid_qty
  
    def get_state(self):
        mid_price = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        order_imbalance, _ = self.exchange.get_book_imbalance()
        mid_price_move = mid_price - self.last_mid_price
        self.last_mid_price = mid_price
        
        return np.array([
            self.inventory,
            order_imbalance,
            spread,
            mid_price_move
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.exchange = Exchange(initial_mid=self.init_price, tick_size=0.01)
        self.mm = Market_Maker(exchange=self.exchange)
        self.mo = Market_Orders(exchange=self.exchange)
        
        self.inventory = 0
        self.last_mid_price = self.init_price
        self.current_step = 0
        self.episode += 1
        
        self.action_records = []
        self.best_bid_records = [self.exchange.get_best_bid()]
        self.best_ask_records = [self.exchange.get_best_ask()]
        
        initial_state = self.get_state()
        info = {}
        return initial_state, info
 

        





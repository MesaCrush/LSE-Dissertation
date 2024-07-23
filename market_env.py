import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from exchange import Exchange
from market_maker import Market_Maker
from market_order import Market_Orders
from reward import Rewards
from order import Order


class MarketEnv(gym.Env):
    def __init__(self, init_price, action_range, quantity_each_trade):
        super(MarketEnv, self).__init__()
        self.init_price = init_price
        self.inventory = 0
        self.last_mid_price = init_price
        self.quantity_each_trade = quantity_each_trade
        
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
        
        
        # Episode tracking
        self.episode = 0
        self.action_records = []
        self.best_bid_records = []
        self.best_ask_records = []
        self.overall_trade_price = 0
        
    def step(self, action):
        # Update exchange state
        agent_ask_qty, agent_bid_qty = self.update_exchange_state(action=action)
        
        # Get new state
        new_state = self.get_state()
        
        # Calculate reward
        reward = self.rewards.calculate_reward(trade_price, order_size, self.inventory)
        
        # Update episode info
        self.action_records.append(action)
        self.best_ask_records.append(self.exchange.get_best_ask())
        self.best_bid_records.append(self.exchange.get_best_bid())
        self.overall_trade_price += trade_price * order_size
        self.inventory += order_size
        
        # For now, we'll set done to False always, as we don't have a natural termination condition
        done = False
        
        info = {
            'trade_price': trade_price,
            'best_ask': self.exchange.get_best_ask(),
            'best_bid': self.exchange.get_best_bid(),
            'inventory': self.inventory
        }
        
        return new_state, reward, done, False, info

    def reset(self, seed=None, options=None):
        self.action_records = []
        self.exchange = Exchange(initial_mid=self.init_price, tick_size=0.01)
        self.best_ask_records = [self.exchange.get_best_ask()]
        self.best_bid_records = [self.exchange.get_best_bid()]
        self.mm = Market_Maker(exchange=self.exchange)
        self.mo = Market_Orders(exchange=self.exchange)
        self.remaining_quantity = self.target_quantity
        self.remaining_time = self.target_time
        self.episode += 1
        self.overall_trade_price = 0
       
        init_state = np.array([
            self.target_quantity,
            self.init_price,
            0,  # initial spread
            0,  # initial imbalance
            0,  # initial aggregated bim
            self.target_time,
        ], dtype=np.float32)

        info = {}
    
        return init_state, info
        

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
  
    def get_state(self, action):
        mid_price = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        imbalance_best, imbalance_rest = self.exchange.get_book_imbalance()
       
        
        observation = np.array([
            mid_price,
            spread,  # initial spread
            imbalance_best,  # initial imbalance
            imbalance_rest,  # initial aggregated bim
        ], dtype=np.float32)

        return observation

        





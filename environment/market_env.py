import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
from .exchange import Exchange
from .market_maker import Market_Maker
from .traders import Traders
from .rewardFuns import RewardFuns
from .order import Order


class MarketEnv(gym.Env):
    def __init__(self, init_price, action_range, quantity_each_trade, max_steps):
        super(MarketEnv, self).__init__()
        self.init_price = init_price
        self.inventory = 0
        self.last_mid_price = init_price
        self.quantity_each_trade = quantity_each_trade
        self.max_steps = max_steps
        self.current_step = 0
        self.action_range = action_range
        self.realized_pnl = 0
        
        self.action_space = MultiDiscrete([action_range, action_range])
       
        low = np.array([-10000, 0, 0, -100], dtype=np.float32)
        high = np.array([10000, 1, 1000, 100], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        
        self.exchange = Exchange(initial_mid=self.init_price, tick_size=0.01)
        self.mm = Market_Maker(self.exchange)
        self.mo = Traders(self.exchange)
        self.rewardfuns = RewardFuns()
        
    def step(self, action):
        # Update exchange state
        self.mm.new_simulated_limit_order_per_step()
        self.mo.new_simulated_market_order_per_step()

        spread = self.exchange.get_spread()
        mid_price = self.exchange.get_mid_price()
        bid_action, ask_action = action
        ask_price = mid_price + ask_action * spread * 0.2
        bid_price = mid_price - bid_action * spread * 0.2

        ask_order = Order(quantity=self.quantity_each_trade, direction='sell', type="limit", price=ask_price)
        self.exchange.limit_order_placement(ask_order)
        bid_order = Order(quantity=self.quantity_each_trade, direction='buy', type="limit", price=bid_price)
        self.exchange.limit_order_placement(bid_order)
        filled_ask, filled_bid = self.exchange.check_agent_order_filled(ask_price=ask_price, ask_qty=self.quantity_each_trade, bid_price=bid_price, bid_qty=self.quantity_each_trade)

        # Get new state
        new_state = self.get_state(filled_ask, filled_bid)
        mid_price_move = new_state[3]
        
        # Calculate reward
        reward = self.rewardfuns.reward_fun_with_inventory_penalty(
            agent_ask_qty=filled_ask, agent_ask_price=ask_price, 
            agent_bid_qty=filled_bid, agent_bid_price=bid_price, 
            agent_inventory=self.inventory, mid_price=self.last_mid_price, 
            mid_price_move=mid_price_move
        )
        
        # Calculate realized PnL for this step
        step_pnl = (filled_ask * ask_price) - (filled_bid * bid_price)
        self.realized_pnl += step_pnl

        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        info = {
            'action': action,
            'inventory': self.inventory,
            'current_step': self.current_step,
            'agent_bid_price': bid_price,
            'agent_ask_price': ask_price,
            'best_bid': self.exchange.get_best_bid(),
            'best_ask': self.exchange.get_best_ask(),
            'mid_price': self.exchange.get_mid_price(),
            'spread': spread,
            'reward': reward,
            'filled_ask': filled_ask,
            'filled_bid': filled_bid,
            'step_pnl': step_pnl,  # Add this line
            'realized_pnl': self.realized_pnl  # Add this line
        }
        
        return new_state, reward, done, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.exchange = Exchange(initial_mid=self.init_price, tick_size=0.01)
        self.mm = Market_Maker(self.exchange)
        self.mo = Traders(self.exchange)
        
        self.inventory = 0
        self.last_mid_price = self.init_price
        self.current_step = 0
        self.realized_pnl = 0
        
        initial_state = self.get_state()
        info = {
            'action': None,
            'inventory': self.inventory,
            'current_step': self.current_step,
            'agent_bid_price': None,
            'agent_ask_price': None,
            'best_bid': self.exchange.get_best_bid(),
            'best_ask': self.exchange.get_best_ask(),
            'mid_price': self.init_price,
            'spread': self.exchange.get_spread(),
            'reward': 0,
            'filled_ask': 0,
            'filled_bid': 0
        }
        return initial_state, info
            
    def get_state(self, agent_bid_qty=0, agent_ask_qty=0):
        mid_price = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        order_imbalance, _ = self.exchange.get_book_imbalance()
        mid_price_move = mid_price - self.last_mid_price
        self.last_mid_price = mid_price
        self.inventory += agent_bid_qty - agent_ask_qty
        
        
        return np.array([
            self.inventory,
            order_imbalance,
            spread,
            mid_price_move
        ], dtype=np.float32)


       





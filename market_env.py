import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
from exchange import Exchange  # Import your Exchange class
import numpy as np
from order import Order
from market_maker import Market_Maker
from market_order import Market_Orders
from reward import Rewards

class MarketEnv(gym.Env):
    def __init__(self, init_price, action_range, price_range, target_quantity, target_time, penalty_rate):
        super(MarketEnv, self).__init__()
        self.target_quantity = target_quantity
        self.target_time = target_time
        self.remaining_quantity = target_quantity
        self.remaining_time = target_time
        self.penalty_rate = penalty_rate
        self.init_price = init_price
        self.rewards = Rewards()
        self.overall_trade_price = 0
       
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = Discrete(action_range)  # Example for discrete actions

        # Define observation space
        low = np.array([-50, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([100000, price_range, price_range, 100, 100, 100000], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_records = []
        self.best_bid_records= []
        self.best_ask_records = []
        self.episode = 0
   
    def step(self, action):
        truncated = False
        # Interpret action and insert order
        trade_price  = self.update_exchange_state(action=action)
        new_state = self.get_state(action=action)
        self.action_records.append(action)
        self.best_ask_records.append(self.exchange.get_best_ask())
        self.best_bid_records.append(self.exchange.get_best_bid())
        self.overall_trade_price += trade_price * action
       
        if new_state[-1] < 3 and new_state[0] > 0:
            reward = -250
        # elif action == 0:
        #     reward = - self.rewards.calculate_dynamic_penalty(current_step=new_state[-1], total_steps=self.target_time,
        #                                                        remaining_quantity=new_state[0], 
        #                                                       penalty_rate=0.01, base_penalty=0.01, scale_type='linear')
        # else:
        #     reward = self.rewards.reward_for_market_impact(agent_avg_price=trade_price, 
        #                                                    initial_market_price=self.exchange.price_before_execution, 
        #                                                    zero_impact_reward=50, tolerance=0.05)
        
        else:
            reward = self.rewards.reward_based_obs(imbalance_best=new_state[3], action=action) + self.rewards.reward_for_market_impact(agent_avg_price=trade_price, 
                                                            initial_market_price=self.exchange.price_before_execution, 
                                                            zero_impact_reward=50, tolerance=0.05) 

                                                            


       
        done = new_state[-1] <= 0 or new_state[0] <= 0
        info = {'agent_trade_price':trade_price, 'best_ask':self.exchange.get_best_ask()}  # Additional info, if any
        #print(done)
        return new_state, reward, truncated, done, info
        # record bid ask in info

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
        # Interpret action and insert order
        if action == 0:
            trade_price = 0
            self.exchange.process_market_orders(agent_order_id=None)
            self.exchange.limit_order_cancellation()
        else:
            order_direction = 'buy' if action > 0 else 'sell'
            order = Order(quantity=abs(action), direction=order_direction, type="market", price=None)
            self.exchange.market_order_placement(order)
            trade_price = self.exchange.process_market_orders(agent_order_id=order.id)
            self.exchange.limit_order_cancellation()
        return trade_price
  
    def get_state(self, action):
        mid_price = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        imbalance_best, imbalance_rest = self.exchange.get_book_imbalance()
        self.remaining_quantity -= abs(action)
        self.remaining_time -= 1
        
        observation = np.array([
            self.remaining_quantity,
            mid_price,
            spread,  # initial spread
            imbalance_best,  # initial imbalance
            imbalance_rest,  # initial aggregated bim
            self.remaining_time,
        ], dtype=np.float32)

        return observation

    def render(self, mode='human', close=False):
        pass
     
    def close(self):
        # Cleanup
        pass
        





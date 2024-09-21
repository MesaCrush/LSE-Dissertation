import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
import numpy as np
import pandas as pd
from reward_funs import RewardFuns

class HistoricalMarketEnv(gym.Env):
    def __init__(self, data_path, action_range, quantity_each_trade, max_steps, start_option='random'):
        super(HistoricalMarketEnv, self).__init__()
        self.data = pd.read_csv(data_path)
        self.data['ts_recv'] = pd.to_datetime(self.data['ts_recv'])
        self.data['ts_event'] = pd.to_datetime(self.data['ts_event'])
        self.data = self.data.sort_values('ts_event')
      
        self.action_range = action_range
        self.quantity_each_trade = quantity_each_trade
        self.max_steps = max_steps
        self.start_option = start_option
        self.current_step = 0
        self.start_step = 0
        self.inventory = 0
        
        self.action_space = MultiDiscrete([action_range, action_range])
        
        low = np.array([-10000, -1, 0, -100, 0, 0], dtype=np.float32)
        high = np.array([10000, 1, 1000, 100, np.inf, np.inf], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        
        self.rewardfuns = RewardFuns()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.start_option == 'random':
            self.current_step = np.random.randint(0, len(self.data) - self.max_steps)
            self.start_step = self.current_step
        else:  # 'beginning'
            self.current_step = 0
        
        self.inventory = 0
        
        initial_state = self._get_current_state()
        info = {
            'action': None,
            'inventory': self.inventory,
            'current_step': self.current_step,
            'agent_bid_price': None,
            'agent_ask_price': None,
            'best_bid': initial_state['best_bid'],
            'best_ask': initial_state['best_ask'],
            'mid_price': (initial_state['best_bid'] + initial_state['best_ask']) / 2,
            'spread': initial_state['best_ask'] - initial_state['best_bid'],
            'reward': 0,
            'filled_ask': 0,
            'filled_bid': 0,
            'expert_action': [0,0]
        }
        return self._get_observation(initial_state), info
    
    def step(self, action):
        current_state = self._get_current_state()
        
        spread = current_state['best_ask'] - current_state['best_bid']
        mid_price = (current_state['best_ask'] + current_state['best_bid']) / 2
        bid_action, ask_action = action
        ask_price = round(mid_price + ask_action * spread * 0.1, 2)
        bid_price = round(mid_price - bid_action * spread * 0.1, 2)
            
        next_state = self._get_next_state()

        filled_ask = self.quantity_each_trade if ask_price < next_state['best_ask'] else 0
        filled_bid = self.quantity_each_trade if bid_price > next_state['best_bid'] else 0
        
        
        self.inventory += filled_bid - filled_ask
        
        observation = self._get_observation(next_state)
        
        mid_price_move = next_state['mid_price'] - current_state['mid_price']
        realized_step_pnl = (ask_price-bid_price) * min(filled_ask, filled_bid)
        
        reward = self.rewardfuns.reward_fun_with_inventory_penalty(agent_ask_price=ask_price, agent_bid_price=bid_price,
                                                                   filled_ask=filled_ask, filled_bid=filled_bid, 
                                                                   agent_inventory=self.inventory, mid_price=mid_price,
                                                                    mid_price_move=mid_price_move)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps + self.start_step or self.current_step >= len(self.data) - 1
        
        # Generate expert action for the next step
        expert_action = self.generate_expert_action()
        
        info = {
            'action': action,
            'inventory': self.inventory,
            'current_step': self.current_step,
            'agent_bid_price': bid_price,
            'agent_ask_price': ask_price,
            'best_bid': next_state['best_bid'],
            'best_ask': next_state['best_ask'],
            'mid_price': next_state['mid_price'],
            'spread': next_state['best_ask'] - next_state['best_bid'],
            'reward': reward,
            'filled_ask': filled_ask,
            'filled_bid': filled_bid,
            'step_pnl': realized_step_pnl,
            'realized_pnl': realized_step_pnl,
            'expert_action': expert_action
        }
        
        return observation, reward, done, info
    
    def _get_current_state(self):
        row = self.data.iloc[self.current_step]
        return {
            'best_bid': row['bid_px_00'],
            'best_ask': row['ask_px_00'],
            'bid_sizes': [row[f'bid_sz_{i:02d}'] for i in range(10)],
            'ask_sizes': [row[f'ask_sz_{i:02d}'] for i in range(10)],
            'bid_prices': [row[f'bid_px_{i:02d}'] for i in range(10)],
            'ask_prices': [row[f'ask_px_{i:02d}'] for i in range(10)],
            'mid_price': (row['bid_px_00'] + row['ask_px_00']) / 2,
            'timestamp': row['ts_event']
        }
    
    def _get_next_state(self):
        return self._get_current_state() if self.current_step + 1 >= len(self.data) else self._get_state_at(self.current_step + 1)
    
    def _get_state_at(self, index):
        row = self.data.iloc[index]
        return {
            'best_bid': row['bid_px_00'],
            'best_ask': row['ask_px_00'],
            'bid_sizes': [row[f'bid_sz_{i:02d}'] for i in range(10)],
            'ask_sizes': [row[f'ask_sz_{i:02d}'] for i in range(10)],
            'bid_prices': [row[f'bid_px_{i:02d}'] for i in range(10)],
            'ask_prices': [row[f'ask_px_{i:02d}'] for i in range(10)],
            'mid_price': (row['bid_px_00'] + row['ask_px_00']) / 2,
            'timestamp': row['ts_event']
        }
    
    def _get_observation(self, state):
        order_imbalance = (sum(state['bid_sizes']) - sum(state['ask_sizes'])) / (sum(state['bid_sizes']) + sum(state['ask_sizes']))
        spread = state['best_ask'] - state['best_bid']
        mid_price_move = state['mid_price'] - self._get_state_at(max(0, self.current_step - 1))['mid_price']
        
        return np.array([
            self.inventory,
            order_imbalance,
            spread,
            mid_price_move,
            state['best_bid'],
            state['best_ask']
        ], dtype=np.float32)

    def generate_expert_action(self):
        current_state = self._get_current_state()
        next_state = self._get_next_state()
        
        current_mid_price = (current_state['best_bid'] + current_state['best_ask']) / 2
        next_mid_price = (next_state['best_bid'] + next_state['best_ask']) / 2
        spread = next_state['best_ask'] - next_state['best_bid']
        
        # Calculate the actions that would place orders at half the spread away from the next mid price
        bid_action = round((spread/2 + next_mid_price - current_mid_price ) / (spread * 0.1))
        ask_action = round((spread/2 + next_mid_price - current_mid_price) / (spread * 0.1))

        # Ensure actions are within the valid range
        bid_action = 2
        ask_action = 2
        return np.array([bid_action, ask_action])

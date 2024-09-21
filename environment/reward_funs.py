import numpy as np

class RewardFuns:

    def __init__(self, max_inventory=100, spread_capture_weight=0.5, risk_aversion=0.1):
        self.max_inventory = max_inventory
        self.spread_capture_weight = spread_capture_weight
        self.risk_aversion = risk_aversion

    def reward_fun(self, agent_ask_qty, agent_ask_price, agent_bid_qty, agent_bid_price, agent_inventory, mid_price, mid_price_move):
        # Calculate the reward based on the agent's actions and inventory
        unrealized_pnl = agent_inventory * mid_price_move
        realized_pnl = agent_ask_qty * (agent_ask_price - mid_price) + agent_bid_qty * (mid_price - agent_bid_price)
        return unrealized_pnl + realized_pnl 

    def reward_fun_with_inventory_penalty(self, filled_ask, agent_ask_price, filled_bid, 
                                          agent_bid_price, agent_inventory, mid_price, mid_price_move, panalty_rate=0.005):
        # Calculate the reward based on the agent's actions and inventory
        unrealized_pnl = agent_inventory * mid_price_move
        realized_pnl = (agent_ask_price-agent_bid_price) * min(filled_ask, filled_bid)
        inventory_penalty = panalty_rate * abs(agent_inventory)

        return realized_pnl - inventory_penalty 

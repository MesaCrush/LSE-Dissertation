class RewardFuns:

    def __init__(self):
        pass

    def reward_fun(self, agent_ask_qty, agent_ask_price, agent_bid_qty, agent_bid_price, agent_inventory, mid_price, mid_price_move):
        # Calculate the reward based on the agent's actions and inventory
        unrealized_pnl = agent_inventory * mid_price_move
        realized_pnl = agent_ask_qty * (agent_ask_price - mid_price) + agent_bid_qty * (mid_price - agent_bid_price)
        return unrealized_pnl + realized_pnl

    def reward_fun_with_inventory_penalty(self, agent_ask_qty, agent_ask_price, agent_bid_qty, agent_bid_price, agent_inventory, mid_price, mid_price_move, panalty_rate=0.01):
        # Calculate the reward based on the agent's actions and inventory
        unrealized_pnl = agent_inventory * mid_price_move
        realized_pnl = agent_ask_qty * (agent_ask_price - mid_price) + agent_bid_qty * (mid_price - agent_bid_price)
        inventory_penalty = panalty_rate * abs(agent_inventory)
        print('unrealized_pnl:', unrealized_pnl)
        print('realized_pnl:', realized_pnl)
        print('inventory_penalty:', inventory_penalty)
        return unrealized_pnl + realized_pnl - inventory_penalty



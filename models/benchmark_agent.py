
import numpy as np

class BenchmarkAgent:
    def __init__(self, action_range):
        self.action_range = action_range

    def get_action(self, state, env):
        best_bid = env.exchange.get_best_bid()
        best_ask = env.exchange.get_best_ask()
        mid_price = env.exchange.get_mid_price()

        bid_action = 2
        ask_action = 2

        return [bid_action, ask_action]


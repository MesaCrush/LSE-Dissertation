
import numpy as np

class BenchmarkAgent:
    def __init__(self, action_range):
        self.action_range = action_range

    def get_action(self, state, env):
    

        bid_action = 4
        ask_action = 4

        return [bid_action, ask_action]


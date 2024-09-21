import numpy as np
import random
from .order import Order

class Traders:
    def __init__(self, exchange):
        self.exchange = exchange
        self.trend = 0
        self.trend_strength = 0.1

    def update_trend(self):
        # Simple trend following based on recent price movements
        recent_prices = self.exchange.prices[-10:]
        if len(recent_prices) > 1:
            self.trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            self.trend = 0

    def uninformed_trader(self):
        side = 'buy' if np.random.random() < 0.5 else 'sell'
        qty = np.random.randint(10, 100)
        # Slightly favor the trend direction
        if (self.trend > 0 and side == 'buy') or (self.trend < 0 and side == 'sell'):
            qty = int(qty * (1 + abs(self.trend) * self.trend_strength))
        order = Order(qty, side, "market")
        self.exchange.market_order_placement(order)

    def informed_trader(self):
        bim_best, bim_rest = self.exchange.get_book_imbalance()
        
        # Combine book imbalance and trend information
        prob_up = 0.5 + (bim_best - 0.5) * 0.4 + (bim_rest - 0.5) * 0.2 + self.trend * self.trend_strength
        
        side = 'buy' if random.random() < prob_up else 'sell'
        
        # Larger orders for informed traders
        qty = np.random.randint(50, 200)
        if (bim_best < 0.3 and side == 'buy') or (bim_best > 0.7 and side == 'sell'):
            qty = int(qty * 1.5)  # Increase size for potential arbitrage opportunities
        
        order = Order(quantity=qty, direction=side, type="market")
        self.exchange.market_order_placement(order)

    def new_simulated_market_order_per_step(self):
        self.update_trend()
        
        num_uninformed = np.random.poisson(3)  # Average 3 uninformed trades per step
        num_informed = np.random.poisson(1)    # Average 1 informed trade per step
        
        for _ in range(num_uninformed):
            self.uninformed_trader()
        
        for _ in range(num_informed):
            self.informed_trader()
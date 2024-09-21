import numpy as np
from .order import Order

class Market_Maker:
    def __init__(self, exchange):
        self.exchange = exchange
        self.tick_size = exchange.tick_size
        self.inventory = 0
        self.max_inventory = 1000
        self.risk_aversion = 0.01
        self.min_order_count = 10  # Minimum number of orders on each side

    def quote_prices(self):
        mid_price = self.exchange.get_mid_price()
        spread = max(self.exchange.get_spread(), self.tick_size * 2)
        bim_best, bim_rest = self.exchange.get_book_imbalance()

        # Adjust spread based on inventory risk
        inventory_risk = self.risk_aversion * self.inventory
        adjusted_spread = spread * (1 + abs(inventory_risk))

        # Adjust mid price based on order book imbalance
        imbalance_adjustment = (bim_best - 0.5) * self.tick_size * 2
        adjusted_mid = mid_price + imbalance_adjustment

        return adjusted_mid, adjusted_spread

    def generate_order_book(self):
        adjusted_mid, adjusted_spread = self.quote_prices()
        
        bid_orders = []
        ask_orders = []

        for i in range(1, self.min_order_count + 1):
            bid_price = round(adjusted_mid - adjusted_spread / 2 - i * self.tick_size, 2)
            ask_price = round(adjusted_mid + adjusted_spread / 2 + i * self.tick_size, 2)
            
            bid_qty = max(10, int(np.random.exponential(50)))
            ask_qty = max(10, int(np.random.exponential(50)))
            
            bid_orders.append(Order(quantity=bid_qty, direction='buy', type='limit', price=bid_price))
            ask_orders.append(Order(quantity=ask_qty, direction='sell', type='limit', price=ask_price))

        return bid_orders, ask_orders

    def new_simulated_limit_order_per_step(self):
        # Generate new order book
        bid_orders, ask_orders = self.generate_order_book()
        
        # Place all generated orders
        for order in bid_orders + ask_orders:
            self.exchange.limit_order_placement(order)
        
        # Ensure minimum number of orders on each side
        self.ensure_minimum_orders()

    def ensure_minimum_orders(self):
        bid_book = self.exchange.LOB['bid']
        ask_book = self.exchange.LOB['ask']
        
        if len(bid_book) < self.min_order_count:
            self.add_orders('buy', self.min_order_count - len(bid_book))
        
        if len(ask_book) < self.min_order_count:
            self.add_orders('sell', self.min_order_count - len(ask_book))

    def add_orders(self, direction, count):
        mid_price = self.exchange.get_mid_price()
        
        for i in range(count):
            if direction == 'buy':
                price = round(mid_price - (i + 1) * self.tick_size, 2)
            else:
                price = round(mid_price + (i + 1) * self.tick_size, 2)
            
            qty = max(10, int(np.random.exponential(50)))
            order = Order(quantity=qty, direction=direction, type='limit', price=price)
            self.exchange.limit_order_placement(order)

    def update_inventory(self, filled_bid, filled_ask):
        self.inventory += filled_bid - filled_ask
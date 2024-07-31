import numpy as np
import random
from .order import Order

class Market_Maker:
    """
    Simulate other market makers' behaviors. There are 2 types of market makers:
    1) Insiders (sophisticated traders): their quote prices are correlated with BIM
    2) Uninformed market makers: quote prices are randomized using geometric Brownian motion
    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.bid_book = exchange.LOB['bid']
        self.ask_book = exchange.LOB['ask']
        
    def uninformed_market_maker(self):
        mid = self.exchange.get_mid_price()
        half_spread = self.exchange.get_spread() / 2
 
        # Geometric Brownian Motion
        # drift = -0.01
        # vol = 0.05
        # dt = 1 
        # new_mid = mid * np.exp((drift - 0.5 * vol**2) * dt + vol * np.random.normal() * np.sqrt(dt))
        vol = 0.001  # Reduced volatility for more stable short-term movements
        dt = 1 
        price_change = mid * vol * np.random.normal() * np.sqrt(dt)
        new_mid = mid + price_change
        
        random_multiplier = np.random.randint(1, 3)
        bid_price = new_mid - half_spread * random_multiplier
        ask_price = new_mid + half_spread * random_multiplier
        
        # Ensure reasonable quotes
        best_bid = self.exchange.get_best_bid()
        best_ask = self.exchange.get_best_ask()
        bid_price = min(bid_price, best_ask - 0.01)
        ask_price = max(ask_price, best_bid + 0.01)
                
        bid_qty = np.random.randint(20, 80)
        ask_qty = np.random.randint(20, 80)
       
        self.exchange.limit_order_placement(Order(quantity=bid_qty, direction='buy', type='limit', price=round(bid_price, 2)))
        self.exchange.limit_order_placement(Order(quantity=ask_qty, direction='sell', type='limit', price=round(ask_price, 2)))

    def insider_market_maker(self):
        tick_size = self.exchange.tick_size
        mid = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        half_spread = spread / 2
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        
        # Probability of price moving up (simplified)
        prob_up = 0.5 + (bim_best - 0.5) * 0.6 + (bim_rest - 0.5) * 0.4
        
        # Calculate expected price move
        expected_move = (prob_up - 0.5) * 5 * tick_size
        new_mid = mid + expected_move

        # Adjust spread based on imbalance
        imbalance_factor = 1 + abs(bim_best - 0.5)
        adjusted_half_spread = half_spread * imbalance_factor
        
        # Set bid and ask prices
        bid_price = new_mid - adjusted_half_spread
        ask_price = new_mid + adjusted_half_spread
        
        # Determine order quantities
        base_qty = np.random.randint(20, 100)
        bid_qty = int(base_qty * (1 + (0.5 - prob_up)))
        ask_qty = int(base_qty * (1 + (prob_up - 0.5)))
        
        # Ensure reasonable quotes
        best_bid = self.exchange.get_best_bid()
        best_ask = self.exchange.get_best_ask()
        bid_price = min(bid_price, best_ask - tick_size)
        ask_price = max(ask_price, best_bid + tick_size)
        
        self.exchange.limit_order_placement(Order(quantity=bid_qty, direction='buy', type='limit', price=round(bid_price, 2)))
        self.exchange.limit_order_placement(Order(quantity=ask_qty, direction='sell', type='limit', price=round(ask_price, 2)))

    def new_simulated_limit_order_per_step(self):
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        bim_best_abs = abs(bim_best - 0.5)
        prob_uninformed = 0.5
        prob_insider = 0.5 * (1 + bim_best_abs * 2)  # Range: (0.5, 1)
        
        n_uninformed = np.random.binomial(10, prob_uninformed) 
        n_insider = np.random.binomial(20, prob_insider)  
                
        for _ in range(n_uninformed):
            self.uninformed_market_maker()

        for _ in range(n_insider):
            self.insider_market_maker()
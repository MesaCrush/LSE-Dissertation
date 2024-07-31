import numpy as np
import random
from .order import Order

class Traders:
    """
    Simulate market orders to take the quotes from the LOB.
    There are two kinds of traders:
    1) Insiders (sophisticated traders): place orders based on market opportunities (e.g., BIM)
    2) Uninformed traders: buy and sell with equal probability
    """
    def __init__(self, exchange):
        self.exchange = exchange
           
    def uninformed_trader(self):
        side = 'buy' if np.random.binomial(1, 0.5) == 1 else "sell"
        qty = np.random.randint(20, 50)      
        order = Order(qty, side, "market")
        if qty > 0:
            self.exchange.market_order_placement(order)
        
    def insider_trader(self):
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        
        # Probability of price going up: if bid_qty > ask_qty, prob_up > 0.5, also we want give bigger weight to bim_best
        prob_up = 0.5 + (bim_best - 0.5) * 0.8 + (bim_rest - 0.5) * 0.2
        print(f'prob_up: {prob_up}')
        
        side = 'buy' if prob_up > 0.5 else 'sell'
        
        qty = np.random.randint(20, 80)    
        order = Order(quantity=qty, direction=side, type="market")
        self.exchange.market_order_placement(order)

    def new_simulated_market_order_per_step(self):
        """
        Simulate market orders from both uninformed and insider traders.
        The number of orders placed depends on the book imbalance and available liquidity.
        """
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        bim_best_abs = abs(bim_best - 0.5)
        prob_uninformed = 0.5
        prob_insider = 0.5 * (1 + bim_best_abs * 2)  # Range: (0.5, 1)
        
        n_uninformed = np.random.binomial(10, prob_uninformed)
        n_insider = np.random.binomial(6, prob_insider) 
        
        # Restrict trading if there's a lack of liquidity
        order_num = self.exchange.get_distinct_limit_order_num() 
        if order_num < 20:
            n_uninformed = 0
            n_insider = 0
                  
        for _ in range(n_uninformed):
            self.uninformed_trader()

        for _ in range(n_insider):
            self.insider_trader()
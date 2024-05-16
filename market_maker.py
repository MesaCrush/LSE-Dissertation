
import numpy as np
import random
from order import Order


class Market_Maker():
    """
    simulate other market makers' behaviours, note they always quote both bid and ask in the market at the same time
    there are 2 types of market makers
    1) insiders (sophisticated traders): their quote price are correlated with BIM (first step, can add complexity, e.g. spread, traded volume)
       idea is: if there are more bid qty in the market, i.e. more buyer, we will expect the market moves up, 
       so the market maker will raise their quotes if bim > 0.5, and vice versa
       
    2) uninformed market maker: quote prices are randomized
       idea is the market makers' expectation of the fair price follows a geometric brownian motion, which is widely used to model financial prices 
       then, they quote bid and ask around their expected fair price, the spread is the current market spread * a random multiplier

    """
    def __init__(self, exchange):
        self.exchange = exchange
        self.bid_book = exchange.LOB['bid']
        self.ask_book = exchange.LOB['ask']
        
    def uninformed_market_maker(self):
        mid = self.exchange.get_mid_price()
        half_spread = (self.exchange.get_spread()) / 2
 
        # GBM: S(t)=S_0 * exp((drift − 0.5 * vol^2 )t + vol * W_t)
        # W_t is a standard Brownian motion.
        # dt is the time increment
        drift = -0.01
        vol = 0.05
        dt = 1 
        new_mid = mid * np.exp((drift - 0.5 * vol **2) * dt + vol * np.random.normal() * np.sqrt(dt))
        
        # symmetric bid and ask around the market maker's expectation of the fair price
        random_mutliplier = np.random.randint(1, 3)
        bid_price = new_mid - half_spread * random_mutliplier
        ask_price = new_mid + half_spread * random_mutliplier
        
        # make sure they are quoting reasonable quotes
        best_bid = list(self.exchange.LOB['bid'])[0]
        best_ask = list(self.exchange.LOB['ask'])[0]
        bid_price = min(bid_price, best_ask - 0.01)
        ask_price = max(ask_price, best_bid + 0.01)
                
        bid_qty = np.random.randint(1, 50)
        ask_qty = np.random.randint(1, 50)
       
        self.exchange.limit_order_placement(Order(quantity= bid_qty, direction='buy', type='limit', price=round(bid_price,2)),False)
        self.exchange.limit_order_placement(Order(quantity=ask_qty, direction='sell', type='limit', price=round(ask_price,2)),False)


    def insider_market_maker(self):
        """
        insiders use BIM to generate price trend signals
        if BIM > 0.5, expect an upward price movement, i.e. expect new mid rate is higher than the current mid
        The larger BIM, the higher probability of upward price movement, and larger magnitude of price increment
        This can be translate into the following:
        if no signal, Prob(up) = Prob(down) = 0.5
        Then, P(up) increases when BIM > 0.5, and decreases when BIM < 0.5
        i.e. P(up) = (1 + (BIM-0.5) * 2) * 0.5
        also, bim_best is a more important signal than bim_rest, so give bim_best a higher weight
        
        """
        tick_size = self.exchange.tick_size
        mid = self.exchange.get_mid_price()
        spread = self.exchange.get_spread()
        half_spread = spread / 2
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        
        # probability of raising price
        prob_up = (1 + (bim_best - 0.5) * 2 * 0.8 + (bim_rest - 0.5) * 2 * 0.2) * 0.5
        
        probability = random.random()
        if probability <= prob_up: 
            # market maker expect price to move up
            # the price increment is in the range of (0,10) ticks, and is correlated to the BIM
            new_mid = mid + (bim_best * 0.8 + bim_rest * 0.2) * 10 * tick_size
            random_mutliplier = np.random.randint(1, 3)
            bid_price = new_mid - half_spread * random_mutliplier
            ask_price = new_mid + half_spread * random_mutliplier 
            # if the bim signal is strong enough, then the market maker will skew the quote
            if (bim_best > 0.8) and (bim_best < 0.9):
                bid_price = bid_price + tick_size
            
            bid_qty = np.random.randint(15, 70)
            ask_qty = np.random.randint(30, 80)
                 
        else:
            # market maker expect price to move down
            # the price decrease is in the range of (0,5) ticks, and is correlated to the BIM
            new_mid = mid + ((bim_best - 0.5) * 0.8 + (bim_rest - 0.5) * 0.2) * 20 * tick_size
            random_mutliplier = np.random.randint(1, 3)
            bid_price = new_mid - half_spread * random_mutliplier
            ask_price = new_mid + half_spread * random_mutliplier 
            # if the bim signal is strong enough, then the market maker will skew the quote
            if (bim_best < 0.2) and (bim_best > 0.1):
                ask_price = ask_price - tick_size
            
            bid_qty = np.random.randint(30, 80)
            ask_qty = np.random.randint(15, 70)
        
        # make sure they are quoting reasonable quotes
        best_bid = list(self.exchange.LOB['bid'])[0]
        best_ask = list(self.exchange.LOB['ask'])[0]
        
        bid_price = min(bid_price, best_ask - 0.01)
        ask_price = max(ask_price, best_bid + 0.01)
        
        self.exchange.limit_order_placement(Order(quantity= bid_qty, direction='buy', type='limit', price=round(bid_price,2)),False)
        self.exchange.limit_order_placement(Order(quantity=ask_qty, direction='sell', type='limit', price=round(ask_price,2)),False)


    def new_simulated_limit_order_per_step(self):
        """
        suppose there are 5 market makers in the simulated environment
        Assume more sophisticated market makers
        they have different probability to place limit orders
        therefore, each time step, there could be at most 18 new quotes, and also can have 0 quotes
        
        For uninformed market makers, the probability of placing new quotes is 0.5
        For sophisticated market makers, the probability of placing new quotes also depends on bim
        the larger magnitude of (bim_best - 0.5), the stronger signal, the higher probability of placing orders
                """
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        bim_best = abs(bim_best - 0.5)
        prob_uninformed = 0.5
        prob_insider = 0.5 * (1 + bim_best * 2)  # in the range of (0.5 , 1)
        mid = self.exchange.get_mid_price()
        
        n_uninformed = np.random.binomial(6, prob_uninformed) 
        n_insider = np.random.binomial(12, prob_insider)  
                
        # in order to make sure there's enough liquidity in the LOB
        # if either bid or ask book has less than 10 levels, overwrite both number of new orders to be a large number
        order_num = self.exchange.get_distinct_limit_order_num() 
        if order_num[0] < 10 or order_num[1] < 10:
            n_uninformed = 8  
            n_insider = 17       
            
        if order_num[0] < 5 or order_num[1] < 5:
            n_uninformed = 10   
            n_insider = 20
        
        if bim_best >= 0.9:
            self.exchange.limit_order_placement(Order(quantity=np.random.randint(100,300), direction='sell', type='limit', price=round(mid,2)),False)

        if bim_best <= 0.1:
            self.exchange.limit_order_placement(Order(quantity=np.random.randint(100,300), direction='buy', type='limit', price=round(mid,2)),False)
        
        for _ in range(n_uninformed):
            self.uninformed_market_maker()

        for _ in range(n_insider):
            self.insider_market_maker()


import numpy as np
import random
from order import Order


class Market_Orders:
    """
    Simulate market orders to take the quotes from the LOB
    There are two kinds of traders:
    1) insiders (sophisticated traders): place orders when they see opportunities, e.g. BIM
       idea is if bim > 0.5, i.e. there are more buyers in the market, this indicates an upward price trend
       hence, the trader will decide to buy now, i.e. buy when price has not raised yet
       on the other side, if the bim < 0.5, the trader will decide to sell now
       and he will only trade when the bim reaches certain threshold (i.e. the signal is strong enough)
       
    2) uninformed traders: buy and sell at equal probability
    """
    def __init__(self, exchange):
        self.exchange = exchange
           
    def uninformed_trader(self):
        side = 'buy' if 1 == np.random.binomial(1, 0.5) else "sell"
        qty = np.random.randint(1, 100)      
        order =  Order(qty, side, "market")
        if qty > 0:
            # print(order)
            self.exchange.market_order_placement(order)
        
    def insider_trader(self):
        bim_best, bim_rest = self.exchange.get_book_imbalance()      
        
        # probability of price up
        # abs(bim_best - 0.5) is in the range of (0,0.5), so 2*abs(bim_best - 0.5) is within (0,1)
        # prob_up is in the range of (0,1) according to the below function
        prob_up = (1 + (bim_best - 0.5) * 2 * 0.8 + (bim_rest - 0.5) * 2 * 0.2) * 0.5
        
        probability = random.random()
        if probability <= prob_up: 
            # insider trader expects price to move up
            # will place a buy order
            side = 'buy'            
        else:
            # when the insider expects price to move down
            # will place a sell order
            side = 'sell'
        
        qty = np.random.randint(1,100)
        if (bim_best < 0.1) or (bim_best > 0.9):
                qty = np.random.randint(50, 150)   
        order =  Order(qty, side, "market")
        if qty > 0:
            self.exchange.market_order_placement(order)


    def new_simulated_market_order_per_step(self):
         """
         suppose there are 5 traders to place market orders in the simulated environment
         Assume more sophisticated traders
         they have different probability to place market orders
         therefore, each time step, there could be at most 5 new quotes, and also can have 0 quotes
         
         For uninformed traders, the probability of placing new quotes is 0.5
         For sophisticated traders, the probability of placing new quotes also depends on bim
         the larger magnitude of (bim_best - 0.5), the stronger signal, the higher probability of placing orders
         """
         bim_best, bim_rest = self.exchange.get_book_imbalance()      
         bim_best = abs(bim_best - 0.5)
         prob_uninformed = 0.5
         prob_insider = 0.5 * (1 + bim_best * 2)  # in the range of (0.5 , 1)
         
         n_uninformed = np.random.binomial(3, prob_uninformed)
         n_insider = np.random.binomial(6, prob_insider) 
         
         # if there is lack of liquidity in the market, then here we restrict the number of traders
         # this can avoid the issue that running out of the quotes
         order_num = self.exchange.get_distinct_limit_order_num() 
         if order_num[0] < 10 or order_num[1] < 10:
            n_uninformed = 0
            n_insider = 0
                  
         qty = np.random.randint(80,180)
         if bim_rest >= 0.9:
             self.exchange.market_order_placement(Order(qty,"sell","market"))
         if bim_rest <= 0.1:
             self.exchange.market_order_placement(Order(qty,"buy","market"))
    
                             
         for _ in range(n_uninformed):
             self.uninformed_trader()

         for _ in range(n_insider):
             self.insider_trader()
       
       




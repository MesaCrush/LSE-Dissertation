import random
import numpy as np
import itertools
from order import Order
from collections import deque

class Exchange:
    """    
    This class simulates the dynamics of the LOB
    - initialise a symmetric LOB by inputting initial_mid 
    - define tick_size which is minimum price movement
    - initialise 20-level LOB, each level is 1 tick_size away
    
    Then LOB will be updated by new Market orders and limit orders   
    """
    def __init__(self, initial_mid, tick_size, depth=30):
        self.prices = [initial_mid]
        self.initial_mid = initial_mid
        self.initial_bid = initial_mid - 3 * tick_size
        self.initial_ask = initial_mid + 3 * tick_size
        self.tick_size = tick_size
        self.LOB = {'bid': {}, 'ask': {}}
        Exchange.limit_order_queue = deque()
        Exchange.market_order_queue = deque()
        
        for level in range(0, depth):
            bid_qty =  np.random.randint(1, 100)
            ask_qty =  np.random.randint(1, 100)
            bid_price = round(self.initial_bid - level * tick_size, 2)
            ask_price = round(self.initial_ask + level * tick_size, 2)
            
            self.limit_order_queue.append(Order(quantity = bid_qty, direction = 'buy', type = 'limit', price = bid_price))
            self.limit_order_queue.append(Order(quantity = ask_qty, direction = 'sell', type = 'limit', price = ask_price))
        
    def market_order_placement(self, order):
        Exchange.market_order_queue.append(order)  
    
    def limit_order_placement(self, order):
        Exchange.limit_order_queue.append(order)  

    def limit_order_process(self):
        bid_book = self.LOB['bid']
        ask_book = self.LOB['ask']
        while self.limit_order_queue:
            order = self.limit_order_queue.popleft()
            if order.type == 'limit':
                book = bid_book if order.direction == 'buy' else ask_book
            if order.price in book:
                book[order.price] += order.qty
            else:
                book[order.price] = order.qty
      
     
        self.LOB['bid'] = dict(sorted(bid_book.items(), reverse=True))
        self.LOB['ask'] = dict(sorted(ask_book.items(), reverse=False))   

    # private method
    def __market_order_process(self):
        while Exchange.market_order_queue:
            order = Exchange.market_order_queue.popleft()
            if order.direction == 'buy':
                rest_qty = order.qty
                for price in self.LOB['ask']:
                    if rest_qty > self.LOB['ask'][price]:
                        rest_qty -= self.LOB['ask'][price]
                        self.LOB['ask'][price] = 0
                        if price == self.agent_ask_price:
                            self.agent_ask_qty = 0
                    else:
                        self.LOB['ask'][price] -= rest_qty
                        if price == self.agent_ask_price:
                            self.agent_ask_qty -= rest_qty
                        rest_qty = 0
                        break
            if order.direction == 'sell':
                rest_qty = order.qty
                for price in self.LOB['bid']:
                    if rest_qty > self.LOB['bid'][price]:
                        rest_qty -= self.LOB['bid'][price]
                        self.LOB['bid'][price] = 0
                        if price == self.agent_bid_price:
                            self.agent_bid_qty -= rest_qty 
                    else:
                        self.LOB['bid'][price] -= rest_qty
                        if price == self.agent_bid_price:
                            self.agent_bid_qty -= rest_qty 
                        rest_qty = 0 
                        break   
    
    def check_agent_order_filled(self, ask_price=0, ask_qty=0, bid_price=0, bid_qty=0):
        self.agent_ask_price = ask_price
        self.agent_ask_qty = ask_qty
        self.agent_bid_price = bid_price
        self.agent_bid_qty = bid_qty
        self.__market_order_process()
        return self.agent_ask_qty, self.agent_bid_qty
        
          
    def get_best_bid(self):
        return  list(self.LOB['bid'].keys())[0]
    
    def get_best_ask(self):
        return  list(self.LOB['ask'].keys())[0]
        
    def get_mid_price(self):
        # Ensure there is at least one buy order and one sell order
        if not self.LOB['bid'] or not self.LOB['ask']:
            print("Cannot calculate mid price without both buy and sell orders.")
            return self.initial_mid
    
        # Get the best bid (highest bid and best ask (lowest bid)
        best_bid = list(self.LOB['bid'].keys())[0]
        best_ask = list(self.LOB['ask'].keys())[0]
        
        # Calculate the mid price
        mid_price = (best_bid + best_ask) / 2.0

        return mid_price

    def get_spread(self):
        if not self.LOB['bid'] or not self.LOB['ask']:
            print("Spread can't be calculated if either side is empty")
            return 0 # Spread can't be calculated if either side is empty
        # Get the best bid (highest bid and best ask (lowest bid)
        best_bid = list(self.LOB['bid'].keys())[0]
        best_ask = list(self.LOB['ask'].keys())[0]

        return round(best_ask - best_bid,2)
    
    def get_book_imbalance(self):  
        """
        this function calculates the book imbalance (bim) of the current Limit Order Book
        bim = bid_qty / (bid_qty + ask_qty)
        in the financial market, the best bid offer bim is the most important signal
        also, we calculate the bim from the total LOB to understand the overview ratio of the buyers and sellers   
        """
        bid_book = self.LOB['bid']
        ask_book = self.LOB['ask']
        
        if not ( ask_book and bid_book ):
            print("Imbalance can't be calculated if either side is empty")
            return 0,0       
     
        best_bid_key = next(iter(bid_book), None)
        best_ask_key = next(iter(ask_book), None)
                
        best_bid_total_qty = sum(item[1] for item in bid_book[best_bid_key])
        best_ask_total_qty = sum(item[1] for item in ask_book[best_ask_key])

        total_best_qty = best_bid_total_qty + best_ask_total_qty
        bim_best = best_bid_total_qty / total_best_qty if total_best_qty else 0

        # there could be quite many levels of bid and ask quotes
        # the very deep quotes contains little signals, so only aggregate the qty for the best 10 quote price levels     
        total_bid_qty = 0
        total_ask_qty = 0 
        for price_list in itertools.islice(bid_book.keys(),10):
            total_bid_qty += (sum(item[1] for item in bid_book[price_list]))

        for price_list in itertools.islice(ask_book.keys(),10):
            total_ask_qty += (sum(item[1] for item in ask_book[price_list]))

        rest_bid_volume = total_bid_qty - best_bid_total_qty
        rest_ask_volume = total_ask_qty - best_ask_total_qty

        total_rest_qty = rest_ask_volume + rest_bid_volume
        bim_rest = rest_bid_volume / total_rest_qty if total_rest_qty else 0

        return bim_best, bim_rest

    def get_first_three_level_ask_qty(self):
        ask_book = self.LOB['ask']

        ask_total_qty_per_price = []
        
        for price in list(ask_book.keys())[0:3]:
            ask_total_qty_per_price.append(sum(item[1] for item in ask_book[price]))
        return ask_total_qty_per_price

    def calculate_realtime_vwap(self):
        """
        Update the VWAP with the new price and volume data from each trade.
        """
        if self.total_volume == 0:
            return 0  
        return self.total_pv / self.total_volume


def test():
    # Initialize the exchange with an initial mid price of 100, tick size of 1, and depth of 20
    exchange = Exchange(initial_mid=100, tick_size=0.01, depth=20)
    exchange.limit_order_process()
    print("After init:")
    print("Limit Order Book (Bids):", exchange.LOB['bid'])
    print("Limit Order Book (Asks):", exchange.LOB['ask'])


    # Place some limit orders
    exchange.limit_order_placement(Order(quantity=50, direction='buy', type='limit', price=99.95))
    exchange.limit_order_placement(Order(quantity=30, direction='sell', type='limit', price=100.03))
    exchange.limit_order_process()
    print("check limit order")
    print("Limit Order Book (Bids):", exchange.LOB['bid'])
    print("Limit Order Book (Asks):", exchange.LOB['ask'])

    # Place some market orders

    exchange.market_order_placement(Order(quantity=120, direction='buy', type='market'))
    exchange.market_order_placement(Order(quantity=110, direction='sell', type='market'))

  

    # Process the market orders
    print(exchange.check_agent_order_filled(ask_price=100.03, ask_qty=50, bid_price=99.95, bid_qty=30))

    # Print the state of the LOB
    print("check market order")
    print("Limit Order Book (Bids):", exchange.LOB['bid'])
    print("Limit Order Book (Asks):", exchange.LOB['ask'])


test()
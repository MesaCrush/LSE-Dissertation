import numpy as np
from .order import Order
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
            bid_qty =  np.random.randint(50, 100)
            ask_qty =  np.random.randint(50, 100)
            bid_price = round(self.initial_bid - level * tick_size, 2)
            ask_price = round(self.initial_ask + level * tick_size, 2)
            
            self.limit_order_queue.append(Order(quantity = bid_qty, direction = 'buy', type = 'limit', price = bid_price))
            self.limit_order_queue.append(Order(quantity = ask_qty, direction = 'sell', type = 'limit', price = ask_price))

        self.__limit_order_process()
        
    def market_order_placement(self, order):
        Exchange.market_order_queue.append(order)  
    
    def limit_order_placement(self, order):
        Exchange.limit_order_queue.append(order)  

    def __limit_order_process(self):
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

    def __remove_zero_quantity_levels(self, book):
        """Remove price levels with zero quantity from the given book."""
        return {price: qty for price, qty in book.items() if qty > 0}

    def __market_order_process(self):
        while self.market_order_queue:
            order = self.market_order_queue.popleft()
            rest_qty = order.qty
            book = self.LOB['ask'] if order.direction == 'buy' else self.LOB['bid']
            agent_price = self.agent_ask_price if order.direction == 'buy' else self.agent_bid_price
            
            for price in list(book.keys()):  # Use list() to avoid runtime error
                if rest_qty >= book[price]:
                    filled_qty = book[price]
                    rest_qty -= filled_qty
                    book[price] = 0
                else:
                    filled_qty = rest_qty
                    book[price] -= rest_qty
                    rest_qty = 0
                
                if price == agent_price:
                    if order.direction == 'buy':
                        self.filled_ask += min(filled_qty, self.agent_ask_qty - self.filled_ask)
                    else:
                        self.filled_bid += min(filled_qty, self.agent_bid_qty - self.filled_bid)
                
                if rest_qty == 0:
                    break
            
            book = self.__remove_zero_quantity_levels(book)
            if order.direction == 'buy':
                self.LOB['ask'] = book
            else:
                self.LOB['bid'] = book

    def check_agent_order_filled(self, ask_price=0, ask_qty=0, bid_price=0, bid_qty=0):
        self.agent_ask_price = ask_price
        self.agent_ask_qty = ask_qty
        self.agent_bid_price = bid_price
        self.agent_bid_qty = bid_qty
        self.filled_ask = 0
        self.filled_bid = 0
        self.__limit_order_process()
        self.__market_order_process()
        return self.filled_ask, self.filled_bid
    
    def get_best_bid(self):
        return max(self.LOB['bid'].keys())
    
    def get_best_ask(self):
        return min(self.LOB['ask'].keys())
    
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
 
        best_bid_qty = list(bid_book.values())[0]
        best_ask_qty = list(ask_book.values())[0]

        return best_bid_qty / (best_bid_qty + best_ask_qty), sum(bid_book.values()) / (sum(bid_book.values()) + sum(ask_book.values()))

    def get_distinct_limit_order_num(self):
        return len(self.LOB['bid']) + len(self.LOB['ask'])


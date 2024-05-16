import random
import numpy as np
import itertools
from order import Order

class Exchange:
    """    
    This class simulates the dynamics of the LOB
    - initialise a symmetric LOB by inputting initial_mid 
    - define tick_size which is minimum price movement
    - initialise 20-level LOB, each level is 1 tick_size away
    
    Then LOB will be updated by new Market orders and limit orders   
    """
    market_order_lst = []
    
    def __init__(self, initial_mid, tick_size):
        # self.prices = [initial_mid]
        self.initial_mid = initial_mid
        self.initial_bid = initial_mid - 3 * tick_size
        self.initial_ask = initial_mid + 3 * tick_size
        self.tick_size = tick_size
        self.target_id_lst = []
        self.target_price_lst = []
        self.target_qty_lst = []
        self.trade_price = []
        self.trade_id = []
        self.trade_qty = []
        # Variables related to reward function calculations
        self.total_volume = 0
        self.total_pv = 0 # price * volume 
        self.price_before_execution = 0
        
       
        self.depth = 30
        self.LOB = {'bid': {}, 'ask': {}}
        self.prices = [initial_mid]
        
        self.limit_order_placement(Order(quantity = 50, direction = 'buy', type = 'limit', price = self.initial_bid),False)
        self.limit_order_placement(Order(quantity = 50, direction = 'sell', type = 'limit', price = self.initial_ask),False)
            
        for level in range(1, self.depth):
            bid_qty =  np.random.randint(1, 100)
            ask_qty =  np.random.randint(1, 100)
            bid_price = round(self.initial_bid - level * tick_size, 2)
            ask_price = round(self.initial_ask + level * tick_size, 2)
            
            self.limit_order_placement(Order(quantity = bid_qty, direction = 'buy', type = 'limit', price = bid_price),False)
            self.limit_order_placement(Order(quantity = ask_qty, direction = 'sell', type = 'limit', price = ask_price),False)


    def limit_order_placement(self, order, add_to_target_id_list):
        """
        this function is to process the new placement of both limit and market order
        book is a dictionary, key is price, value is a list-of-list, [id, qty]
        at the same price, order follows first-in-first-out principle
        bid is in descending order, ask is in ascending order
        """
        trade_num = int(len(self.trade_price))
        if add_to_target_id_list:
            self.target_id_lst.append(order.id)
            self.target_price_lst.append(order.price)
            self.target_qty_lst.append(order.qty)
       
        bid_book = self.LOB['bid']
        ask_book = self.LOB['ask']
       
        if order.type == 'limit':
            book = bid_book if order.direction == 'buy' else ask_book
           
            if order.price in book:
                book[order.price].append( [order.id, order.qty] )
            else:
                book[order.price] = [ [order.id, order.qty] ]
           
            self.LOB['bid'] = dict(sorted(bid_book.items(), reverse=True))
            self.LOB['ask'] = dict(sorted(ask_book.items(), reverse=False))        
           
           # the function also returns the traded price, qty and order_id if the limit order was sent from our agent and got filled
            res = self.process_bid_ask_cross()
            
            if add_to_target_id_list and trade_num != len(self.trade_price):
                if sum(res[trade_num:]) == order.qty:
                    self.target_id_lst.remove(order.id)
                    self.target_price_lst.remove(order.price)
                    self.target_qty_lst.remove(order.qty)
        else: # market order
            return
 
       
    def process_bid_ask_cross(self):
        """
        this function is to make sure the LOB's ask is always higher than the bid (i.e. no cross market)
        This need to be handled differently when the limit order is placed by the simulator or the agent
        
        1) simulated limit order: cancel the overlapped bid/ask immediately
            - assume some other market participants found the arbitrage opportunity
            - example: current best bid price = 100.01, qty = 50
                       new coming ask price = 100, qty = 30 -> there exists a cross market, i.e. ask <= bid
                       the current best bid will be updated as price = 100.01, qty = 50 - 30 = 20
                       the new coming ask price will be cancelled, 
                       because we assume there exists some traders in the market figure out this arbitrage opportunity and take the bid and ask immediately
                       
        2) limit order placed by the agent: record the available ask rate and qty as trade_price,trade_qty
           i.e. this means the agent is willing to buy at any price better or equal to the quoted-bid-rate
           therefore, any LOB's ask rate that is lower than the quoted-bid-rate becomes 'available' traded price
           Then, the traded_qty should be min(bid_qty, availalbe_ask_qty)
                   
        """
        trade_id = self.trade_id
        trade_price = self.trade_price
        trade_qty = self.trade_qty
       
        bid_book = self.LOB['bid']
        ask_book = self.LOB['ask']
       
        if not ( ask_book and bid_book ):
            return
       
        while ask_book:
            key = list(ask_book.keys())[0]
            value = ask_book[key]
           
            if not bid_book:
                break
           
            first_bid_book_key = list(bid_book.keys())[0]
            if key > first_bid_book_key:
                break
           
            # same quote price could have several orders, so aggregate the corresponding qty
            ask_total_qty = 0.0
            for item in value:
                ask_total_qty += item[1]
           
            bid_total_qty = 0.0
            for item in bid_book[first_bid_book_key]:
                bid_total_qty += item[1]
           
            book_removed = ask_book if ask_total_qty <= bid_total_qty else bid_book
            book_left = ask_book if ask_total_qty > bid_total_qty else bid_book
            remove_qty = ask_total_qty if ask_total_qty < bid_total_qty else bid_total_qty
           
            for item in book_removed[list(book_removed.keys())[0]]:
                # if the quote is placed by our agent, record
                if item[0] in self.target_id_lst:
                    trade_id.append(item[0])
                    trade_price.append(key)
                    trade_qty.append(item[1])
           
            book_removed.pop(list(book_removed.keys())[0], None)
           
            left_key = list(book_left.keys())[0]
            while remove_qty > 0:
                do_remove = False
                qty = 0.0
                if book_left[left_key][0][1] <= remove_qty:
                    remove_qty -= book_left[left_key][0][1]
                    do_remove = True
                    qty = book_left[left_key][0][1]
                else:
                    book_left[left_key][0][1] -= remove_qty
                    qty = remove_qty
                    remove_qty = 0
               
                if book_left[left_key][0][0] in self.target_id_lst:
                    trade_id.append(book_left[left_key][0][0])
                    trade_price.append(key)
                    trade_qty.append(qty)
               
                if do_remove:
                    book_left[left_key].pop(0)
                   
            if not book_left[left_key]:
                book_left.pop(left_key, None)
            
        return trade_qty

    
    def get_distinct_limit_order_id(self):
        """ 
        count distinct order_id across price levels.
        """
        bid_order_ids = set()
        ask_order_ids = set()

        # Process bid orders
        for orders in self.LOB['bid'].values():
            for order in orders:
                bid_order_ids.add(order[0])

        # Process ask orders
        for orders in self.LOB['ask'].values():
            for order in orders:
                ask_order_ids.add(order[0])

        return bid_order_ids, ask_order_ids
        

    def get_distinct_limit_order_num(self):
        order = self.get_distinct_limit_order_id()
        bid_order_num = len(order[0])
        ask_order_num = len(order[1])
        return bid_order_num, ask_order_num


    def process_limit_cancellation(self, book):
        if len(book) <= 3:
            return "book price level <= 3"
        
        price_lst = list(book.keys())
        num = len(price_lst)
        first_3 = price_lst[:3]
        remaining_first_half = price_lst[3:num//2]
        remaining_second_half = price_lst[num//2:]
        
        part_select_prob = [0.05, 0.15, 0.8]
        selected_part = random.choices([first_3, remaining_first_half, remaining_second_half], weights=part_select_prob, k=3)
        
        cancel_num = np.random.randint(0, 3) if num < 25 else np.random.randint(2,5)
        
        if 0 < cancel_num <= len(selected_part[0]):
            px_cancel = random.sample(selected_part[0], cancel_num)
        
            for px in px_cancel:
                del book[px][0]
                if len(book[px]) == 0:
                    del book[px]
        
        
    def limit_order_cancellation(self):
        """
        Some market makers might want to cancel their limit orders for many kinds of reasons, e.g. they react on the price movements, etc.
        In order to simulate this behaviour, we cancel some quotes in the LOB
        Different level range have different survival rate
        e.g. more deeper in the book, the more likely to get cancelled
        Also, if there are fewer limit orders, then don't do anything
        """
        ord_id_lst = self.get_distinct_limit_order_id()
        bid_num = len(ord_id_lst[0])
        ask_num = len(ord_id_lst[1])

        if bid_num < 10 or ask_num < 10:
            return
        
        bid_book = self.LOB['bid']
        ask_book = self.LOB['ask']
        
        self.process_limit_cancellation(bid_book)
        self.process_limit_cancellation(ask_book)


    def cancel_agent_limit_order(self):
        if len(self.target_price_lst) == 0:
            return
        self.target_id_lst.clear()
        self.target_price_lst.clear()
        self.target_qty_lst.clear()
        

    def market_order_placement(self, order):
        Exchange.market_order_lst.append(order)    
    
    def check_LOB_zero_qty(self, LOB_side):
         book = self.LOB[LOB_side]
         price_list = list(book.keys())
         
         for price in price_list:
             n_ord = int(len(book[price]))
             for i in range(0,n_ord):
                 if book[price][0][1] != 0:
                     break
                 if book[price][0][1] == 0:
                     if book[price][0][0] in self.target_id_lst:
                         self.trade_id.append(book[price][0][0])
                         self.trade_price.append(self.target_price_lst[0])
                         self.trade_qty.append(self.target_qty_lst[0])                      
                         self.target_price_lst.clear()
                         self.target_id_lst.clear()
                         self.target_qty_lst.clear()
                     del book[price][0]
                 if len(book[price]) == 0:
                     del book[price]
                 

    def process_market_orders(self, agent_order_id):
        random.shuffle(Exchange.market_order_lst)
        agent_trade_price = 0
        executed_qty = 0
        executed_amount = 0
        while Exchange.market_order_lst:
            order = Exchange.market_order_lst.pop(0)
            if order.direction == 'buy':
                for price in sorted(self.LOB['ask']):
                    self.price_before_execution = price
                    for i, each_order in enumerate(self.LOB['ask'][price]):
                        available_qty = each_order[1]
                        if available_qty > 0:
                            trade_qty = min(order.qty, available_qty)
                            self.LOB['ask'][price][i][1] -= trade_qty
                            order.qty -= trade_qty
                            self.total_volume += trade_qty
                            self.total_pv += trade_qty * price
                            if agent_order_id == order.id:
                                executed_amount += price * trade_qty
                                executed_qty += trade_qty
                self.check_LOB_zero_qty('ask')
            else:  # 'sell'
                for price in sorted(self.LOB['bid'], reverse=True):
                    self.price_before_execution = price
                    for i, each_order in enumerate(self.LOB['bid'][price]):
                        available_qty = each_order[1]
                        if available_qty > 0:
                            trade_qty = min(order.qty, available_qty)
                            self.LOB['bid'][price][i][1] -= trade_qty
                            order.qty -= trade_qty
                            self.total_volume += trade_qty
                            self.total_pv += trade_qty * price
                            if agent_order_id == order.id:
                                executed_amount += price * trade_qty
                                executed_qty += trade_qty
                self.check_LOB_zero_qty('bid')
         
        if executed_qty > 0:
            agent_trade_price = executed_amount / executed_qty  # Calculate average price

        return agent_trade_price
        

    def check_agent_limit_order_fill_process(self):
        if len(self.target_price_lst) == 0:
            return
        
        if self.target_price_lst[0] not in list(self.LOB['bid'].keys()):
            self.target_id_lst.clear()
            self.target_price_lst.clear()
            self.target_qty_lst.clear()
            return
        
        for item in self.LOB['bid'][self.target_price_lst[0]]:
            if item[0] in self.target_id_lst:
                remaining_qty = item[1]
                trade_qty = self.target_qty_lst[0] - remaining_qty
                self.trade_id.append(self.target_id_lst[0])
                self.trade_price.append(self.target_price_lst[0])
                self.trade_qty.append(trade_qty)
                self.target_qty_lst[0] = remaining_qty
        
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

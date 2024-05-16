# -*- coding: utf-8 -*-
from Aggregated_Market_Simulation import Order
from Aggregated_Market_Simulation import Exchange
from Aggregated_Market_Simulation import Market_Maker
from Aggregated_Market_Simulation import Market_Orders
import matplotlib.pyplot as plt

exchange = Exchange(100,0.01)
print("initial book: ", exchange.LOB)

# # place new limit orders to test the LOB dynamic
# # True means the limit order is placed by our agent

exchange.limit_order_placement(Order(quantity=50, direction='sell', type='limit', price=99.95),False)
exchange.limit_order_placement(Order(quantity=10, direction='buy', type='limit', price=99.9),False)
exchange.limit_order_placement(Order(quantity=24, direction='sell', type='limit', price=100.35),False)
exchange.limit_order_placement(Order(quantity=30, direction='buy', type='limit', price=99.8),False)
exchange.limit_order_placement(Order(quantity=30, direction='sell', type='limit', price=99.8),True)
print("after inserting new orders", exchange.LOB)

# # test a few functions
# # exchange class
exchange.get_mid_price()
exchange.get_book_imbalance()
exchange.get_spread()
exchange.limit_order_cancellation()
exchange.get_distinct_limit_order_id()
exchange.get_distinct_limit_order_num()


# # market_maker class
exchange = Exchange(100,0.01)
exchange.LOB

mm = Market_Maker(exchange)
mm.new_simulated_limit_order_per_step()
print(exchange.LOB)


# # market order class
trader = Market_Orders(exchange)
print("after traders")
trader.new_simulated_market_order_per_step()
exchange.process_market_orders(agent_order_id=None)

print(exchange.LOB)

   

# run the simulation result
exchange = Exchange(100,0.01)
mm = Market_Maker(exchange)
mo = Market_Orders(exchange)

mid_rate = []

for i in range (0, 1000):
    print(i)
    exchange.limit_order_cancellation()
    mm.new_simulated_limit_order_per_step()
    if i <= 2:
        mm.new_simulated_limit_order_per_step()
    mo.new_simulated_market_order_per_step()
    exchange.process_market_orders(agent_order_id=None)
    mid = exchange.get_mid_price()
    mid_rate.append(mid)
    print(exchange.get_book_imbalance())
    print(exchange.get_spread())
    # print(exchange.LOB)
   
t_step = list(range(len(mid_rate)))
plt.figure(figsize=(10, 5)) 
plt.plot(t_step, mid_rate, linestyle='-')  
plt.grid(True)  

# Show the plot
plt.show()


mid_rate













class Order:
    last_id = 0
    def __init__(self, quantity, direction, type, price=None) -> None:
        self.qty = quantity
        self.direction = direction
        self.type = type
        self.price = price
        self.id = Order.generate_order_id()
       
    def __str__(self) -> str:
        return f'order_id:{self.id},  order_type:{self.type}, order_price:{self.price}, quantity:{self.qty}, side:{self.direction}'
   
    def __repr__(self) -> str:
        return f'order_id:{self.id},  order_type:{self.type}, order_price:{self.price}, quantity:{self.qty}, side:{self.direction}'

    @classmethod
    def generate_order_id(cls):
        """
        Generates a new order ID by incrementing the last used ID by 1.
        """
        cls.last_id += 1  # Increment the last ID
        return cls.last_id
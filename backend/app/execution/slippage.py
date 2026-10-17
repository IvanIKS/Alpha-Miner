from app.execution.order import Order


class Slippage:
    def __init__(self):
        pass

class SimpleSlippage(Slippage):
    def calculate(self, order : Order):
        return order.price * 0.01
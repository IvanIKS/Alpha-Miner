from decimal import Decimal

from app.execution.order import Order


class Slippage:
    def __init__(self):
        pass

class SimpleSlippage(Slippage):
    def __init__(self, rate=0.01):
        super().__init__()
        self.rate = rate

    def calculate(self, order : Order) -> Decimal:
        return Decimal(order.price * self.rate)
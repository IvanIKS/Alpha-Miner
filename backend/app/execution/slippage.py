from decimal import Decimal

from app.execution.order import Order


class Slippage:
    def __init__(self):
        pass

class SimpleSlippage(Slippage):
    def calculate(self, order : Order) -> Decimal:
        return Decimal(order.price * 0.01)
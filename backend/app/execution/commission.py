from decimal import Decimal

from app.execution.order import Order


class Commission:
    def __init__(self, buy_commission_percents, sell_commission_percents):
        self.buy_commission = buy_commission_percents
        self.sell_commission = sell_commission_percents

    def calculate(self, order : Order) -> Decimal:
        if order.side == "buy":
            return Decimal(order.quantity * (self.buy_commission / 100))
        else:
            return Decimal(order.quantity * (self.sell_commission / 100))
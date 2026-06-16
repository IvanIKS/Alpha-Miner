import unittest
from decimal import Decimal


class TestEquityCurve(unittest.TestCase):

    def test_equity_curve(self):
        from app.execution.execution_engine import BacktestExecutionEngine
        engine = BacktestExecutionEngine()
        # Tracks {symbol: {'price': float, 'qty': float, 'side': str}}
        engine.open_positions = {'BTCUSDT': {'side': 'buy',  'quantity' : 1000, 'price' : Decimal(500.0)}}

        engine.populate_equity_curve({'BTCUSDT' : Decimal(1000.0)}, 1000.0)


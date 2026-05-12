import unittest


from app.backtest.engine import BacktestEngine, BacktestResult  # Update 'your_module' to the actual folder

# Import the specific strategy
from app.strategies.implementation.RSIStrategy import RSIStrategy


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        """Set up the test environment before each test runs."""
        self.engine = BacktestEngine()
        self.strategy = RSIStrategy(symbol="BTCUSDT", timeframe="1m")
        
        # Your specific test data file
        self.historical_data_file = r"/app/data\backtest_data\binance\backtest_data_btc_usdt_2026-02-01_00-00-00_2026-03-01_00-00-00_1m.csv"
        
        self.initial_money = 1000.0
        self.money_symbol = 'USDT'

    def test_run_with_rsi_strategy(self):
        """Test the run method using RSIStrategy and the local 1m BTC/USDT dataset."""
        print("\n--- Starting Integration Test ---")
        
        # Act
        try:
            result = self.engine.run(
                strategy=self.strategy,
                historical_data_file=self.historical_data_file,
                money_amount=self.initial_money,
                money_symbol=self.money_symbol
            )

            print(result)
        except Exception as e:
            self.fail(f"BacktestEngine.run() raised an unexpected exception: {e}")

        # Assert
        # 1. Ensure a BacktestResult object is returned
        self.assertIsInstance(result, BacktestResult, "Result must be a BacktestResult object")
        
        # 2. Verify the strategy name carried over to the result correctly
        self.assertEqual(result.strategy_name, self.strategy.name, "Result should map to the RSI strategy name")
        

if __name__ == '__main__':
    unittest.main()
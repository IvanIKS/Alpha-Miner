from collections import deque

from app.execution.signal import SimpleSignal
from app.strategies.Strategy import Strategy

import talib
import numpy as np

from app.strategies.rules.strategy_requirements import MustBePandasDataFrame, MustBeBinanceOHLCVData

class RSIStrategy(Strategy):
    """Default values were gotten from Backtests.ipynb"""
    def __init__(self, symbol : str ="BTC/USDT", exchange = "binance", timeframe="1m", rsi_period=8, overbought_threshold=62, oversold_threshold=36):
        super().__init__("RSIStrategy",
                         "Simple RSI strategy")
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.rsi_period = rsi_period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
        self.data_requirements = [MustBePandasDataFrame(), MustBeBinanceOHLCVData()]
        self.max_window = rsi_period + 50
        self.data_window = deque(maxlen=self.max_window)

    def on_tick(self, context):
        self.data_window.append(context.market_data['close'])

        if len(self.data_window) < self.max_window:
            return []

        close_prices = np.fromiter(self.data_window, dtype=float)

        rsi_values = talib.RSI(close_prices, timeperiod=self.rsi_period)
        current_rsi = rsi_values[-1]

        if np.isnan(current_rsi):
            return []


        portfolio = context.portfolio
        last_close_price = close_prices[-1]
        if current_rsi > self.overbought_threshold and len(portfolio.positions) > 0:
            return [SimpleSignal(symbol=self.symbol,
                                 quantity=portfolio.positions[0].quantity,
                                 price=last_close_price,
                                 side="sell")]
        elif current_rsi < self.oversold_threshold and len(portfolio.positions) <= 0:
            target_asset_quantity = portfolio.base_token_amount / last_close_price
            return [SimpleSignal(symbol=self.symbol,
                                 quantity=target_asset_quantity,
                                 price=last_close_price,
                                 side="buy")]

        return []
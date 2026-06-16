from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict

from asyncpg import cursor

from app.data.service.ohlcv_service import get_ohlcv_between_dates
from app.strategies.Strategy import Strategy

import pandas as pd

class MarketDataProvider:
    def get_market_data(self, strategy):
        """Build a Context for the given strategy's required symbols/timeframe."""
        raise NotImplementedError("DataProvider is an abstract class. Implement get_market_data in a subclass.")

    def get_asset_prices(self, strategy: Strategy):
        """A way for different data providers to get asset prices as of date equity curve"""
        raise NotImplementedError("DataProvider is an abstract class. Implement get_asset_prices in a subclass.")

class PandasBacktestDataProvider(MarketDataProvider):
    def __init__(self, strategy : Strategy, historical_data_file):
        """Build a  for the given strategy's required symbols/timeframe."""
        self.strategy = strategy
        self.historical_data = pd.read_csv(historical_data_file)
        strategy.validate_data_requirements(self.historical_data)
        self.first_row = self.historical_data.iloc[0]
        self.last_row = self.historical_data.iloc[-1]
        self.cursor = 0

    def get_market_data(self, strategy : Strategy):
        #returns next row of historical data as a Context object
        if self.historical_data.empty:
            return None
        if self.cursor >= len(self.historical_data):
            return None
        row = self.historical_data.iloc[self.cursor]
        self.cursor += 1

        return row

    def get_asset_prices(self, strategy: Strategy):
        return {self.strategy.symbol : Decimal(self.historical_data.iloc[self.cursor]["close"])}





class GapsFoundError(Exception):
    def __init__(self, gaps, timeframe):
        self.gaps = gaps
        self.timeframe = timeframe

class OHLCVDataProvider(MarketDataProvider):
    def __init__(
        self,
        strategy: Strategy,
        start_date: datetime,
        end_date: datetime,
        exchange: str,
        symbol: str,
        timeframe: str
    ):
        """
        Initializes the provider by fetching data from the DB
        and triggering backfills for missing gaps.
        """
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe


        rows, gaps = get_ohlcv_between_dates(
            start_date, end_date, exchange, symbol, timeframe
        )

        if gaps:
            raise GapsFoundError(gaps, timeframe)

        self.data = rows
        self.cursor = 0

    def get_market_data(self, strategy: Strategy) -> Optional[Dict]:
        """Returns the next row of data, simulating a live feed."""
        if self.cursor >= len(self.data):
            return None

        row = self.data[self.cursor]
        self.cursor += 1
        return row

    def get_asset_prices(self, strategy: Strategy) -> Optional[Dict]:
        return {self.symbol : Decimal(self.data[self.cursor - 1]["close"])}

    def __len__(self):
        return len(self.data)

from datetime import datetime

from app.adapters.implementations.prediction.types import MarketSnapshot, PricePoint
from app.execution.commission import Commission


class Adapter:
    def get_commission(self) -> Commission:
        raise NotImplementedError("Commission fetching is not implemented.")

class ExchangeAdapter(Adapter):
    def backfill_ohlcv(self, start_date : datetime, end_date : datetime, symbol : str, timeframe : str):
        """Take OHLCV data from exchange for timeframe from start_date to end_date and write it into the database"""
        raise NotImplementedError("Backfill OHLCV data is not implemented.")

class PredictionMarketAdapter(Adapter):
    """Base class for prediction-market data sources (Polymarket, Kalshi, ...).

    Implementations expose two capabilities:
      * ``fetch_markets`` — a snapshot of currently-open markets with their
        current outcome prices (market discovery).
      * ``fetch_price_history`` — a historical price time-series for a single
        market outcome / token.
    """

    platform: str = ""

    def fetch_markets(self, limit: int | None = None) -> list[MarketSnapshot]:
        """Fetch currently-open markets (optionally capped at ``limit``)."""
        raise NotImplementedError("Market snapshot fetching is not implemented.")

    def fetch_price_history(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> list[PricePoint]:
        """Fetch the historical price time-series for a single outcome token."""
        raise NotImplementedError("Price history fetching is not implemented.")

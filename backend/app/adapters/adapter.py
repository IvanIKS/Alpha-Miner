from datetime import datetime

from app.execution.commission import Commission


class Adapter:
    def backfill_ohlcv(self, start_date : datetime, end_date : datetime, symbol : str, timeframe : str):
        """Take OHLCV data from exchange for timeframe from start_date to end_date and write it into the database"""
        raise NotImplementedError("Backfill OHLCV data is not implemented.")

    def get_commission(self) -> Commission:
        return NotImplementedError("Commission fetching is not implemented.")
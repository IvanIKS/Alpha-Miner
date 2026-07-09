from datetime import datetime

from app.adapters.adapter_registry import get_adapter
from app.data.database.session import find_gaps, db_session

class BackfillService:
    def __init__(self, exchange: str):
        self.exchange = exchange
        self.adapter = get_adapter(exchange)

    def run(self, start_timestamp: datetime, end_timestamp: datetime,symbol: str, timeframe: str):
        with db_session() as session:
            gaps = find_gaps(session, start_timestamp, end_timestamp, timeframe, self.exchange, symbol)
            print(f"Found {len(gaps)} gaps")
            #Duplicate logic needed to ensure you don't write same candles twice.
            if gaps:
                for gap in gaps:
                    print(f"Backfilling {gap[0]} to {gap[1]}")
                    self.adapter.backfill_ohlcv(gap[0], gap[1], symbol, timeframe)

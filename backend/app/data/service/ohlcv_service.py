from datetime import datetime

from app.data.database.session import db_session, find_gaps
from app.data.database.models import OHLCV
from app.background_tasks.tasks.backfill import backfill_ohlcv_task

def get_ohlcv_between_dates(start_date: datetime, end_date: datetime, exchange: str, symbol: str, timeframe: str):
    with db_session() as session:
        gaps = set(find_gaps(session.connection(), start_date, end_date,  timeframe, exchange, symbol))
        if len(gaps) > 0:
            for gap in gaps:
                backfill_ohlcv_task.delay(
                    exchange=exchange,
                    symbol=symbol,
                    start_timestamp=gap[0],
                    end_timestamp=gap[1],
                    timeframe=timeframe,
                )

        rows = session.query(OHLCV).filter(
            OHLCV.timestamp >= start_date,
            OHLCV.timestamp <= end_date,
            OHLCV.exchange == exchange,
            OHLCV.symbol == symbol,
            OHLCV.timeframe == timeframe,
        ).order_by(OHLCV.timestamp).all()

        return [
            {
                "timestamp": r.timestamp,
                "exchange": r.exchange,
                "symbol": r.symbol,
                "timeframe": r.timeframe,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ], [(g[0].isoformat(), g[1].isoformat()) for g in gaps]


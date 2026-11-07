from datetime import datetime

from sqlalchemy import Float, cast, select

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

        # Core select of just the needed columns, with Numeric cast to float in the DB:
        # avoids building one mapped OHLCV object per row and the per-value Decimal
        # allocation, which is what made 1m ranges take ~an hour to materialise.
        stmt = (
            select(
                OHLCV.timestamp,
                OHLCV.exchange,
                OHLCV.symbol,
                OHLCV.timeframe,
                cast(OHLCV.open, Float).label("open"),
                cast(OHLCV.high, Float).label("high"),
                cast(OHLCV.low, Float).label("low"),
                cast(OHLCV.close, Float).label("close"),
                cast(OHLCV.volume, Float).label("volume"),
            )
            .where(
                OHLCV.timestamp >= start_date,
                OHLCV.timestamp <= end_date,
                OHLCV.exchange == exchange,
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe,
            )
            .order_by(OHLCV.timestamp)
        )
        rows = session.execute(stmt).mappings().all()

        return [dict(r) for r in rows], [(g[0].isoformat(), g[1].isoformat()) for g in gaps]


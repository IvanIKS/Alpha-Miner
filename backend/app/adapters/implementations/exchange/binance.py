from dotenv import load_dotenv
import ccxt
import os
import datetime
import time

from psycopg2._psycopg import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.adapters.adapter import ExchangeAdapter
from app.data.database.session import db_session
from app.data.database.models import OHLCV
from app.execution.commission import Commission
from app.execution.order import Order
from app.utils.timeframe import get_candles_left, seconds_map



class BinanceAdapter(ExchangeAdapter):
    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)

        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,  # helps prevent rate-limit issues
        })

        # Binance features discount commission for spot BNB trading
        self.commission = BinanceCommission()


    def backfill_ohlcv(self, start_date : datetime.datetime, end_date : datetime.datetime, symbol, timeframe):
        """Write OHLCV for [start_date, end_date) into the DB.

        The exchange omits candles for hours it has no data for (outages / no-trade
        hours). Those holes are "interior" when they sit between real candles — we
        synthesise a flat bar (prev close as O/H/L/C, volume 0, forward_filled=True)
        so the gap is resolved permanently instead of being re-detected every run.
        Trailing holes after the last real candle the exchange returns are left
        alone: we never fabricate not-yet-existing (future) data.
        """
        step_ms = seconds_map[timeframe] * 1000
        start_time = int(start_date.timestamp() * 1000)
        end_time = int(end_date.timestamp() * 1000)

        # Seed the forward-fill price from the last real candle before the gap.
        last_close = self._last_close_before(start_date, symbol, timeframe)
        expected = start_time  # next grid timestamp (ms) we still owe a row for

        since = start_time
        candles_left = get_candles_left(since, end_time, timeframe)

        while candles_left > 0:
            limit = 1000
            if candles_left < 1000:
                limit = candles_left

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            rows = []
            for candle in ohlcv:
                timestamp_ms = candle[0]

                # If there is a gap in the exchange's candles, fill it with a flat bar.
                while expected < min(timestamp_ms, end_time) and last_close is not None:
                    rows.append(self._candle_row(expected, symbol, timeframe, last_close))
                    expected += step_ms

                if timestamp_ms >= end_time:
                    break

                rows.append(self._candle_row(
                    timestamp_ms, symbol, timeframe,
                    open_=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5],
                ))
                last_close = candle[4]
                expected = timestamp_ms + step_ms

            self._upsert(rows)

            since = ohlcv[-1][0] + 1
            candles_left = get_candles_left(since, end_time, timeframe)
            time.sleep(self.exchange.rateLimit / 1000)
            filled = sum(1 for r in rows if r["forward_filled"])
            print(f"Loaded {len(rows)} candles ({filled} forward-filled) up to {since}")

    def _last_close_before(self, before: datetime.datetime, symbol: str, timeframe: str):
        """Close of the most recent real candle strictly before ``before``, or None."""
        with db_session() as session:
            stmt = (
                select(OHLCV.close)
                .where(
                    OHLCV.exchange == "binance",
                    OHLCV.symbol == symbol,
                    OHLCV.timeframe == timeframe,
                    OHLCV.timestamp < before,
                )
                .order_by(OHLCV.timestamp.desc())
                .limit(1)
            )
            return session.execute(stmt).scalar_one_or_none()

    def _candle_row(self, ts_ms, symbol, timeframe, close, *, open_=None, high=None, low=None, volume=None):
        """Build a row dict. With only ``close`` given it's a forward-filled flat bar."""
        forward_filled = open_ is None
        if forward_filled:
            open_ = high = low = close
            volume = 0
        return {
            "timestamp": datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc),
            "exchange": "binance",
            "symbol": symbol,
            "timeframe": timeframe,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "forward_filled": forward_filled,
        }

    def _upsert(self, rows):
        """Insert rows, skipping any whose (timestamp, exchange, symbol, timeframe)
        already exists — so re-running a backfill never duplicates candles."""
        if not rows:
            return
        with db_session() as session:
            stmt = pg_insert(OHLCV.__table__).values(rows).on_conflict_do_nothing(
                index_elements=["timestamp", "exchange", "symbol", "timeframe"]
            )
            session.execute(stmt)

    def get_commission(self) -> Commission:
        return self.commission

class BinanceCommission(Commission):
    # TODO : add commission fetch
    def __init__(self, buy_commission_percents=0.1, sell_commission_percents=0.1):
        super().__init__(buy_commission_percents, sell_commission_percents)
        self.BNB_discounted_buy_commission_percents = 0.075
        self.BNB_discounted_sell_commission_percents = 0.075

    def calculate(self, order : Order) -> Decimal:
        buy_commission = self.buy_commission
        sell_commission = self.sell_commission
        if order.symbol.contains("BNB") :
            buy_commission = self.BNB_discounted_buy_commission_percents
            sell_commission = self.BNB_discounted_sell_commission_percents
        if order.side == "buy":
            return order.quantity * (buy_commission / 100)
        else:
            return order.quantity * (sell_commission / 100)

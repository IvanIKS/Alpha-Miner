from dotenv import load_dotenv
import ccxt
import os
import datetime
import time

from app.adapters.adapter import Adapter
from app.data.database.session import db_session
from app.data.database.models import OHLCV
from app.execution.commission import Commission
from app.utils.timeframe import get_candles_left



class BinanceAdapter(Adapter):
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

        #TODO : add commission fetch
        self.commission = Commission(buy_commission_percents=0.1, sell_commission_percents=0.1)


    def backfill_ohlcv(self, start_date : datetime.datetime, end_date : datetime.datetime, symbol, timeframe):
        start_time = int(start_date.timestamp() * 1000)
        end_time = int(end_date.timestamp() * 1000)

        since = start_time
        candles_left = get_candles_left(since, end_time, timeframe)

        while candles_left > 0:
            limit = 1000
            if candles_left < 1000:
                limit = candles_left

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            rows = [
                OHLCV(
                    timestamp=datetime.datetime.fromtimestamp(candle[0] / 1000, tz=datetime.timezone.utc),
                    exchange="binance",
                    symbol=symbol,
                    timeframe=timeframe,
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5],
                )
                for candle in ohlcv
            ]

            with db_session() as session:
                session.bulk_save_objects(rows)

            since = ohlcv[-1][0] + 1
            candles_left = get_candles_left(since, end_time, timeframe)
            time.sleep(self.exchange.rateLimit / 1000)
            print(f"Loaded {len(rows)} candles up to {since}")

        def get_commission(self) -> Commission:
            return self.commission
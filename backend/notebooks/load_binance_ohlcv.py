from dotenv import load_dotenv
import ccxt
import os
import pandas as pd
import datetime
import time


start_date = datetime.datetime(2020, 1, 1, 0, 0)
end_date = datetime.datetime(2020, 6, 1, 0, 0)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,  # helps prevent rate-limit issues
})

symbol = 'BTC/USDT'
timeframe = '1m'



start_time = int(start_date.timestamp() * 1000)
end_time = int(end_date.timestamp() * 1000)

all_ohlcv = []
since = start_time

while since < end_time:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
    if not ohlcv:
        break
    all_ohlcv.extend(ohlcv)
    since = ohlcv[-1][0] + 1  # move to next timestamp
    time.sleep(exchange.rateLimit / 1000)  # respect rate limit
    print("Loaded for timestamp: " + str(since))

# Create DataFrame
df = pd.DataFrame(all_ohlcv, columns=[
    'open time', 'open', 'high', 'low', 'close', 'volume'
])

# Convert timestamp and types
#df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
for col in ['open time', 'open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(df.head(), "\n", df.tail())

start_str = start_date.strftime('%Y-%m-%d_%H-%M-%S')
end_str = end_date.strftime('%Y-%m-%d_%H-%M-%S')

filename = f'backtest_data_btc_usdt_{start_str}_{end_str}_1m.csv'
df.to_csv(filename, index=False)

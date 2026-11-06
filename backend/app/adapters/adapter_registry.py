from app.adapters.adapter import Adapter, PredictionMarketAdapter
from app.adapters.implementations.exchange.binance import BinanceAdapter
from app.adapters.implementations.prediction.polymarket import PolymarketAdapter
from app.adapters.implementations.prediction.kalshi import KalshiAdapter
from app.adapters.implementations.prediction.predict_fun import PredictFunAdapter

ADAPTERS = {
    "binance": BinanceAdapter()
}

# Prediction-market adapters are kept in a separate registry: they implement the
# PredictionMarketAdapter interface (fetch_markets / fetch_price_history), not the
# OHLCV backfill path that BackfillService drives ADAPTERS through.
PREDICTION_ADAPTERS = {
    "polymarket": PolymarketAdapter(),
    "kalshi": KalshiAdapter(),
    "predict.fun": PredictFunAdapter(),
}

def get_adapter(exchange_name : str) -> Adapter:
    name = exchange_name.lower()
    adapter = ADAPTERS.get(name)
    if adapter is None:
        raise NotImplementedError(f"Backfilling from exchange: {exchange_name} is not implemented yet")
    return adapter


def get_prediction_adapter(platform: str) -> PredictionMarketAdapter:
    adapter = PREDICTION_ADAPTERS.get(platform.lower())
    if adapter is None:
        raise NotImplementedError(f"Prediction market platform '{platform}' is not implemented yet")
    return adapter
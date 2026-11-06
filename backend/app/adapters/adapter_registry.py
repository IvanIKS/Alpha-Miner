from app.adapters.adapter import Adapter
from app.adapters.implementations.exchange.binance import BinanceAdapter

ADAPTERS = {
    "binance": BinanceAdapter()
}

def get_adapter(exchange_name : str) -> Adapter:
    name = exchange_name.lower()
    adapter = ADAPTERS.get(name)
    if adapter is None:
        raise NotImplementedError(f"Backfilling from exchange: {exchange_name} is not implemented yet")
    return adapter
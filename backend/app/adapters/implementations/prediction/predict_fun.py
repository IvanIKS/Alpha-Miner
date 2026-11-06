"""Predict.fun prediction-market adapter (interface-conformant stub).

Predict.fun is a Blast-based prediction market that runs a Polymarket-style CLOB,
but it does not currently expose a stable, documented public REST API for market
discovery or historical prices. This adapter therefore conforms to the
``PredictionMarketAdapter`` interface so it can be registered and routed like the
others, but its live calls raise ``NotImplementedError`` until the API surface is
confirmed.

TODO: wire up real calls once the public endpoints are available. The expected
shape is a CLOB gateway (markets + ``prices-history``) similar to Polymarket's
``clob.polymarket.com``; see ``polymarket.py`` for the pattern to follow.
"""

from __future__ import annotations

from datetime import datetime

from app.adapters.adapter import PredictionMarketAdapter
from app.adapters.implementations.prediction.types import MarketSnapshot, PricePoint

_PENDING = "Predict.fun API integration pending: no stable public API available yet."


class PredictFunAdapter(PredictionMarketAdapter):
    platform = "predict.fun"

    def fetch_markets(self, limit: int | None = None) -> list[MarketSnapshot]:
        raise NotImplementedError(_PENDING)

    def fetch_price_history(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> list[PricePoint]:
        raise NotImplementedError(_PENDING)

"""Kalshi prediction-market adapter.

Snapshot discovery uses the public ``/markets`` endpoint; historical price
candles use ``/series/{series}/markets/{ticker}/candlesticks``. Both are public
read endpoints on the production host ``https://api.elections.kalshi.com``.

Kalshi quotes prices in cents (0..100); they are normalised here to the
0.0..1.0 probability range to match the other adapters. An optional API key can
be supplied via ``KALSHI_API_KEY`` (loaded from a local ``.env``) for
authenticated/higher-rate access, but market data reads work without it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from app.adapters.adapter import PredictionMarketAdapter
from app.adapters.implementations.prediction.types import (
    MarketOutcome,
    MarketSnapshot,
    PricePoint,
)
from app.utils.timeframe import seconds_map

# Kalshi only supports these candlestick period intervals (minutes).
_ALLOWED_PERIODS = (1, 60, 1440)


def _cents_to_prob(cents) -> float | None:
    """Convert a Kalshi cents price (0..100) to a 0.0..1.0 probability."""
    if cents is None:
        return None
    try:
        return float(cents) / 100.0
    except (TypeError, ValueError):
        return None


def _period_interval(timeframe: str) -> int:
    """Map a timeframe string to the nearest allowed Kalshi period_interval."""
    minutes = max(1, seconds_map.get(timeframe, 60) // 60)
    return min(_ALLOWED_PERIODS, key=lambda allowed: abs(allowed - minutes))


class KalshiAdapter(PredictionMarketAdapter):
    platform = "kalshi"

    def __init__(self):
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)

        self.base_url = os.getenv(
            "KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2"
        ).rstrip("/")
        self.api_key = os.getenv("KALSHI_API_KEY")

    def _headers(self) -> dict:
        headers = {"Accept": "application/json", "User-Agent": "AlphaMiner/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _market_to_snapshot(self, raw: dict) -> MarketSnapshot:
        ticker = str(raw.get("ticker", ""))
        yes = _cents_to_prob(raw.get("last_price") or raw.get("yes_bid"))
        outcomes = [MarketOutcome(name="Yes", price=yes, token_id=ticker)]
        if yes is not None:
            outcomes.append(MarketOutcome(name="No", price=round(1.0 - yes, 6), token_id=""))

        return MarketSnapshot(
            platform="kalshi",
            market_id=ticker,
            condition_id=str(raw.get("event_ticker", "")),
            question=raw.get("title", "") or "",
            slug=ticker,
            description=raw.get("subtitle", "") or raw.get("yes_sub_title", "") or "",
            end_date=raw.get("close_time", "") or "",
            created_at=raw.get("open_time", "") or "",
            outcomes=outcomes,
            volume=float(raw.get("volume") or 0.0),
            volume_24h=float(raw.get("volume_24h") or 0.0),
            liquidity=float(raw.get("liquidity") or 0.0),
            active=(raw.get("status") == "active" or raw.get("status") == "open"),
            closed=(raw.get("status") == "closed" or raw.get("status") == "settled"),
            accepting_orders=bool(raw.get("can_close_early", True)),
            event_slug=str(raw.get("event_ticker", "")),
            event_title=raw.get("title", "") or "",
            tags=[c for c in [raw.get("category")] if c],
        )

    def fetch_markets(self, limit: int | None = None) -> list[MarketSnapshot]:
        markets: list[MarketSnapshot] = []
        cursor: str | None = None

        while True:
            params = {"limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor

            resp = requests.get(
                f"{self.base_url}/markets", params=params, headers=self._headers(), timeout=30
            )
            resp.raise_for_status()
            payload = resp.json()

            page = payload.get("markets") or []
            if not page:
                break

            for raw in page:
                markets.append(self._market_to_snapshot(raw))
                if limit and len(markets) >= limit:
                    return markets

            cursor = payload.get("cursor")
            if not cursor:
                break

        return markets

    def fetch_price_history(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> list[PricePoint]:
        # Kalshi ticker format is SERIES-EVENT-STRIKE; the series is the prefix.
        series_ticker = token_id.split("-")[0]
        params = {
            "start_ts": int(start_date.timestamp()),
            "end_ts": int(end_date.timestamp()),
            "period_interval": _period_interval(timeframe),
        }

        resp = requests.get(
            f"{self.base_url}/series/{series_ticker}/markets/{token_id}/candlesticks",
            params=params,
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        candles = resp.json().get("candlesticks") or []

        points: list[PricePoint] = []
        for c in candles:
            price = c.get("price") or {}
            close = _cents_to_prob(price.get("close") if price.get("close") is not None else price.get("mean"))
            if close is None:
                continue
            open_ = _cents_to_prob(price.get("open"))
            high = _cents_to_prob(price.get("high"))
            low = _cents_to_prob(price.get("low"))
            ts = datetime.fromtimestamp(int(c["end_period_ts"]), tz=timezone.utc)
            points.append(
                PricePoint(
                    timestamp=ts,
                    open=open_ if open_ is not None else close,
                    high=high if high is not None else close,
                    low=low if low is not None else close,
                    close=close,
                    volume=float(c["volume"]) if c.get("volume") is not None else None,
                    open_interest=float(c["open_interest"]) if c.get("open_interest") is not None else None,
                )
            )
        return points

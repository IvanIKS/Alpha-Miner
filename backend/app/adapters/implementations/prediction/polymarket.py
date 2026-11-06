"""Polymarket prediction-market adapter.

Snapshot discovery uses the public Gamma API (``/events?closed=false``), ported
from ``backend/notebooks/load_polymarket.py``. Historical price time-series use
the public CLOB ``/prices-history`` endpoint. Neither requires authentication.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from app.adapters.adapter import PredictionMarketAdapter
from app.adapters.implementations.prediction.types import (
    MarketOutcome,
    MarketSnapshot,
    PricePoint,
)
from app.utils.timeframe import seconds_map

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
EVENTS_ENDPOINT = f"{GAMMA_API_BASE}/events"

CLOB_API_BASE = "https://clob.polymarket.com"
PRICES_HISTORY_ENDPOINT = f"{CLOB_API_BASE}/prices-history"

DEFAULT_PAGE_SIZE = 100  # max items per request
REQUEST_DELAY = 0.15     # seconds between paginated requests (stay under rate limits)

# The CLOB /prices-history endpoint rejects a startTs/endTs window longer than
# ~2 weeks ("interval is too long"), so long ranges are fetched in chunks.
_MAX_HISTORY_WINDOW = timedelta(days=10)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _json_list(raw: Any) -> list:
    """Parse a field that Polymarket encodes as a JSON-string list (or plain list)."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [raw] if raw else []
    return []


def _parse_outcomes(raw: dict) -> list[MarketOutcome]:
    """Extract outcomes with prices and CLOB token ids from a raw market dict."""
    outcome_names = _json_list(raw.get("outcomes"))
    outcome_prices = [_safe_float(p) for p in _json_list(raw.get("outcomePrices"))]
    token_ids = _json_list(raw.get("clobTokenIds"))

    results: list[MarketOutcome] = []
    for i, name in enumerate(outcome_names):
        price = outcome_prices[i] if i < len(outcome_prices) else None
        tid = token_ids[i] if i < len(token_ids) else ""
        results.append(MarketOutcome(name=str(name), price=price, token_id=str(tid)))
    return results


def _parse_market(raw: dict, event_info: dict | None = None) -> MarketSnapshot:
    """Convert a raw Gamma market dict into a MarketSnapshot."""
    tags_raw = raw.get("tags") or []
    tag_labels: list[str] = []
    if isinstance(tags_raw, list):
        for t in tags_raw:
            if isinstance(t, dict):
                tag_labels.append(t.get("label", t.get("slug", "")))
            else:
                tag_labels.append(str(t))

    event_slug = event_info.get("slug", "") if event_info else ""
    event_title = event_info.get("title", "") if event_info else ""

    return MarketSnapshot(
        platform="polymarket",
        market_id=str(raw.get("id", "")),
        condition_id=str(raw.get("conditionId", "")),
        question=raw.get("question", "") or "",
        slug=raw.get("slug", "") or "",
        description=raw.get("description", "") or "",
        end_date=raw.get("endDate", "") or "",
        created_at=raw.get("createdAt", "") or "",
        outcomes=_parse_outcomes(raw),
        volume=_safe_float(raw.get("volumeNum") or raw.get("volume")),
        volume_24h=_safe_float(raw.get("volume24hr")),
        liquidity=_safe_float(raw.get("liquidityNum") or raw.get("liquidity")),
        active=bool(raw.get("active", True)),
        closed=bool(raw.get("closed", False)),
        accepting_orders=bool(raw.get("acceptingOrders", True)),
        event_slug=event_slug,
        event_title=event_title,
        tags=tag_labels,
    )


class PolymarketAdapter(PredictionMarketAdapter):
    platform = "polymarket"

    def fetch_markets(self, limit: int | None = None) -> list[MarketSnapshot]:
        """Fetch all currently-open markets via the Gamma ``/events`` endpoint.

        Uses ``/events`` (with ``closed=false``) because each event embeds its
        child markets, which is the retrieval strategy recommended by
        Polymarket's own documentation.
        """
        all_markets: list[MarketSnapshot] = []
        offset = 0
        seen_ids: set[str] = set()

        while True:
            params = {
                "closed": "false",
                "order": "id",
                "ascending": "false",
                "limit": DEFAULT_PAGE_SIZE,
                "offset": offset,
            }

            resp = requests.get(EVENTS_ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            events: list[dict] = resp.json()
            if not events:
                break

            for event in events:
                for raw_mkt in event.get("markets") or []:
                    mid = str(raw_mkt.get("id", ""))
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)

                    if raw_mkt.get("closed", False):
                        continue

                    all_markets.append(_parse_market(raw_mkt, event_info=event))
                    if limit and len(all_markets) >= limit:
                        return all_markets

            if len(events) < DEFAULT_PAGE_SIZE:
                break

            offset += DEFAULT_PAGE_SIZE
            time.sleep(REQUEST_DELAY)

        return all_markets

    def fetch_price_history(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> list[PricePoint]:
        """Fetch a CLOB price time-series for a single outcome token.

        Polymarket returns one probability price per timestamp, so the point is
        stored as a flat candle (open == high == low == close). Long ranges are
        fetched in chunks because the endpoint caps the window length.
        """
        fidelity = max(1, seconds_map.get(timeframe, 60) // 60)

        points: dict[int, PricePoint] = {}
        window_start = start_date
        while window_start < end_date:
            window_end = min(window_start + _MAX_HISTORY_WINDOW, end_date)
            params = {
                "market": token_id,
                "startTs": int(window_start.timestamp()),
                "endTs": int(window_end.timestamp()),
                "fidelity": fidelity,
            }

            resp = requests.get(PRICES_HISTORY_ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            history = resp.json().get("history") or []

            for point in history:
                t = int(point["t"])
                price = _safe_float(point.get("p"))
                points[t] = PricePoint(
                    timestamp=datetime.fromtimestamp(t, tz=timezone.utc),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                )

            window_start = window_end
            if window_start < end_date:
                time.sleep(REQUEST_DELAY)

        return [points[t] for t in sorted(points)]

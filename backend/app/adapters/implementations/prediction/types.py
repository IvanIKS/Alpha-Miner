"""Platform-agnostic domain DTOs for prediction-market adapters.

These are plain dataclasses used as the common currency between the adapters
(Polymarket / Kalshi / Predict.fun) and the persistence/service layer. They are
deliberately decoupled from the SQLAlchemy models in
``app.data.database.models`` so adapters can be used standalone (e.g. in
notebooks) without a database.

The snapshot structures mirror the dataclasses originally prototyped in
``backend/notebooks/load_polymarket.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MarketOutcome:
    """A single outcome (e.g. Yes / No) of a prediction market."""

    name: str
    price: float | None = None
    token_id: str = ""


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of one prediction market across any platform."""

    platform: str = ""
    market_id: str = ""
    condition_id: str = ""
    question: str = ""
    slug: str = ""
    description: str = ""
    end_date: str = ""
    created_at: str = ""
    outcomes: list[MarketOutcome] = field(default_factory=list)
    volume: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    closed: bool = False
    accepting_orders: bool = True
    event_slug: str = ""
    event_title: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def yes_price(self) -> float | None:
        """Return the 'Yes' outcome price, or the first outcome price."""
        for o in self.outcomes:
            if o.name.lower() == "yes":
                return o.price
        return self.outcomes[0].price if self.outcomes else None


@dataclass
class PricePoint:
    """A single historical price candle for one market outcome / token.

    Platforms that expose only a single probability price per timestamp
    (e.g. Polymarket) set ``open == high == low == close``. Platforms with full
    candles (e.g. Kalshi) populate all four plus ``volume``/``open_interest``.
    All prices are normalised to the ``0.0 .. 1.0`` probability range.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    open_interest: float | None = None

"""Persistence layer for prediction-market snapshots and price history.

Mirrors ``BackfillService`` for the OHLCV pipeline: it pulls data from a
``PredictionMarketAdapter`` and upserts it into the ``prediction_market`` /
``prediction_price`` tables idempotently.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.adapters.adapter_registry import get_prediction_adapter
from app.data.database.models import PredictionMarket, PredictionPrice
from app.data.database.session import db_session


class PredictionMarketService:
    def __init__(self, platform: str):
        self.platform = platform
        self.adapter = get_prediction_adapter(platform)

    def sync_markets(self, limit: int | None = None) -> int:
        """Fetch open markets and upsert their snapshots. Returns rows written."""
        markets = self.adapter.fetch_markets(limit=limit)
        if not markets:
            return 0

        rows = [
            {
                "platform": self.platform,
                "market_id": m.market_id,
                "condition_id": m.condition_id,
                "question": m.question,
                "slug": m.slug,
                "description": m.description,
                "end_date": m.end_date,
                "created_at": m.created_at,
                "outcomes": [asdict(o) for o in m.outcomes],
                "volume": m.volume,
                "volume_24h": m.volume_24h,
                "liquidity": m.liquidity,
                "active": m.active,
                "closed": m.closed,
                "accepting_orders": m.accepting_orders,
                "event_slug": m.event_slug,
                "event_title": m.event_title,
                "tags": m.tags,
            }
            for m in markets
        ]

        with db_session() as session:
            stmt = pg_insert(PredictionMarket).values(rows)
            # Refresh the snapshot on conflict (markets change over time).
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in PredictionMarket.__table__.columns
                if c.name not in ("platform", "market_id")
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["platform", "market_id"], set_=update_cols
            )
            session.execute(stmt)

        return len(rows)

    def backfill_prices(
        self,
        market_id: str,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> int:
        """Fetch and store a price history time-series. Returns rows fetched."""
        points = self.adapter.fetch_price_history(token_id, start_date, end_date, timeframe)
        if not points:
            return 0

        rows = [
            {
                "platform": self.platform,
                "market_id": market_id,
                "token_id": token_id,
                "timeframe": timeframe,
                "timestamp": p.timestamp,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
                "open_interest": p.open_interest,
            }
            for p in points
        ]

        # Idempotent: skip candles already stored (same pattern as the OHLCV backfill).
        with db_session() as session:
            stmt = pg_insert(PredictionPrice).values(rows).on_conflict_do_nothing(
                index_elements=["platform", "market_id", "token_id", "timeframe", "timestamp"]
            )
            session.execute(stmt)

        return len(rows)

    def get_markets(self, limit: int | None = None) -> list[PredictionMarket]:
        with db_session() as session:
            query = (
                session.query(PredictionMarket)
                .filter(PredictionMarket.platform == self.platform)
                .order_by(PredictionMarket.volume_24h.desc().nullslast())
            )
            if limit:
                query = query.limit(limit)
            rows = query.all()
            session.expunge_all()
            return rows

    def get_prices(
        self,
        market_id: str,
        token_id: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[PredictionPrice]:
        with db_session() as session:
            rows = (
                session.query(PredictionPrice)
                .filter(
                    PredictionPrice.platform == self.platform,
                    PredictionPrice.market_id == market_id,
                    PredictionPrice.token_id == token_id,
                    PredictionPrice.timeframe == timeframe,
                    PredictionPrice.timestamp >= start_date,
                    PredictionPrice.timestamp <= end_date,
                )
                .order_by(PredictionPrice.timestamp)
                .all()
            )
            session.expunge_all()
            return rows

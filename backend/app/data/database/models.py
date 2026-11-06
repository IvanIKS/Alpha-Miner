from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func, Column, PrimaryKeyConstraint, Numeric, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.data.database.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    login: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    bots: Mapped[List["Bot"]] = relationship(back_populates="strategy", cascade="all, delete-orphan")
    backtests: Mapped[List["BacktestResult"]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), index=True, nullable=False)
    additional_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    money_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    money_symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), nullable=False, index=True)
    strategy: Mapped["Strategy"] = relationship(back_populates="bots")

    symbols: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), nullable=False, index=True)
    strategy: Mapped["Strategy"] = relationship(back_populates="backtests")

    test_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)

    results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default='done')
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    equity_curve: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    buy_and_hold_curve: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class OHLCV(Base):
    __tablename__ = "ohlcv"

    timestamp = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    exchange = Column(String, nullable=False, primary_key=True)
    symbol = Column(String, nullable=False, primary_key=True)
    timeframe = Column(String, nullable=False, primary_key=True)
    open = Column(Numeric, nullable=False)
    high = Column(Numeric, nullable=False)
    low = Column(Numeric, nullable=False)
    close = Column(Numeric, nullable=False)
    volume = Column(Numeric, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "exchange", "symbol", "timeframe"),
    )

    def __repr__(self):
        return (
            f"OHLCV(timestamp={self.timestamp}, exchange={self.exchange}, "
            f"symbol={self.symbol}, timeframe={self.timeframe}, "
            f"open={self.open}, high={self.high}, low={self.low}, "
            f"close={self.close}, volume={self.volume})"
        )


class PredictionMarket(Base):
    """Snapshot of one prediction market (Polymarket / Kalshi / Predict.fun).

    One row per (platform, market_id); re-syncing refreshes the snapshot.
    """

    __tablename__ = "prediction_market"

    platform = Column(String, nullable=False, primary_key=True)
    market_id = Column(String, nullable=False, primary_key=True)

    condition_id = Column(String, nullable=True)
    question = Column(Text, nullable=True)
    slug = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    end_date = Column(String, nullable=True)
    created_at = Column(String, nullable=True)
    # outcomes: list of {name, price, token_id}
    outcomes = Column(JSON, nullable=False, default=list)
    volume = Column(Numeric, nullable=True)
    volume_24h = Column(Numeric, nullable=True)
    liquidity = Column(Numeric, nullable=True)
    active = Column(Boolean, nullable=False, server_default="true")
    closed = Column(Boolean, nullable=False, server_default="false")
    accepting_orders = Column(Boolean, nullable=False, server_default="true")
    event_slug = Column(String, nullable=True)
    event_title = Column(String, nullable=True)
    tags = Column(JSON, nullable=False, default=list)
    fetched_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("platform", "market_id"),
    )


class PredictionPrice(Base):
    """Historical price candle for a single market outcome / token.

    Mirrors the OHLCV model. Platforms exposing only a single probability price
    set open == high == low == close; prices are normalised to 0.0 .. 1.0.
    """

    __tablename__ = "prediction_price"

    platform = Column(String, nullable=False, primary_key=True)
    market_id = Column(String, nullable=False, primary_key=True)
    token_id = Column(String, nullable=False, primary_key=True)
    timeframe = Column(String, nullable=False, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, primary_key=True)
    open = Column(Numeric, nullable=False)
    high = Column(Numeric, nullable=False)
    low = Column(Numeric, nullable=False)
    close = Column(Numeric, nullable=False)
    volume = Column(Numeric, nullable=True)
    open_interest = Column(Numeric, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("platform", "market_id", "token_id", "timeframe", "timestamp"),
    )


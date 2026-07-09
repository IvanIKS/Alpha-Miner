from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
import importlib
import inspect
import pkgutil

from sqlalchemy import select

from app.data.database.base import Base
from app.utils.hash import hash_password
from app.data.database.models import Strategy as StrategyModel, User
from app.strategies.Strategy import Strategy as BaseStrategy

DEFAULT_LOGIN = "user"
DEFAULT_PASSWORD = "user"


def _iter_strategy_instances() -> list[BaseStrategy]:
    """Dynamically import strategy implementations and return instances of them.

    Modify the strategy in code and it'll persist in the database after running the seed function, without the need to modify the seed function itself.
    """
    strategy_instances: list[BaseStrategy] = []

    package_name = "app.strategies.implementation"
    package = importlib.import_module(package_name)

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg:
            continue

        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseStrategy) or obj is BaseStrategy:
                continue

            strategy_instances.append(obj())

    return strategy_instances


def _timeframe_value(value: object) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def seed_defaults() -> None:
    """Create default records if they do not exist yet."""
    init_db()

    with db_session() as db:
        existing_user = db.execute(select(User).where(User.login == DEFAULT_LOGIN)).scalar_one_or_none()
        if existing_user is None:
            db.add(User(login=DEFAULT_LOGIN, password_hash=hash_password(DEFAULT_PASSWORD)))

        for strategy in _iter_strategy_instances():
            existing_strategy = (
                db.execute(select(StrategyModel).where(StrategyModel.name == strategy.name)).scalar_one_or_none()
            )
            if existing_strategy is not None:
                continue

            db.add(
                StrategyModel(
                    name=strategy.name,
                    description=strategy.description
                )
            )

def _database_url() -> str:
    return os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/app")


engine = create_engine(_database_url())




SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)


def init_db() -> None:
    from app.data.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    _migrate_backtest_results(engine)

    init_timescale_db(engine)


def _migrate_backtest_results(eng: Engine) -> None:
    with eng.begin() as conn:
        for stmt in [
            "ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'done'",
            "ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS celery_task_id VARCHAR(255)",
            "ALTER TABLE backtest_results ADD COLUMN IF NOT EXISTS equity_curve TEXT",
        ]:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                print(f"Migration note: {exc}")

def init_timescale_db(engine: Engine) -> None:
    with engine.begin() as connection:
        # Enable TimescaleDB extension if not already enabled
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

        init_hypertables(connection)

        init_indexes(connection)

        init_functions(connection)



def init_hypertables(connection):
    try:
        connection.execute(
            text("SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE)")
        )
    except Exception as e:
        # If it's already a hypertable or table doesn't exist, continue
        print(f"Hypertable creation note: {e}")

def init_indexes(connection):
    # Create composite index for common queries
    try:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_ohlcv_exchange_symbol_timeframe_timestamp "
                "ON ohlcv (exchange, symbol, timeframe, timestamp DESC)"
            )
        )
    except Exception as e:
        print(f"Index creation note: {e}")

def init_functions(connection):
    sql_path = Path(__file__).parent / "sql_scripts" / "find_gaps.sql"
    try:
        sql = sql_path.read_text()
        connection.exec_driver_sql(sql)
    except Exception as e:
        print(f"SQL function or type creation error: {e}")

def find_gaps(
    conn,
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
    exchange: str,
    symbol: str,
    check_right: bool = True,
) -> set[tuple[datetime, datetime]]:
    """Call the find_ohlcv_gaps SQL function and return list of (gap_start, gap_end)."""
    query = text("""
        SELECT gap_start, gap_end
        FROM find_ohlcv_gaps(
            :start_date,
            :end_date,
            :timeframe,
            :exchange,
            :symbol
        )
    """)

    rows = conn.execute(query, {
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "exchange": exchange,
            "symbol": symbol
        }).fetchall()

    return set([(row[0], row[1]) for row in rows])

def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
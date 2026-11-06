from datetime import datetime
from typing import Any, List, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from app.background_tasks.celery_app import celery
from app.background_tasks.tasks.prediction import (
    backfill_prediction_prices_task,
    sync_prediction_markets_task,
)
from app.data.service.prediction_market_service import PredictionMarketService

router = APIRouter(prefix="/prediction_markets", tags=["Prediction markets"])


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


class JobResponse(BaseModel):
    status: str
    job_id: str
    platform: str
    message: str


class MarketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    platform: str
    market_id: str
    question: Optional[str] = None
    slug: Optional[str] = None
    end_date: Optional[str] = None
    outcomes: Any = None
    volume: Optional[float] = None
    volume_24h: Optional[float] = None
    liquidity: Optional[float] = None
    closed: bool = False
    event_title: Optional[str] = None


class PricePointResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    open_interest: Optional[float] = None


@router.post("/{platform}/sync", response_model=JobResponse)
async def sync_markets(
    platform: str,
    limit: Optional[int] = Query(None, description="Max markets to fetch (None = all)"),
) -> JobResponse:
    """Queue a job that fetches open markets for the platform and stores snapshots."""
    task = sync_prediction_markets_task.delay(platform=platform, limit=limit)
    return JobResponse(
        status="sync_queued",
        job_id=task.id,
        platform=platform,
        message="Market snapshot sync has been queued for processing",
    )


@router.get("/{platform}/markets", response_model=List[MarketResponse])
async def list_markets(
    platform: str,
    limit: Optional[int] = Query(100, description="Max rows to return"),
) -> List[MarketResponse]:
    """Return stored market snapshots for the platform (highest 24h volume first)."""
    service = PredictionMarketService(platform)
    return service.get_markets(limit=limit)


@router.post("/{platform}/prices", response_model=JobResponse)
async def backfill_prices(
    platform: str,
    market_id: str = Query(..., description="Platform market id"),
    token_id: str = Query(..., description="Outcome token id (Polymarket clobTokenId / Kalshi ticker)"),
    start_time: str = Query(..., description="Start timestamp (ISO 8601)"),
    end_time: str = Query(..., description="End timestamp (ISO 8601)"),
    timeframe: str = Query("1h", description="Candle timeframe (1m, 1h, 1d)"),
) -> JobResponse:
    """Queue a job that backfills a market outcome's historical price time-series."""
    task = backfill_prediction_prices_task.delay(
        platform=platform,
        market_id=market_id,
        token_id=token_id,
        start_timestamp=_parse_dt(start_time),
        end_timestamp=_parse_dt(end_time),
        timeframe=timeframe,
    )
    return JobResponse(
        status="backfill_queued",
        job_id=task.id,
        platform=platform,
        message="Price history backfill has been queued for processing",
    )


@router.get("/{platform}/prices", response_model=List[PricePointResponse])
async def get_prices(
    platform: str,
    market_id: str = Query(...),
    token_id: str = Query(...),
    start_time: str = Query(...),
    end_time: str = Query(...),
    timeframe: str = Query("1h"),
) -> List[PricePointResponse]:
    """Return stored price history for a market outcome."""
    service = PredictionMarketService(platform)
    return service.get_prices(
        market_id=market_id,
        token_id=token_id,
        timeframe=timeframe,
        start_date=_parse_dt(start_time),
        end_date=_parse_dt(end_time),
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = AsyncResult(job_id, app=celery)
    return {
        "job_id": job_id,
        "state": result.state,
        "meta": result.info if result.state != "PENDING" else None,
    }

import csv
from datetime import datetime
import io
from typing import List

from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse

from app.background_tasks.celery_app import celery
from app.background_tasks.tasks.backfill import backfill_ohlcv_task
import app.data.service as service
from celery.result import AsyncResult

from fastapi import APIRouter, Query, HTTPException

router = APIRouter(prefix="/historical_data", tags=["Historical data"])

class OHLCVCandle(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCVResponse(BaseModel):
    status: str
    exchange: str
    symbol: str
    start_timestamp: datetime
    end_timestamp: datetime
    data: List[OHLCVCandle]
    count: int


class BackfillRequest(BaseModel):
    start_time: str
    end_time: str
    timeframe: str
    exchange: str
    symbol: str


class BackfillResponse(BaseModel):
    status: str
    job_id: str
    exchange: str
    symbol: str
    start_timestamp: datetime
    end_timestamp: datetime
    message: str



def results_to_csv(results) -> str:
    output = io.StringIO()
    fields = ["timestamp", "exchange", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()


@router.get("/ohlcv")
async def get_ohlcv(
        start_time: str = Query(..., description="Start timestamp (ISO 8601 format, e.g., 2024-01-01T00:00:00Z)"),
        end_time: str = Query(..., description="End timestamp (ISO 8601 format, e.g., 2024-01-02T00:00:00Z)"),
        exchange: str = Query("binance", description="Exchange name"),
        symbol: str = Query(..., description="Trading symbol"),
        timeframe: str = Query("1m", description="Timeframe (1m, 1h, etc.)"),
        response_format: str = Query("csv", description="Response format: json or csv"),
):
    start_dt = datetime.fromisoformat(start_time.strip().replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.strip().replace("Z", "+00:00"))

    results, gaps = service.ohlcv_service.get_ohlcv_between_dates(
        start_date=start_dt,
        end_date=end_dt,
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
    )

    status = "complete" if not gaps else "partial"

    if response_format == "csv":
        if status == "complete":
            csv_data = results_to_csv(results)
            return StreamingResponse(
                io.StringIO(csv_data),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=ohlcv.csv"},
            )
        else:
            raise HTTPException(status_code=404,
                                detail="Requested data not found in database. Backfill job queued, try again in few minutes.")
    else :
        return OHLCVResponse(
            status=status,
            exchange=exchange,
            symbol=symbol,
            start_timestamp=start_dt,
            end_timestamp=end_dt,
            data=results,
            count=len(results),
        )

@router.post("/ohlcv", response_model=BackfillResponse)
async def backfill_ohlcv(start_time: str = Query(..., description="Start timestamp (ISO 8601 format, e.g., 2024-01-01T00:00:00Z)"),
    end_time: str = Query(..., description="End timestamp (ISO 8601 format, e.g., 2024-01-02T00:00:00Z)"),
    timeframe: str = Query("1m", description="Timeframe (1m, 1h, etc.)"),
    exchange: str = Query("binance", description="Exchange name"),
    symbol: str = Query(..., description="Trading symbol"),
) -> BackfillResponse:
    """
    Trigger a backfill job to fetch and store OHLCV data.

    Request Body:
    - start_timestamp: Unix timestamp (seconds) for data start
    - end_timestamp: Unix timestamp (seconds) for data end
    - exchange: Exchange identifier (binance, kraken, coinbase, etc.)
    - symbol: Trading pair symbol (BTC/USDT, ETH/USDC, etc.)

    Returns a job ID for tracking the backfill progress.
    In production, this would queue a background task (Celery, RQ, etc.)
    """
    start_dt = datetime.fromisoformat(start_time.strip().replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.strip().replace("Z", "+00:00"))
    task = backfill_ohlcv_task.delay(
        exchange=exchange,
        symbol=symbol,
        start_timestamp=start_dt,
        end_timestamp=end_dt,
        timeframe=timeframe,
    )

    return BackfillResponse(
        status="backfill_queued",
        job_id=task.id,
        exchange=exchange,
        symbol=symbol,
        start_timestamp=start_dt,
        end_timestamp=end_dt,
        message="Backfill job has been queued for processing"
    )

class OHLCVCoverageEntry(BaseModel):
    exchange: str
    symbol: str
    timeframe: str
    from_date: datetime
    to_date: datetime
    candle_count: int


@router.get("/coverage", response_model=List[OHLCVCoverageEntry])
async def get_coverage():
    """Return distinct (exchange, symbol, timeframe) combinations with date ranges and candle counts."""
    from app.data.database.session import db_session
    from app.data.database.models import OHLCV as OHLCVModel
    from sqlalchemy import func as sa_func

    with db_session() as session:
        rows = (
            session.query(
                OHLCVModel.exchange,
                OHLCVModel.symbol,
                OHLCVModel.timeframe,
                sa_func.min(OHLCVModel.timestamp).label("from_date"),
                sa_func.max(OHLCVModel.timestamp).label("to_date"),
                sa_func.count().label("candle_count"),
            )
            .group_by(OHLCVModel.exchange, OHLCVModel.symbol, OHLCVModel.timeframe)
            .order_by(OHLCVModel.exchange, OHLCVModel.symbol, OHLCVModel.timeframe)
            .all()
        )
        return [
            OHLCVCoverageEntry(
                exchange=r.exchange,
                symbol=r.symbol,
                timeframe=r.timeframe,
                from_date=r.from_date,
                to_date=r.to_date,
                candle_count=r.candle_count,
            )
            for r in rows
        ]


@router.get("/ohlcv/jobs/{job_id}")
async def get_job_status(job_id: str):
    result = AsyncResult(job_id, app=celery)
    return {
        "job_id": job_id,
        "state": result.state,
        "meta": result.info if result.state != "PENDING" else None,
    }


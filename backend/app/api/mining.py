"""API for the alpha-mining strategy comparison (Celery): launch a run and inspect results."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select

from app.background_tasks.tasks.lab.lab import compare_strategy_mining
from app.data.database.models import ArmResult, MiningRun
from app.data.database.session import db_session

router = APIRouter(prefix="/mining", tags=["Alpha mining"])


class CompareRequest(BaseModel):
    """Any field overrides the defaults in tasks.lab.lab.DEFAULTS; omit for the notebook run."""
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    holdout_fraction: Optional[float] = None
    budget: Optional[int] = None
    pop_size: Optional[int] = None
    k: Optional[int] = None
    buy_commission: Optional[float] = None      # percent per side
    sell_commission: Optional[float] = None     # percent per side
    slippage_percent: Optional[float] = None    # percent of price per side
    cost_bps: Optional[float] = None            # derived from commission + slippage if omitted
    money: Optional[float] = None
    model_id: Optional[str] = None
    arms: Optional[list[int]] = None
    seeds: Optional[list[int]] = None


class CompareResponse(BaseModel):
    task_id: str
    status: str


class ArmResultResponse(BaseModel):
    arm: int
    arm_name: Optional[str]
    seed: int
    status: str
    rule: Optional[str]
    minutes: Optional[float]
    model_calls: Optional[int]
    backtests_used: Optional[int]
    backtest_result_id: Optional[int]
    summary: Optional[Any]


class MiningRunResponse(BaseModel):
    id: int
    created_at: datetime
    symbol: str
    exchange: str
    timeframe: str
    status: str
    arms: Any
    seeds: Any
    summary: Optional[Any]
    arm_results: list[ArmResultResponse]


@router.post("/compare", response_model=CompareResponse, status_code=status.HTTP_202_ACCEPTED)
def launch_comparison(payload: CompareRequest | None = None) -> CompareResponse:
    config = payload.model_dump(exclude_none=True) if payload else {}
    task = compare_strategy_mining.delay(config)
    return CompareResponse(task_id=task.id, status="queued")


@router.get("/runs")
def list_runs() -> list[dict]:
    with db_session() as db:
        runs = db.execute(select(MiningRun).order_by(MiningRun.id.desc())).scalars().all()
        return [{"id": r.id, "created_at": r.created_at, "status": r.status,
                 "symbol": r.symbol, "exchange": r.exchange, "timeframe": r.timeframe,
                 "arms": r.arms, "seeds": r.seeds} for r in runs]


@router.get("/{run_id}", response_model=MiningRunResponse)
def get_run(run_id: int) -> MiningRunResponse:
    with db_session() as db:
        run = db.get(MiningRun, run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mining run not found")
        arm_rows = db.execute(
            select(ArmResult).where(ArmResult.mining_run_id == run_id)
            .order_by(ArmResult.arm.asc(), ArmResult.seed.asc())
        ).scalars().all()
        return MiningRunResponse(
            id=run.id, created_at=run.created_at, symbol=run.symbol, exchange=run.exchange,
            timeframe=run.timeframe, status=run.status, arms=run.arms, seeds=run.seeds,
            summary=run.summary,
            arm_results=[ArmResultResponse(
                arm=a.arm, arm_name=a.arm_name, seed=a.seed, status=a.status, rule=a.rule,
                minutes=a.minutes, model_calls=a.model_calls, backtests_used=a.backtests_used,
                backtest_result_id=a.backtest_result_id, summary=a.summary,
            ) for a in arm_rows],
        )

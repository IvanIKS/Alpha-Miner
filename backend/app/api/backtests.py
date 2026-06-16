from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtest.engine import BacktestEngine
from app.data.database.models import BacktestResult, Strategy
from app.data.database.session import get_db
from app.execution.runtime_registry import StrategyFactory

router = APIRouter(prefix="/backtests", tags=["Backtests"])

_strategy_factory = StrategyFactory()


class BacktestCreateRequest(BaseModel):
    strategy_id: int
    dataset_name: str
    results: str


class BacktestUpdateRequest(BaseModel):
    strategy_id: int | None = None
    dataset_name: str | None = None
    results: str | None = None


class BacktestResponse(BaseModel):
    id: int
    strategy_id: int
    strategy_name: str
    test_time: datetime
    dataset_name: str
    results: str


def _require_strategy(db: Session, strategy_id: int) -> Strategy:
    print(f"Fetching strategy with ID: {strategy_id}")
    strategy = db.get(Strategy, strategy_id)
    if strategy is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return strategy


def _to_response(backtest: BacktestResult, strategy_name: str) -> BacktestResponse:
    return BacktestResponse(
        id=backtest.id,
        strategy_id=backtest.strategy_id,
        strategy_name=strategy_name,
        test_time=backtest.test_time,
        dataset_name=backtest.dataset_name,
        results=backtest.results,
    )


@router.get("", response_model=list[BacktestResponse])
def list_backtests(db: Session = Depends(get_db)) -> list[BacktestResponse]:
    rows = (
        db.execute(
            select(BacktestResult, Strategy.name)
            .join(Strategy, BacktestResult.strategy_id == Strategy.id)
            .order_by(BacktestResult.id.asc())
        )
        .all()
    )
    return [_to_response(backtest=row[0], strategy_name=row[1]) for row in rows]


@router.get("/{backtest_id}", response_model=BacktestResponse)
def get_backtest(backtest_id: int, db: Session = Depends(get_db)) -> BacktestResponse:
    row = db.execute(
        select(BacktestResult, Strategy.name)
        .join(Strategy, BacktestResult.strategy_id == Strategy.id)
        .where(BacktestResult.id == backtest_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest result not found")
    return _to_response(backtest=row[0], strategy_name=row[1])


@router.post("", response_model=BacktestResponse, status_code=status.HTTP_201_CREATED)
def create_backtest(payload: BacktestCreateRequest, db: Session = Depends(get_db)) -> BacktestResponse:
    strategy = _require_strategy(db, payload.strategy_id)

    backtest = BacktestResult(
        strategy_id=payload.strategy_id,
        dataset_name=payload.dataset_name,
        results=payload.results,
    )
    db.add(backtest)
    db.commit()
    db.refresh(backtest)
    return _to_response(backtest=backtest, strategy_name=strategy.name)


@router.put("/{backtest_id}", response_model=BacktestResponse)
def update_backtest(
    backtest_id: int,
    payload: BacktestUpdateRequest,
    db: Session = Depends(get_db),
) -> BacktestResponse:
    backtest = db.get(BacktestResult, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest result not found")

    strategy_name = backtest.strategy.name

    if payload.strategy_id is not None:
        strategy = _require_strategy(db, payload.strategy_id)
        backtest.strategy_id = payload.strategy_id
        strategy_name = strategy.name
    if payload.dataset_name is not None:
        backtest.dataset_name = payload.dataset_name
    if payload.results is not None:
        backtest.results = payload.results

    db.commit()
    db.refresh(backtest)
    return _to_response(backtest=backtest, strategy_name=strategy_name)


@router.delete("/{backtest_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_backtest(backtest_id: int, db: Session = Depends(get_db)) -> None:
    backtest = db.get(BacktestResult, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest result not found")

    db.delete(backtest)
    db.commit()


class RunBacktestRequest(BaseModel):
    strategy_id: int
    dataset_name: str
    money_amount: float = 0.0
    money_symbol: str = 'USDT'

class RunBacktestResponse(BaseModel):
    id: int
    strategy_id: int
    strategy_name: str
    test_time: datetime
    dataset_name: str
    overall_profit_percent: float
    mean_monthly_profit_percent: float
    max_drawdown_percent: float
    win_rate_percent: float


@router.post("/run", response_model=RunBacktestResponse, status_code=status.HTTP_201_CREATED)
def run_backtest(
    strategy_id: int = Form(...),
    money_amount: float = Form(0.0),
    money_symbol: str = Form('USDT'),
    dataset_name: str | None = Form(None),
    dataset_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> RunBacktestResponse:
    strategy_db = _require_strategy(db, strategy_id)

    try:
        print(f"Creating runtime strategy for '{strategy_db.name}'")
        runtime_strategy = _strategy_factory.create(strategy_db.name)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))

    engine = BacktestEngine()
    engine_result = engine.run(runtime_strategy, dataset_file, money_amount, money_symbol)

    resolved_dataset_name = dataset_name or dataset_file.filename or f"manual-{datetime.utcnow().isoformat()}"

    results_json = json.dumps({
        'overall_profit_percent': engine_result.overall_profit_percent,
        'mean_monthly_profit_percent': engine_result.mean_monthly_profit_percent,
        'max_drawdown_percent': engine_result.max_drawdown_percent,
        'win_rate_percent': engine_result.win_rate_percent,
    })

    backtest = BacktestResult(
        strategy_id=strategy_id,
        dataset_name=resolved_dataset_name,
        results=results_json,
    )
    db.add(backtest)
    db.commit()
    db.refresh(backtest)

    return RunBacktestResponse(
        id=backtest.id,
        strategy_id=backtest.strategy_id,
        strategy_name=strategy_db.name,
        test_time=backtest.test_time,
        dataset_name=backtest.dataset_name,
        overall_profit_percent=engine_result.overall_profit_percent,
        mean_monthly_profit_percent=engine_result.mean_monthly_profit_percent,
        max_drawdown_percent=engine_result.max_drawdown_percent,
        win_rate_percent=engine_result.win_rate_percent,
    )


run_router = APIRouter(prefix="/backtest", tags=["Backtests"])


@run_router.post("", response_model=RunBacktestResponse, status_code=status.HTTP_201_CREATED)
def run_backtest_v2(
    strategy_id: int = Form(...),
    money_amount: float = Form(0.0),
    symbol: str = Form(''),
    exchange: str = Form(''),
    dataset_name: str | None = Form(None),
    dataset_file: UploadFile = File(...),
    parameters: str | None = Form(None),
    db: Session = Depends(get_db),
) -> RunBacktestResponse:
    strategy_db = _require_strategy(db, strategy_id)

    try:
        runtime_strategy = _strategy_factory.create(strategy_db.name)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))

    engine = BacktestEngine()
    engine_result = engine.run(runtime_strategy, dataset_file, money_amount, symbol or 'USDT')

    label_parts = [p for p in [symbol, exchange, dataset_name or dataset_file.filename] if p]
    resolved_dataset_name = '_'.join(label_parts) if label_parts else f"manual-{datetime.utcnow().isoformat()}"

    results_json = json.dumps({
        'overall_profit_percent': engine_result.overall_profit_percent,
        'mean_monthly_profit_percent': engine_result.mean_monthly_profit_percent,
        'max_drawdown_percent': engine_result.max_drawdown_percent,
        'win_rate_percent': engine_result.win_rate_percent,
    })

    backtest = BacktestResult(
        strategy_id=strategy_id,
        dataset_name=resolved_dataset_name,
        results=results_json,
    )
    db.add(backtest)
    db.commit()
    db.refresh(backtest)

    return RunBacktestResponse(
        id=backtest.id,
        strategy_id=backtest.strategy_id,
        strategy_name=strategy_db.name,
        test_time=backtest.test_time,
        dataset_name=backtest.dataset_name,
        overall_profit_percent=engine_result.overall_profit_percent,
        mean_monthly_profit_percent=engine_result.mean_monthly_profit_percent,
        max_drawdown_percent=engine_result.max_drawdown_percent,
        win_rate_percent=engine_result.win_rate_percent,
    )

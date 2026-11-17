"""Level-3 (leaf) tasks of the mining comparison: calibration and the authoritative backtest.

Both operate on the single finalist rule an arm hands in -- never on the inner search loop,
which stays in-process inside run_arm. calibrate_finalist re-tunes the finalist's numbers with
Optuna (arms 3/4/5); backtest_finalist runs the platform BacktestEngine over the full window
and persists a BacktestResult, exactly as the existing backtest_task does.
"""
from __future__ import annotations

import json
from datetime import datetime

from celery import shared_task
from sqlalchemy import select

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.platform_bridge as pb
import app.strategies.genetic.strategy_gp as sg
from app.backtest.engine import BacktestEngine
from app.background_tasks.tasks.backtest import (
    _compute_buy_and_hold,
    _downsample,
    approximate_wait_time,
)
from app.data.data_provider import GapsFoundError, OHLCVDataProvider
from app.data.database.models import ArmResult, BacktestResult, Strategy as StrategyModel
from app.data.database.session import db_session
from app.strategies.genetic.rule_strategy import RuleStrategy


def _as_dt(value) -> datetime:
    """Accept a datetime or an ISO-8601 string (the form that survives the Celery broker)."""
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _json_safe(obj):
    """Round-trip through JSON, coercing numpy scalars/arrays so a summary stores cleanly."""
    import numpy as np

    def conv(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")

    return json.loads(json.dumps(obj, default=conv))


def _build_data(symbol: str, exchange: str, timeframe: str, start, end):
    """Drain the DB candles into the numpy `data` dict, the notebook's candle-loading path.

    Raises GapsFoundError if the backfill left holes (callers retry)."""
    probe = RuleStrategy(symbol=symbol, exchange=exchange, timeframe=timeframe)
    provider = OHLCVDataProvider(probe, _as_dt(start), _as_dt(end), exchange, symbol, timeframe)
    data, stamps = pb.bars_from_provider(provider, probe)
    return data, stamps


@shared_task(bind=True, name="calibrate_finalist", max_retries=5)
def calibrate_finalist(self, payload):
    """Re-tune the finalist's free numbers with Optuna (arms 3/4/5). Passthrough when there is
    no rule or the rule has no free slots. Returns the payload with a polished `rule`."""
    if not payload or payload.get("rule") is None:
        return payload

    lab.BARS_PER_YEAR = pb.BARS_PER_YEAR[payload["timeframe"]]
    lab.COST_BPS = payload["cost_bps"]

    tree = sg.parse(payload["rule"], allow_slots=False)
    template, _ = sg.lift(tree)
    if not sg.slot_names(template):
        return payload

    try:
        data, _ = _build_data(payload["symbol"], payload["exchange"], payload["timeframe"],
                              payload["start_date"], payload["end_date"])
    except GapsFoundError as exc:
        wait = sum(approximate_wait_time(g[0], g[1], exc.timeframe) for g in exc.gaps)
        raise self.retry(exc=exc, countdown=wait)

    ledger = lab.Ledger(data, payload["budget"], n_search=payload["n_search"])
    _, values = lab.tune(template, ledger, payload["k"])
    bound = sg.bind(template, values)
    return {**payload, "rule": sg.to_text(bound), "calibrated": True}


@shared_task(bind=True, name="backtest_finalist", max_retries=5)
def backtest_finalist(self, payload):
    """Run the finalist rule through the platform BacktestEngine over the full window and save a
    BacktestResult, then link it to the arm's ArmResult row. Reuses the save path of
    backtest_task (results/equity/buy-and-hold). Passthrough when no rule passed the gates."""
    if not payload or payload.get("rule") is None:
        return {"arm": (payload or {}).get("arm"), "seed": (payload or {}).get("seed"),
                "rule": None, "backtest_id": None,
                "arm_result_id": (payload or {}).get("arm_result_id")}

    symbol, exchange, timeframe = payload["symbol"], payload["exchange"], payload["timeframe"]
    rule, money = payload["rule"], payload["money"]

    strategy = RuleStrategy(rule=rule, symbol=symbol, exchange=exchange, timeframe=timeframe)
    try:
        provider = OHLCVDataProvider(strategy, _as_dt(payload["start_date"]),
                                     _as_dt(payload["end_date"]), exchange, symbol, timeframe)
    except GapsFoundError as exc:
        wait = sum(approximate_wait_time(g[0], g[1], exc.timeframe) for g in exc.gaps)
        raise self.retry(exc=exc, countdown=wait)

    name = f"mining-run-{payload['run_id']}-arm{payload['arm']}-seed{payload['seed']}"
    dataset_name = f"{symbol}_{exchange}_{timeframe}_{payload['start_date']}_{payload['end_date']}"
    with db_session() as db:
        strat = db.execute(
            select(StrategyModel).where(StrategyModel.name == name)
        ).scalar_one_or_none()
        if strat is None:
            strat = StrategyModel(name=name, description=rule)
            db.add(strat)
            db.flush()
        else:
            strat.description = rule
        backtest = BacktestResult(
            strategy_id=strat.id, dataset_name=dataset_name,
            status="running", celery_task_id=self.request.id,
        )
        db.add(backtest)
        db.flush()
        backtest_id = backtest.id

    engine = BacktestEngine()
    result = engine.run(strategy, provider, money)
    buy_and_hold = _compute_buy_and_hold(provider, money)
    result.buy_and_hold_curve = buy_and_hold

    results_json = result.to_json()
    equity_json = json.dumps(_downsample(result.equity_curve))
    buy_and_hold_json = json.dumps(_downsample(buy_and_hold))

    with db_session() as db:
        backtest = db.get(BacktestResult, backtest_id)
        backtest.results = results_json
        backtest.equity_curve = equity_json
        backtest.buy_and_hold_curve = buy_and_hold_json
        backtest.status = "done"
        arm_result = db.get(ArmResult, payload["arm_result_id"]) if payload.get("arm_result_id") else None
        if arm_result is not None:
            arm_result.backtest_result_id = backtest_id

    return {"arm": payload["arm"], "seed": payload["seed"], "rule": rule,
            "backtest_id": backtest_id, "arm_result_id": payload.get("arm_result_id")}

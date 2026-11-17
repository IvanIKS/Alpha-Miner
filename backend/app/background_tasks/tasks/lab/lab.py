"""Master (level-1) task of the alpha-mining comparison and its chord callback.

compare_strategy_mining builds a MiningRun, then dispatches a Celery canvas:

    chain(
        backfill_ohlcv  (once, shared candles),
        chord(
            group(  run_arm | [calibrate_finalist] | backtest_finalist  per (arm, seed) ),
            aggregate_comparison,
        ),
    )

The search of each arm runs in-process (see arms.run_arm); the chord members run in parallel
across worker slots, so concurrency on the worker must be >= number of (arm, seed) members.
"""
from __future__ import annotations

from celery import chain, chord, group, shared_task
from sqlalchemy import select

from app.background_tasks.tasks.backfill import backfill_ohlcv_task
from app.background_tasks.tasks.lab.arms import ARM_NAMES, CALIBRATION_ARMS, run_arm
from app.background_tasks.tasks.lab.finalize import _as_dt, backtest_finalist, calibrate_finalist
from app.data.database.base import Base
from app.data.database.models import ArmResult, MiningRun
from app.data.database.session import db_session, engine

# Defaults mirror strategies_comparison.ipynb. Dates are ISO strings so they survive the broker.
DEFAULTS = {
    "symbol": "BTC/USDT", "exchange": "binance", "timeframe": "1h",
    "start_date": "2022-01-01T00:00:00+00:00", "end_date": "2026-10-01T00:00:00+00:00",
    "holdout_fraction": 0.25, "budget": 2000, "pop_size": 30, "k": 10,
    "cost_bps": 12.0, "money": 10_000.0, "model_id": "claude-haiku-5-5",
    "arms": [1, 2, 3, 4, 5], "seeds": [0],
    # Arm 5 (GP + Optuna) searches structure x trials, so the notebook gives it 25x the
    # per-arm backtest budget; applied in run_arm._effective_budget.
    "arm5_budget_multiplier": 25,
}

# Summary metrics lifted from each ArmResult.summary into the run's comparison table.
_SUMMARY_KEYS = ("train sharpe", "val sharpe", "hold-out sharpe", "Q",
                 "DSR null V", "DSR ledger V", "budget used", "structures", "rules")


def _ensure_tables() -> None:
    """Create mining_runs / arm_results if a worker runs before the API seeded them."""
    Base.metadata.create_all(
        bind=engine, tables=[MiningRun.__table__, ArmResult.__table__]
    )


@shared_task(bind=True, name="compare_strategy_mining")
def compare_strategy_mining(self, config=None):
    """Launch the full comparison. `config` overrides any of DEFAULTS. Returns the run id;
    does not block on the arms."""
    _ensure_tables()
    cfg = {**DEFAULTS, **(config or {})}
    arms = [int(a) for a in cfg["arms"]]
    seeds = [int(s) for s in cfg["seeds"]]
    cfg["arms"], cfg["seeds"] = arms, seeds

    with db_session() as db:
        run = MiningRun(
            symbol=cfg["symbol"], exchange=cfg["exchange"], timeframe=cfg["timeframe"],
            start_date=_as_dt(cfg["start_date"]), end_date=_as_dt(cfg["end_date"]),
            holdout_fraction=cfg["holdout_fraction"], budget=cfg["budget"],
            pop_size=cfg["pop_size"], k=cfg["k"], cost_bps=cfg["cost_bps"], money=cfg["money"],
            model_id=cfg["model_id"], arms=arms, seeds=seeds,
            status="running", celery_task_id=self.request.id,
        )
        db.add(run)
        db.flush()
        run_id = run.id
        for arm in arms:
            for seed in seeds:
                db.add(ArmResult(mining_run_id=run_id, arm=arm, arm_name=ARM_NAMES.get(arm),
                                 seed=seed, status="pending"))

    members = []
    for arm in arms:
        for seed in seeds:
            sig = run_arm.s(run_id, arm, seed, cfg)
            if arm in CALIBRATION_ARMS:
                sig |= calibrate_finalist.s()
            sig |= backtest_finalist.s()
            members.append(sig)

    canvas = chain(
        backfill_ohlcv_task.si(cfg["exchange"], cfg["symbol"],
                               _as_dt(cfg["start_date"]), _as_dt(cfg["end_date"]), cfg["timeframe"]),
        chord(group(members), aggregate_comparison.s(run_id)),
    )
    canvas.apply_async()

    return {"run_id": run_id, "status": "running", "arms": arms, "seeds": seeds}


@shared_task(bind=True, name="aggregate_comparison")
def aggregate_comparison(self, arm_returns, run_id):
    """Chord callback: assemble the comparison table from the run's ArmResult rows and mark the
    MiningRun done. `arm_returns` is the list of backtest_finalist results (not used directly)."""
    with db_session() as db:
        run = db.get(MiningRun, run_id)
        arm_results = db.execute(
            select(ArmResult).where(ArmResult.mining_run_id == run_id)
            .order_by(ArmResult.arm.asc(), ArmResult.seed.asc())
        ).scalars().all()

        table = []
        for ar in arm_results:
            row = {"arm": ar.arm, "arm_name": ar.arm_name, "seed": ar.seed,
                   "status": ar.status, "rule": ar.rule, "minutes": ar.minutes,
                   "model_calls": ar.model_calls, "backtests_used": ar.backtests_used,
                   "backtest_result_id": ar.backtest_result_id}
            if ar.summary:
                row.update({key: ar.summary[key] for key in _SUMMARY_KEYS if key in ar.summary})
            table.append(row)

        if run is not None:
            run.status = "done"
            run.summary = {"table": table}

    return {"run_id": run_id, "status": "done", "arms": len(table)}

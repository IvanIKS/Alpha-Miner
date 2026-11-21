"""Level-2 tasks: one run_arm per (arm, seed). Each runs a whole alpha-mining search
in-process -- the ledger loop is a tight numpy/Optuna/LLM inner loop that cannot be farmed
out per call -- then records its finalist in an ArmResult row and hands the rule downstream
to the calibration/backtest leaf tasks.

The arm entry points and their call signatures mirror strategies_comparison.ipynb exactly.
"""
from __future__ import annotations

import time
from pathlib import Path

from celery import shared_task
from sqlalchemy import select

import app.strategies.genetic.experiment.arm1_llm_fully as arm1_llm_fully
import app.strategies.genetic.experiment.arm2_gp_llm_tune as arm2_gp_llm_tune
import app.strategies.genetic.experiment.arm3_llm_gen_optuna as arm3_llm_gen_optuna
import app.strategies.genetic.experiment.arm4_llm_gen_optuna_finalists_only as arm4_finalists
import app.strategies.genetic.experiment.arm5_demo_gp_optuna as arm5_demo_gp_optuna
import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_models as lm
import app.strategies.genetic.platform_bridge as pb
from app.background_tasks import retry
from app.background_tasks.tasks.backtest import approximate_wait_time
from app.background_tasks.tasks.lab.finalize import _build_data, _json_safe, persist_ledger_backtests
from app.data.data_provider import GapsFoundError
from app.data.database.models import ArmResult
from app.data.database.session import db_session

ARM_NAMES = {
    1: "1: LLM only",
    2: "2: GP + LLM tune",
    3: "3: LLM + Optuna",
    4: "4: LLM + Optuna, finalists only",
    5: "5: GP + Optuna",
}
LLM_ARMS = {1, 2, 3, 4}            # need a ChatModel built in the worker
CALIBRATION_ARMS = {3, 4, 5}       # get the extra Optuna calibrate_finalist step


def _model_log_path(run_id, arm, seed) -> Path:
    """Stable per-(run, arm, seed) JSONL log for the arm's LLM calls. Anchored to the llm
    package's runs/ dir so it is the same across a worker's retries regardless of CWD. A
    retried arm resumes from this log (RecordedModel resume mode) instead of paying for the
    calls again; only the interrupted/new calls reach the API."""
    return Path(lm.__file__).parent / "runs" / f"mining-run-{run_id}" / f"arm{arm}_seed{seed}.jsonl"


def _effective_budget(arm, cfg):
    """Scale the per-arm budget by that arm's `budget_multipliers` coefficient. The coefficients
    are tuned empirically so every arm finishes in roughly the same wall-clock time."""
    multiplier = cfg.get("budget_multipliers", {}).get(f"arm{arm}", 1)
    return int(cfg["budget"] * multiplier)


def _run_search(arm, data, cfg, model):
    """Dispatch to the arm's entry point; returns (ledger, model_calls)."""
    budget, seed, n_search = cfg["budget"], cfg["seed"], cfg["n_search"]
    pop, k = cfg["pop_size"], cfg["k"]

    if arm == 1:
        ledger, gen = arm1_llm_fully.fully_llm_search(
            data, model=model, seed=seed, budget=budget, n_search=n_search)
        return ledger, gen.stats.get("calls", 0)
    if arm == 2:
        ledger, _generations, stats = arm2_gp_llm_tune.evolve(
            data, model=model, seed=seed, budget=budget, pop_size=pop, k=k, n_search=n_search)
        return ledger, stats.get("calls", 0)
    if arm == 3:
        ledger, gen = arm3_llm_gen_optuna.llm_optuna_optimisation(
            data, model=model, seed=seed, budget=budget, n_search=n_search)
        return ledger, gen.stats.get("calls", 0)
    if arm == 4:
        ledger, gen = arm4_finalists.llm_optuna_optimisation_finalists_only(
            data, model=model, seed=seed, budget=budget, n_search=n_search)
        return ledger, gen.stats.get("calls", 0)
    if arm == 5:
        ledger, _generations = arm5_demo_gp_optuna.evolve(
            data, seed=seed, budget=budget, pop_size=pop, k=k, n_search=n_search)
        return ledger, 0
    raise ValueError(f"unknown arm {arm}")


@shared_task(bind=True, name="run_arm", max_retries=5)
def run_arm(self, run_id, arm, seed, config):
    """Run one arm's mining search, persist its ArmResult, and return the finalist payload for
    the calibration/backtest chain."""
    cfg = {**config, "run_id": run_id, "arm": arm, "seed": seed}
    lab.BARS_PER_YEAR = pb.BARS_PER_YEAR[cfg["timeframe"]]
    lab.COST_BPS = cfg["cost_bps"]

    with db_session() as db:
        ar = db.execute(
            select(ArmResult).where(
                ArmResult.mining_run_id == run_id,
                ArmResult.arm == arm,
                ArmResult.seed == seed,
            )
        ).scalar_one_or_none()
        arm_result_id = ar.id if ar else None
        if ar is not None:
            ar.status = "running"
            ar.celery_task_id = self.request.id

    try:
        data, _ = _build_data(cfg["symbol"], cfg["exchange"], cfg["timeframe"],
                              cfg["start_date"], cfg["end_date"])
    except GapsFoundError as exc:
        wait = sum(approximate_wait_time(g[0], g[1], exc.timeframe) for g in exc.gaps)
        raise self.retry(exc=exc, countdown=wait)

    n = len(data["close"])
    n_search = int(n * (1 - cfg["holdout_fraction"]))
    cfg["n_search"] = n_search
    cfg["budget"] = _effective_budget(arm, cfg)

    model = None
    if arm in LLM_ARMS:
        # Resume from the log so a retried arm replays the calls made before the connection
        # dropped (free) and only re-issues the interrupted/remaining ones.
        model = lm.RecordedModel(lm.ClaudeModel(cfg["model_id"]),
                                 _model_log_path(run_id, arm, seed), mode="resume")

    started = time.perf_counter()
    try:
        ledger, model_calls = _run_search(arm, data, cfg, model)
    except retry.connection_errors() as exc:
        # Unstable connection to the LLM mid-search: wait and retry the whole arm.
        countdown = retry.retry_countdown(self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)
    except Exception:
        with db_session() as db:
            ar = db.get(ArmResult, arm_result_id) if arm_result_id else None
            if ar is not None:
                ar.status = "failed"
        raise
    minutes = (time.perf_counter() - started) / 60

    summary = lab.summarize(data, ledger)
    rule = summary.get("rule")

    # Persist one BacktestResult per distinct rule the search backtested (not probes).
    saved = persist_ledger_backtests(ledger, {**cfg, "rule": rule}, cfg["money"])

    with db_session() as db:
        ar = db.get(ArmResult, arm_result_id) if arm_result_id else None
        if ar is not None:
            ar.status = "done"
            ar.rule = rule
            ar.summary = _json_safe(summary)
            ar.minutes = minutes
            ar.model_calls = model_calls
            ar.backtests_used = ledger.used

    return {
        "run_id": run_id, "arm": arm, "seed": seed, "arm_result_id": arm_result_id,
        "rule": rule, "ledger_backtests_saved": saved,
        "symbol": cfg["symbol"], "exchange": cfg["exchange"], "timeframe": cfg["timeframe"],
        "start_date": cfg["start_date"], "end_date": cfg["end_date"],
        "n_search": n_search, "budget": cfg["budget"], "k": cfg["k"],
        "cost_bps": cfg["cost_bps"], "money": cfg["money"],
    }

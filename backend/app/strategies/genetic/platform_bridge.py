"""Bridge between the strategy grammar and the platform: candles from its data provider, a rule
run through its BacktestEngine, the same rule in the fast engine, and their comparison.

Only run_platform imports the platform (app.*); everything else works without it.
"""
from __future__ import annotations

import time

import numpy as np

import app.strategies.genetic.strategy_gp as sg

FIELDS = ("open", "high", "low", "close", "volume")
BARS_PER_YEAR = {"1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040, "30m": 17_520,
                 "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460, "8h": 1_095, "12h": 730,
                 "1d": 365}                                      # crypto trades around the clock


def bars_from_provider(provider, strategy):
    """Drain a platform data provider into numpy arrays: (data, timestamps). Uses only
    get_market_data(strategy), the call BacktestEngine itself makes, until it returns None.
    A provider is used up afterwards; build a new one for a backtest."""
    rows = []
    while True:
        bar = provider.get_market_data(strategy)
        if bar is None:
            break
        rows.append(bar)
    if not rows:
        raise ValueError("the data provider returned no bars")
    missing = [f for f in FIELDS if f not in rows[0]]
    if missing:
        raise KeyError(f"bars have no fields {missing}; they have {sorted(rows[0])}")
    data = {f: np.array([float(row[f]) for row in rows]) for f in FIELDS}
    return data, [row.get("timestamp") for row in rows]


def run_platform(rule, make_provider, money=10_000.0, **strategy_options):
    """Run one rule (text with every number set) through the platform's BacktestEngine.
    make_provider(strategy) must return a fresh data provider. Returns (BacktestResult, seconds)."""
    from app.backtest.engine import BacktestEngine
    from app.strategies.genetic.rule_strategy import RuleStrategy

    strategy = RuleStrategy(rule=rule, **strategy_options)
    provider = make_provider(strategy)
    started = time.perf_counter()
    result = BacktestEngine().run(strategy, provider, money)
    return result, time.perf_counter() - started


def fast_equity(data, rule, cost_bps=0.0):
    """Equity curve (starting at 1) and positions of a rule in the fast engine: the position is
    decided at a close and held over the next bar, as the platform buys and sells at the close."""
    tree = sg.parse(rule, allow_slots=False) if isinstance(rule, str) else rule
    pos = sg.positions(*sg.Evaluator(data).signals(tree))
    return np.cumprod(1.0 + sg.net_returns(pos, data["close"], cost_bps)), pos


def window_metrics(equity, start=0, bars_per_year=8_760):
    """Return, annualized Sharpe and maximum drawdown of an equity curve from bar `start` on
    (the bar before `start` is the base, so the first return is that of bar `start`)."""
    e = np.asarray(equity, dtype=float)[max(start - 1, 0):]
    r = e[1:] / e[:-1] - 1.0
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    peak = np.maximum.accumulate(e)
    return {"return %": float(e[-1] / e[0] - 1.0) * 100,
            "sharpe": float(r.mean()) / sd * float(np.sqrt(bars_per_year)) if sd > 0 else 0.0,
            "max DD %": float(np.max(1.0 - e / peak)) * 100}


def parity(data, rule, result, money):
    """Compare a platform run of a rule with the fast engine at zero cost: closed trades,
    profit and the largest relative gap between the two equity curves."""
    fast, pos = fast_equity(data, rule, 0.0)
    plat = np.asarray(result.equity_curve, dtype=float) / money
    n = min(len(fast), len(plat))
    return {"rule": rule, "bars": len(fast), "bars platform": len(plat),
            "closed trades fast": sg.n_trades(pos) - int(pos[-1] == 1.0),
            "closed trades platform": len(result.trade_history),
            "profit % fast": float(fast[n - 1] - 1.0) * 100, "profit % platform": float(plat[n - 1] - 1.0) * 100,
            "max equity gap %": float(np.max(np.abs(plat[:n] / fast[:n] - 1.0))) * 100}


def prefixed(prefix, metrics):
    return {f"{prefix}: {k}": v for k, v in metrics.items()}

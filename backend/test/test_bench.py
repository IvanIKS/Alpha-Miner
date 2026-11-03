"""Checks for arm 0 (arm0_random.py), the GP arm's search window (demo_gp_optuna.evolve), the arm
summary (lab.summarize), the bench's timeframe and cost settings, and the platform bridge
(platform_bridge.py) with fakes in place of the platform.

Run: pytest -q test_bench.py
"""
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import app.strategies.genetic.experiment.arm0_random as arm0_random
import app.strategies.genetic.experiment.arm5_demo_gp_optuna as arm5
import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.platform_bridge as pb
import app.strategies.genetic.strategy_gp as sg

DATA = lab.synthetic(seed=3, edge=lab.EDGE, n=6000)
N_SEARCH = 5000


# ------------------------------------------------------------ the arms --
def test_random_search_spends_the_budget_and_repeats_by_seed():
    a = arm0_random.random_search(DATA, 40, seed=1, n_search=N_SEARCH)
    b = arm0_random.random_search(DATA, 40, seed=1, n_search=N_SEARCH)
    c = arm0_random.random_search(DATA, 40, seed=2, n_search=N_SEARCH)
    assert a.used == 40 and sorted(a.rows) == sorted(b.rows) != sorted(c.rows)
    assert all(r["role"] == "candidate" for r in a.rows.values()) and a.structures


def test_evolve_respects_the_search_window_and_budget():
    ledger, generations = arm5.evolve(DATA, seed=1, budget=60, pop_size=8, k=3, n_search=N_SEARCH)
    assert ledger.n_search == N_SEARCH and 0 < ledger.used <= 60 and generations >= 0


def test_summarize_reports_the_chosen_rule_without_spending():
    ledger = arm0_random.random_search(DATA, 80, seed=4, n_search=N_SEARCH)
    used = ledger.used
    s = lab.summarize(DATA, ledger, reference=lab.PLANTED_RULE)
    assert ledger.used == used
    assert {"structures", "rules", "budget used", "rule"} <= set(s)
    if s["rule"] is not None:
        sg.parse(s["rule"], allow_slots=False)
        assert {"train sharpe", "val sharpe", "Q", "hold-out sharpe", "reference hold-out sharpe"} <= set(s)


def test_bench_follows_the_configured_timeframe_and_costs(monkeypatch):
    ledger = lab.Ledger(DATA, 5, n_search=N_SEARCH)
    pos = sg.positions(*ledger.search.signals(sg.parse(sg.CLASSICS["rsi_reversion"])))
    train = ledger.segments["train"]
    hourly = lab.segment_stats(pos, ledger.close, train)
    monkeypatch.setattr(lab, "BARS_PER_YEAR", 365)
    monkeypatch.setattr(lab, "COST_BPS", 0)
    daily = lab.segment_stats(pos, ledger.close, train)
    assert daily["gross"] == pytest.approx(hourly["gross"] * (365 / 8_760) ** 0.5)
    assert daily["sharpe"] == pytest.approx(daily["gross"])            # no costs now


# --------------------------------------------------------- the bridge --
class FakeProvider:
    def __init__(self, rows):
        self.rows = list(rows)

    def get_market_data(self, strategy):
        return self.rows.pop(0) if self.rows else None


def test_bars_from_provider_reads_every_bar():
    rows = [{"timestamp": i, "open": 1.0 + i, "high": 2.0 + i, "low": 0.5 + i, "close": 1.5 + i, "volume": 10.0}
            for i in range(5)]
    data, stamps = pb.bars_from_provider(FakeProvider(rows), strategy=None)
    assert stamps == list(range(5)) and data["close"].tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]
    with pytest.raises(KeyError, match="volume"):
        pb.bars_from_provider(FakeProvider([{"timestamp": 0, "open": 1, "high": 1, "low": 1, "close": 1}]), None)
    with pytest.raises(ValueError):
        pb.bars_from_provider(FakeProvider([]), None)


def test_window_metrics_on_a_known_curve():
    equity = np.array([1.0, 1.1, 1.21, 1.089, 1.1979])
    m = pb.window_metrics(equity, start=2, bars_per_year=4)
    assert m["return %"] == pytest.approx((1.1979 / 1.1 - 1) * 100)
    assert m["max DD %"] == pytest.approx((1 - 1.089 / 1.21) * 100)
    r = np.array([1.21 / 1.1, 1.089 / 1.21, 1.1979 / 1.089]) - 1
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * 2)


def test_parity_of_a_matching_equity_curve():
    data = {k: v[:3000] for k, v in DATA.items()}
    rule = sg.CLASSICS["rsi_reversion"]
    fast, pos = pb.fast_equity(data, rule)
    closed = sg.n_trades(pos) - int(pos[-1] == 1.0)
    result = SimpleNamespace(equity_curve=list(fast * 10_000.0), trade_history=[0.0] * closed)
    p = pb.parity(data, rule, result, 10_000.0)
    assert p["max equity gap %"] == pytest.approx(0.0, abs=1e-9)
    assert p["closed trades fast"] == p["closed trades platform"] and p["bars"] == p["bars platform"] == 3000


def test_costs_lower_the_fast_equity():
    free, pos = pb.fast_equity(DATA, sg.CLASSICS["ma_cross"], 0.0)
    paid, _ = pb.fast_equity(DATA, sg.CLASSICS["ma_cross"], 12)
    assert sg.n_trades(pos) == 0 or paid[-1] < free[-1]


def test_run_platform_drives_the_platform_engine(monkeypatch):
    from test.test_strategy_gp import _install_platform_stubs
    _install_platform_stubs()
    seen = {}

    class FakeEngine:
        def run(self, strategy, provider, money):
            seen.update(rule=strategy.rule, provider=provider, money=money)
            return "result"

    engine = types.ModuleType("app.backtest.engine")
    engine.BacktestEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "app.backtest", types.ModuleType("app.backtest"))
    monkeypatch.setitem(sys.modules, "app.backtest.engine", engine)
    rule = sg.CLASSICS["ibs_reversion"]
    result, seconds = pb.run_platform(rule, lambda s: ("provider for", s.symbol), 500.0, symbol="ETH/USDT")
    assert result == "result" and seconds >= 0
    assert seen == {"rule": rule, "provider": ("provider for", "ETH/USDT"), "money": 500.0}

"""Arm 5 smoke test: GP builds rule structures, Optuna fills their slots, every backtest
is logged once in a ledger.

Synthetic hourly bars: 'planted' carries a symmetric mean-reversion edge (drift up on the
bar after RSI(14) < 30, down after RSI(14) > 70, so buy-and-hold earns nothing from it);
'noise' is the same random path without it. Fitness and the deflated Sharpe ratio use
the search window only; the hold-out window is read once, at the end.

Run: python demo_gp_optuna.py
"""
from __future__ import annotations

import hashlib
import math
import random
import zlib
from collections import deque
from statistics import NormalDist

import numpy as np
import optuna
from deap import base, creator, gp, tools

import strategy_gp as sg

BARS_PER_YEAR = 24 * 365
COST_BPS = 12            # fee + slippage per side
MIN_TRADES = 20          # rules trading less on the search window get fitness -1
N_SEARCH, N_HOLD = 20_000, 8_000
EDGE = 0.15              # planted drift, in units of bar volatility
PLANTED_RULE = "long_flat(osc_below(rsi(close, 14), 30), osc_above(rsi(close, 14), 30))"

optuna.logging.set_verbosity(optuna.logging.ERROR)
if not hasattr(creator, "RuleFitness"):
    creator.create("RuleFitness", base.Fitness, weights=(1.0,))
    creator.create("RuleTree", gp.PrimitiveTree, fitness=creator.RuleFitness, params=list)


def synthetic(seed, edge, n=N_SEARCH + N_HOLD, sigma=0.01):
    """Tick-rounded bars; drift +edge*sigma after RSI(14, Cutler) < 30, -edge*sigma after > 70."""
    rng = np.random.default_rng(seed)
    close = np.empty(n)
    close[0] = 30000.0
    gains, losses = deque(maxlen=14), deque(maxlen=14)
    for t in range(1, n):
        g, lo = sum(gains), sum(losses)
        rsi = 100 * g / (g + lo) if len(gains) == 14 and g + lo > 0 else 50.0
        drift = edge * sigma * (int(rsi < 30) - int(rsi > 70))
        close[t] = round(close[t - 1] * math.exp(drift + sigma * rng.standard_normal()), 2)
        d = close[t] - close[t - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, sigma / 2, (2, n)))
    high = np.round(np.maximum(open_, close) * np.exp(wick[0]), 2)
    low = np.round(np.minimum(open_, close) * np.exp(-wick[1]), 2)
    return {"open": open_, "high": high, "low": low, "close": close,
            "volume": np.round(rng.lognormal(3, 0.5, n), 3)}


class BudgetExhausted(Exception):
    pass


class Ledger:
    """One row per distinct backtest on the search window; each row spends one unit of budget.
    The same ledger is what DSR, PBO and the effective number of trials are computed from."""

    def __init__(self, data, budget):
        self.budget = budget
        self.search = sg.Evaluator({k: v[:N_SEARCH] for k, v in data.items()})
        self.close = data["close"][:N_SEARCH]
        self.rows = {}
        self.structures = set()               # templates that went through tune()

    def run(self, tree, params):
        bound = sg.bind(tree, params)
        key = sg.to_text(bound, canonical=True)
        row = self.rows.get(key)
        if row is None:
            if len(self.rows) >= self.budget:
                raise BudgetExhausted
            pos = sg.positions(*self.search.signals(bound))
            r = sg.net_returns(pos, self.close, COST_BPS)
            sd = r.std(ddof=1)
            row = self.rows[key] = {
                "rule": bound, "trades": sg.n_trades(pos), "sharpe": sg.sharpe(r, BARS_PER_YEAR),
                "sr_obs": 0.0 if sd == 0 else float(r.mean() / sd),
                "behaviour": hashlib.blake2b(pos.tobytes(), digest_size=8).hexdigest()}
        return row["sharpe"] if row["trades"] >= MIN_TRADES else -1.0


def tune(ind, ledger, k):
    """Optuna fills the slots of one structure: k trials, at most k units of budget."""
    for i, node in enumerate(sg.canonical_slots(ind)):   # W2-only and W1-only copies: one structure
        ind[i] = node                         # PrimitiveTree rejects ind[:] = ...
    ledger.structures.add(sg.to_text(ind))
    names = sg.slot_names(ind)
    if not names:
        ind.params = []
        return ledger.run(ind, [])
    sampler = optuna.samplers.TPESampler(seed=zlib.crc32(sg.to_text(ind).encode()), n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    try:
        study.optimize(lambda trial: ledger.run(ind, sg.suggest(trial, ind)), n_trials=k)
    finally:
        done = [t for t in study.trials if t.value is not None]
        if done:
            best = max(done, key=lambda t: t.value)
            ind.params = [best.params[n] for n in names]
    return max(t.value for t in done)


def evolve(data, seed, budget=2400, pop_size=30, k=10):
    random.seed(seed)
    ledger = Ledger(data, budget)
    tb = base.Toolbox()
    tb.register("mate", gp.cxOnePoint)
    tb.register("mut_uniform", gp.mutUniform, expr=lambda pset, type_: sg.gen_tree(pset, 0, 2, type_), pset=sg.PSET)
    tb.register("mut_node", gp.mutNodeReplacement, pset=sg.PSET)
    for name in ("mate", "mut_uniform", "mut_node"):
        tb.decorate(name, gp.staticLimit(key=sg.limit_excess, max_value=0))

    def fresh():
        while True:
            ind = creator.RuleTree(sg.gen_tree(min_=2, max_=4))
            if not sg.limit_excess(ind):
                return ind

    pop, generations = [fresh() for _ in range(pop_size)], 0
    try:
        for ind in pop:
            ind.fitness.values = (tune(ind, ledger, k),)
        while True:
            generations += 1
            offspring = [tb.clone(i) for i in tools.selTournament(pop, pop_size - 2, tournsize=3)]
            for i in range(1, len(offspring), 2):
                if random.random() < 0.5:
                    offspring[i - 1], offspring[i] = tb.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    mutate = tb.mut_uniform if random.random() < 0.7 else tb.mut_node
                    (offspring[i],) = mutate(offspring[i])
                    del offspring[i].fitness.values
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = (tune(ind, ledger, k),)
            pop = tools.selBest(pop, 2) + offspring
    except BudgetExhausted:
        pass
    return ledger, generations


def deflated_sharpe(sr, n, v, returns):
    """Bailey & Lopez de Prado (2014). sr: per-observation Sharpe of the selected rule,
    n: number of trials, v: variance of the trials' per-observation Sharpe ratios."""
    g, nd = 0.5772156649015329, NormalDist()
    sr0 = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))
    z = (returns - returns.mean()) / returns.std()
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    return nd.cdf((sr - sr0) * math.sqrt(len(returns) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def hold_out(data, rule):
    pos = sg.positions(*sg.Evaluator(data).signals(rule))
    r = sg.net_returns(pos, data["close"], COST_BPS)[N_SEARCH:]
    return sg.sharpe(r, BARS_PER_YEAR), sg.n_trades(pos[N_SEARCH:])


def report(name, data, ledger, generations):
    rows = list(ledger.rows.values())
    valid = [r for r in rows if r["trades"] >= MIN_TRADES]
    best = max(valid, key=lambda r: r["sharpe"])
    search = sg.Evaluator({k: v[:N_SEARCH] for k, v in data.items()})

    def search_returns(rule):
        r = sg.net_returns(sg.positions(*search.signals(rule)), data["close"][:N_SEARCH], COST_BPS)
        return r, float(r.mean() / r.std(ddof=1))

    distinct = {}                     # one Sharpe per distinct position path among tradable rules
    for r in valid:
        distinct.setdefault(r["behaviour"], r["sr_obs"])
    # Two conventions for (N, V). The ledger variance mixes estimation noise with real
    # differences between rules (cost-crushed junk), which raises the bar for everyone.
    conventions = {
        "ledger V, all trials": (len(rows), float(np.var([r["sr_obs"] for r in rows], ddof=1))),
        "null V=1/T, distinct tradable": (len(distinct), 1.0 / (N_SEARCH - 1)),
    }
    r_best, sr_best = search_returns(best["rule"])
    r_ref, sr_ref = search_returns(sg.parse(PLANTED_RULE))
    ho_sharpe, ho_trades = hold_out(data, best["rule"])
    ref_sharpe, _ = hold_out(data, sg.parse(PLANTED_RULE))
    print(f"\n== {name}: {generations} generations, {len(ledger.structures)} structures, {len(rows)} trials, "
          f"{len(distinct)} distinct tradable behaviours")
    print(f"best rule:    {sg.to_text(best['rule'])}")
    print(f"search Sharpe {best['sharpe']:.2f} ({best['trades']} trades), "
          f"hold-out Sharpe {ho_sharpe:.2f} ({ho_trades} trades)")
    print(f"planted rule: search Sharpe {sg.sharpe(r_ref, BARS_PER_YEAR):.2f}, hold-out Sharpe {ref_sharpe:.2f}")
    for label, (n, v) in conventions.items():
        print(f"DSR [{label}, N={n}]: best {deflated_sharpe(sr_best, n, v, r_best):.3f}, "
              f"planted rule {deflated_sharpe(sr_ref, n, v, r_ref):.3f}")


if __name__ == "__main__":
    for name, edge in (("planted", EDGE), ("noise", 0.0)):
        data = synthetic(seed=7, edge=edge)
        ledger, generations = evolve(data, seed=1)
        report(name, data, ledger, generations)

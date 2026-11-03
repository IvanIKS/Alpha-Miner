"""Arm 0: random search over the grammar, the null model of the experiment.

Each draw is a random structure (the same typed generator as GP) with random numbers
(log-uniform windows, grid levels); every distinct rule spends one unit of budget in the
shared ledger, and the rule handed in is chosen by lab.select_final like in every other arm.
If an arm does not beat this one on the hold-out window, its search adds nothing over chance.

Run: python arm0_random.py
"""
from __future__ import annotations

import random

from deap import gp

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.strategy_gp as sg


def random_search(data, budget, seed=0, n_search=lab.N_SEARCH, min_depth=2, max_depth=4, max_draws=None):
    """Spend `budget` backtests on random rules; returns the ledger. The same seed gives the
    same rules."""
    rng = random.Random(seed)
    ledger = lab.Ledger(data, budget, n_search=n_search)
    try:
        for _ in range(max_draws or 50 * budget):
            tree = gp.PrimitiveTree(sg.gen_tree(min_=min_depth, max_=max_depth, rng=rng))
            if sg.limit_excess(tree):
                continue
            ledger.structures.add(sg.to_text(sg.canonical_slots(tree)))
            ledger.backtest(tree, sg.random_params(tree, rng))
    except lab.BudgetExhausted:
        pass
    return ledger


if __name__ == "__main__":
    for name, edge in (("planted", lab.EDGE), ("noise", 0.0)):
        data = lab.synthetic(seed=7, edge=edge)
        lab.report(f"arm 0, {name}", data, random_search(data, 2400, seed=1))

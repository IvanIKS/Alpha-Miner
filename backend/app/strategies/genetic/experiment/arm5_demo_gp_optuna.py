"""Arm 5 smoke test: GP builds rule structures, Optuna fills their slots on train, every
backtest is logged once in the shared ledger (lab.py), and the rule handed in is chosen by
the same verifier as in the LLM arms (lab.select_final).

Synthetic hourly bars: 'planted' carries a symmetric mean-reversion edge (drift up on the
bar after RSI(14) < 30, down after RSI(14) > 70, so buy-and-hold earns nothing from it);
'noise' is the same random path without it. Search-time decisions see train and validation
only; the hold-out window is read once, at the end.

Run: python demo_gp_optuna.py
"""
from __future__ import annotations

import random

from deap import base, creator, gp, tools

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.strategy_gp as sg

if not hasattr(creator, "RuleFitness"):
    creator.create("RuleFitness", base.Fitness, weights=(1.0,))
    creator.create("RuleTree", gp.PrimitiveTree, fitness=creator.RuleFitness, params=list)


def evolve(data, seed, budget=2400, pop_size=30, k=10, n_search=lab.N_SEARCH, tune_fn=None):
    """Arm 5 on one dataset: returns the ledger (search window = the first n_search bars) and
    the number of generations.

    `tune_fn(template, ledger, k) -> (fitness, values)` fills the slots of a GP structure and
    is the only thing that separates the arms that share this engine: arm 5 leaves the default
    (`lab.tune`, Optuna); arm 2 passes an LLM tuner with the same contract."""
    tune_fn = tune_fn or lab.tune
    random.seed(seed)
    ledger = lab.Ledger(data, budget, n_search=n_search)
    tb = base.Toolbox()
    tb.register("mate", sg.cx_one_point)                  # gp.cxOnePoint is not reproducible by seed
    tb.register("mut_uniform", gp.mutUniform, expr=lambda pset, type_: sg.gen_tree(pset, 0, 2, type_), pset=sg.PSET)
    tb.register("mut_node", gp.mutNodeReplacement, pset=sg.PSET)
    for name in ("mate", "mut_uniform", "mut_node"):
        tb.decorate(name, gp.staticLimit(key=sg.limit_excess, max_value=0))

    pop, generations = [fresh() for _ in range(pop_size)], 0
    try:
        for ind in pop:
            ind.fitness.values = evaluate(ind, ledger, k, tune_fn)
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
                    ind.fitness.values = evaluate(ind, ledger, k, tune_fn)
            pop = tools.selBest(pop, 2) + offspring
    except lab.BudgetExhausted:
        pass
    return ledger, generations

def fresh():
    while True:
        ind = creator.RuleTree(sg.gen_tree(min_=2, max_=4))
        if not sg.limit_excess(ind):
            return ind

def evaluate(ind, ledger, k, tune_fn=lab.tune):
    for i, node in enumerate(sg.canonical_slots(ind)):   # W2-only and W1-only copies: one structure
        ind[i] = node                                     # PrimitiveTree rejects ind[:] = ...
    fit, ind.params = tune_fn(ind, ledger, k)
    return (fit,)


if __name__ == "__main__":
    for name, edge in (("planted", lab.EDGE), ("noise", 0.0)):
        data = lab.synthetic(seed=7, edge=edge)
        ledger, generations = evolve(data, seed=1)
        lab.report(name, data, ledger, f"{generations} generations")

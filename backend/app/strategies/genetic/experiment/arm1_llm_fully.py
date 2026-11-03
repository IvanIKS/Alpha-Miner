from deap import gp, base, creator, tools
from numpy.random._examples.cffi.extending import rng

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.strategy_gp as sg
import app.strategies.genetic.llm.llm_generator as llm
import random

def fully_llm_search(data, seed, budget=2400, pop_size=10, k=10, n_search=lab.N_SEARCH):
    random.seed(seed)
    ledger = lab.Ledger(data, budget, n_search=n_search)
    tb = base.Toolbox()
    tb.register("mate", sg.cx_one_point)  # gp.cxOnePoint is not reproducible by seed
    tb.register("mut_uniform", gp.mutUniform, expr=lambda pset, type_: sg.gen_tree(pset, 0, 2, type_), pset=sg.PSET)
    tb.register("mut_node", gp.mutNodeReplacement, pset=sg.PSET)
    for name in ("mate", "mut_uniform", "mut_node"):
        tb.decorate(name, gp.staticLimit(key=sg.limit_excess, max_value=0))

    pop, generations = [fresh() for _ in range(pop_size)], 0
    try:
        for ind in pop:
            ind.fitness.values = evaluate(ind, ledger, k)
        while True:
            """
            1. Create strategies using LLM. 
            2. Run strategies against the backtest.
            3. Use LLM to tune the strategy parameters - 
            4. Evaluate the strategies and promote the best.            
            """
            raise lab.BudgetExhausted
    except lab.BudgetExhausted:
        pass
    return ledger, generations


def fresh():
    while True:
        ind = creator.RuleTree(sg.gen_tree(min_=2, max_=4))
        if not sg.limit_excess(ind):
            return ind

def evaluate(ind, ledger, k):
    for i, node in enumerate(sg.canonical_slots(ind)):  # W2-only and W1-only copies: one structure
        ind[i] = node  # PrimitiveTree rejects ind[:] = ...
    fit, ind.params = lab.tune(ind, ledger, k)
    return (fit,)
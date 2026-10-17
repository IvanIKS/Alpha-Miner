"""Checks for strategy_gp: causality, leak detection, engine parity, parser, DEAP operators.

Run: pytest -q test_strategy_gp.py
"""
import random
from dataclasses import dataclass, field

import numpy as np
import optuna
import pytest
from deap import base, creator, gp

import app.strategies.genetic.strategy_gp as sg


# ---------------------------------------------------------------- helpers --
def make_data(n, seed=0, sigma=0.01):
    """Tick-rounded OHLCV random walk: discrete prices make exact ties possible."""
    rng = np.random.default_rng(seed)
    close = np.round(30000 * np.exp(np.cumsum(sigma * rng.standard_normal(n))), 2)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0, sigma * 0.5, (2, n)))
    high = np.round(np.maximum(open_, close) * np.exp(wick[0]), 2)
    low = np.round(np.minimum(open_, close) * np.exp(-wick[1]), 2)
    volume = np.round(rng.lognormal(3, 0.5, n), 3)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def small_params(tree, rng):
    """Random slot values with short windows, so trees trade inside short test series."""
    vals = []
    for _, s in sg.slots(tree):
        if s.token == "W":
            vals.append(rng.randint(2, 40))
        elif s.token == "LAG":
            vals.append(rng.randint(1, 5))
        else:
            vals.append(round(s.lo + s.step * rng.randrange(int(round((s.hi - s.lo) / s.step)) + 1), 6))
    return vals


def random_rules(count, seed):
    rng = random.Random(seed)
    random.seed(seed)
    out = []
    while len(out) < count:
        tree = gp.PrimitiveTree(sg.gen_tree(min_=1, max_=4))
        if sg.limit_excess(tree):
            continue
        out.append(sg.bind(tree, small_params(tree, rng)))
    return out


DATA = make_data(2500, seed=7)


# ------------------------------------------------------------ the grammar --
def test_classics_parse_and_round_trip():
    for name, text in sg.CLASSICS.items():
        tree = sg.parse(text)
        again = sg.parse(sg.to_text(tree))
        assert sg.to_text(again) == sg.to_text(tree), name
        assert not sg.limit_violations(tree), name
    rsi = sg.parse(sg.CLASSICS["rsi_reversion"])
    assert len(rsi) == 11 and sg.n_numbers(rsi) == 4


def test_generator_never_fails_and_types_hold():
    random.seed(0)
    for _ in range(3000):
        tree = gp.PrimitiveTree(sg.gen_tree(min_=0, max_=5))
        assert tree[0].ret is sg.Rule
        if not sg.limit_excess(tree):
            sg.parse(sg.to_text(tree))       # every generated tree is valid text in the same grammar


def test_slots_schema_and_bind():
    tree = sg.parse("long_flat(and_(osc_below(rsi(close, W1), LVL1), gt(close, sma(close, W2))), "
                    "osc_above(rsi(close, W1), LVL2))")
    assert sg.slot_names(tree) == ["W1", "LVL1", "W2", "LVL2"]
    assert list(sg.params_schema(tree)["properties"]) == ["W1", "LVL1", "W2", "LVL2"]
    assert sg.n_numbers(tree) == 5                     # five numeric arguments, four free values
    bound = sg.bind(tree, [14, 30, 200, 70])
    assert sg.slots(bound) == [] and sg.lookback(bound) == 200 and sg.n_numbers(bound) == 5
    assert sg.to_text(sg.bind(tree, {"W1": 14, "LVL1": 30, "W2": 200, "LVL2": 70})) == sg.to_text(bound)
    with pytest.raises(sg.GrammarError, match="W2=1000 is outside"):
        sg.bind(tree, [14, 30, 1000, 70])


def test_reused_placeholder_ties_values():
    """One name, one parameter: like RSIStrategy's single rsi_period used by entry and exit."""
    template = sg.parse("long_flat(osc_below(rsi(close, W1), LVL1), osc_above(rsi(close, W1), LVL2))")
    assert sg.slot_names(template) == ["W1", "LVL1", "LVL2"]
    trial = optuna.trial.FixedTrial({"W1": 14, "LVL1": 30.0, "LVL2": 70.0})
    bound = sg.bind(template, sg.suggest(trial, template))
    assert sg.to_text(bound) == sg.to_text(sg.parse(sg.CLASSICS["rsi_reversion"]))
    untied = sg.parse("long_flat(osc_below(rsi(close, W1), LVL1), osc_above(rsi(close, W2), LVL2))")
    assert len(sg.slots(untied)) == 4
    renamed = sg.parse("long_flat(osc_below(rsi(close, W3), LVL2), osc_above(rsi(close, W3), LVL1))")
    assert sg.to_text(sg.canonical_slots(renamed)) == sg.to_text(template)


def test_binding_keeps_rules_inside_limits():
    """Limits count numeric positions, so an optimized template is still a valid rule
    for allow_slots=False (what RuleStrategy parses)."""
    rng = random.Random(4)
    random.seed(4)
    checked = 0
    while checked < 500:
        tree = gp.PrimitiveTree(sg.gen_tree(min_=1, max_=5))
        if sg.limit_excess(tree):
            continue
        sg.parse(sg.to_text(sg.bind(tree, sg.random_params(tree, rng))), allow_slots=False)
        checked += 1


# ------------------------------------------------------------- causality --
def test_random_rules_are_causal():
    for tree in random_rules(300, seed=1):
        assert sg.prefix_invariant(tree, DATA), sg.to_text(tree)


def test_planted_leak_is_caught():
    leaky = sg.parse("long_flat(gt(lead(close, 1), close), gt(close, lead(close, 1)))", _allow_hidden=True)
    assert not sg.prefix_invariant(leaky, DATA)
    with pytest.raises(sg.GrammarError, match="unknown name 'lead'"):
        sg.parse("long_flat(gt(lead(close, 1), close), gt(close, lead(close, 1)))")
    assert "lead" not in sg.PSET.mapping and "lead" not in sg.catalog()


# --------------------------------------------------------- engine parity --
@dataclass
class _Position:
    quantity: float


@dataclass
class _Portfolio:
    base_token_amount: float = 1000.0
    positions: list = field(default_factory=list)


@dataclass
class _Context:
    market_data: dict
    portfolio: _Portfolio


def _run_event_driven(rule_text, data):
    from app.strategies.genetic.rule_strategy import RuleStrategy
    strat = RuleStrategy(rule=rule_text)
    pf = _Portfolio()
    held = np.zeros(len(data["close"]))
    for t in range(len(data["close"])):
        bar = {k: v[t] for k, v in data.items()}
        for sig in strat.on_tick(_Context(bar, pf)):
            if sig.side == "buy":
                pf.positions, pf.base_token_amount = [_Position(sig.quantity)], 0.0
            else:
                pf.positions, pf.base_token_amount = [], sig.quantity * sig.price
        held[t] = 1.0 if pf.positions else 0.0
    return held


def test_engine_parity():
    data = make_data(1200, seed=11)
    rules = [sg.parse(t) for t in sg.CLASSICS.values()] + random_rules(8, seed=2)
    traded = 0
    for tree in rules:
        text = sg.to_text(tree)
        vector = sg.positions(*sg.Evaluator(data).signals(tree))
        event = _run_event_driven(text, data)
        assert np.array_equal(vector, event), text
        traded += sg.n_trades(vector) > 0
    assert traded >= 10                                   # the comparison is not vacuous


def test_ties_on_tick_prices_do_not_split_engines():
    """On discrete prices close == sma happens often; full-history and windowed sums differ
    in the last bits, so comparisons are rounded to 1e-9 (sg._DECIMALS)."""
    rng = np.random.default_rng(2)
    close = np.round(30000 + 0.01 * np.cumsum(rng.integers(-3, 4, 1500)), 2)
    data = {"open": np.r_[close[0], close[:-1]], "high": close + 0.01, "low": close - 0.01,
            "close": close, "volume": np.ones(len(close))}
    text = "long_flat(gt(close, sma(close, 4)), gt(sma(close, 4), close))"
    vector = sg.positions(*sg.Evaluator(data).signals(sg.parse(text)))
    assert np.array_equal(vector, _run_event_driven(text, data))


# ---------------------------------------------------------------- parser --
@pytest.mark.parametrize("text, fragment", [
    ("long_flat(osc_below(sma(close, 14), 30), osc_above(rsi(close, 14), 70))",
     "Allowed for Osc: rsi, stoch, ibs"),
    ("long_flat(osc_below(rsi(close, 1000), 30), osc_above(rsi(close, 14), 70))", "outside [2, 500]"),
    ("long_flat(osc_below(rsi(close, 14.5), 30), osc_above(rsi(close, 14), 70))", "must be an integer"),
    ("long_flat(osc_below(rsi(close), 30), osc_above(rsi(close, 14), 70))", "Signature: rsi(Price, Win)"),
    ("long_flat(gt(close, 30000), gt(open, close))", "a number cannot stand for Price"),
    ("long_flat(osc_below(rsi(close, 14), 30), __import__)", "unknown name '__import__'"),
    ("long_flat(osc_below(rsi(close, LVL1), 30), osc_above(rsi(close, 14), 70))", "LVL1 fills OscLevel, expected Win"),
    ("long_flat(osc_below(rsi(close, W4), 30), osc_above(rsi(close, 14), 70))", "placeholder W1, W2, W3"),
])
def test_parser_errors_are_actionable(text, fragment):
    with pytest.raises(sg.GrammarError) as err:
        sg.parse(text)
    assert fragment in str(err.value), str(err.value)


def test_numbers_required_when_slots_disallowed():
    with pytest.raises(sg.GrammarError, match="placeholder"):
        sg.parse("long_flat(osc_below(rsi(close, W1), 30), osc_above(rsi(close, 14), 70))", allow_slots=False)


# ----------------------------------------------------------- DEAP wiring --
def test_deap_variation_keeps_trees_valid():
    random.seed(3)
    if not hasattr(creator, "RuleFitness"):
        creator.create("RuleFitness", base.Fitness, weights=(1.0,))
        creator.create("RuleTree", gp.PrimitiveTree, fitness=creator.RuleFitness)
    tb = base.Toolbox()
    tb.register("mate", gp.cxOnePoint)
    tb.register("mut_uniform", gp.mutUniform, expr=lambda pset, type_: sg.gen_tree(pset, 0, 2, type_), pset=sg.PSET)
    tb.register("mut_node", gp.mutNodeReplacement, pset=sg.PSET)
    for name in ("mate", "mut_uniform", "mut_node"):
        tb.decorate(name, gp.staticLimit(key=sg.limit_excess, max_value=0))
    pop = []
    while len(pop) < 60:
        ind = creator.RuleTree(sg.gen_tree(min_=2, max_=4))
        if not sg.limit_excess(ind):
            pop.append(ind)
    for _ in range(600):
        a, b = (tb.clone(x) for x in random.sample(pop, 2))
        a, b = tb.mate(a, b)
        (c,) = tb.mut_uniform(a)
        (d,) = tb.mut_node(b)
        for child in (c, d):
            assert not sg.limit_violations(child)
            sg.parse(sg.to_text(child))                   # typed and within limits
        pop[random.randrange(len(pop))] = c

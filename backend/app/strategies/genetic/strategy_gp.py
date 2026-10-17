"""Typed strategy grammar for genetic programming (DEAP) and LLM search arms.

A strategy = archetype (a fixed state machine at the tree root)
           + condition subtrees built from causal, finite-memory operators
           + numeric slots kept apart from the structure.

One registry (OPS, TERMINALS, SLOTS) produces everything else:
  * PSET, gen_tree             - DEAP PrimitiveSetTyped and a typed tree generator (GP arms);
  * catalog, parse             - text grammar and a strict parser for LLM output (LLM arms);
  * slots, suggest, params_schema - one numeric space for Optuna and for an LLM tool call
                                 (placeholders W1, LVL1, ...; a reused name ties values);
  * Evaluator, lookback        - vectorized signals and the exact history an event-driven
                                 engine needs to reproduce them.
Operators with selectable=False (the planted leak `lead`) exist only for tests: they are
absent from PSET, the catalog and the parser, so look-ahead cannot be expressed.
"""
from __future__ import annotations

import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from deap import gp


class GrammarError(ValueError):
    """Invalid expression. The message is written to be shown back to an LLM."""


# ------------------------------------------------------------------ types --
class Price:    """Price-scaled series (quote currency)."""
class Volume:   """Volume series."""
class Osc:      """Bounded oscillator, 0..100."""
class Z:        """Dimensionless series, roughly N(0, 1)."""
class Bool:     """Condition: 1.0 true, 0.0 false, NaN unknown (warm-up or undefined)."""
class Rule:     """Root: an archetype with its condition subtrees."""

class Win:      """Lookback window, bars."""
class Lag:      """Lag, bars."""
class OscLevel: """Oscillator level."""
class ZLevel:   """Z-score level."""
class KMult:    """Band width, standard deviations."""


@dataclass(frozen=True)
class Slot:
    token: str            # placeholder stem: W, LAG, LVL, ZL, KM -> names W1, W2, ...
    kind: str             # "int" | "float"
    lo: float
    hi: float
    log: bool             # log-uniform sampling (windows)
    step: float | None    # grid step for floats: readable values, fewer pseudo-distinct trials
    meaning: str
    count: int            # named placeholders per rule; reusing a name ties the values

    def check(self, v, name=None):
        """Validate a value for this slot and return a plain int/float."""
        label = name or self.token
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise GrammarError(f"{label} needs a number, got {v!r}") from None
        if not math.isfinite(v):
            raise GrammarError(f"{label} needs a finite number, got {v!r}")
        if self.kind == "int":
            if v != int(v):
                raise GrammarError(f"{label} ({self.meaning}) must be an integer, got {v:g}")
            v = int(v)
        else:
            v = round(v, 6)
        if not self.lo <= v <= self.hi:
            raise GrammarError(f"{label}={v:g} is outside [{self.lo:g}, {self.hi:g}] ({self.meaning})")
        return v


SLOTS = {
    Win:      Slot("W",   "int",   2,    500,  True,  None, "lookback window, bars", 3),
    Lag:      Slot("LAG", "int",   1,    20,   False, None, "lag, bars", 2),
    OscLevel: Slot("LVL", "float", 1.0,  99.0, False, 0.5,  "oscillator level, 0..100", 2),
    ZLevel:   Slot("ZL",  "float", -3.0, 3.0,  False, 0.05, "z-score level", 2),
    KMult:    Slot("KM",  "float", 0.5,  4.0,  False, 0.05, "band width, std", 2),
}
# W1, W2, W3, LAG1, ... -> type. A rule's free parameters are the distinct names it uses,
# so rsi(close, W1) in both entry and exit is one window, as in RSIStrategy.
SLOT_TYPE_BY_NAME = {f"{s.token}{i}": t for t, s in SLOTS.items() for i in range(1, s.count + 1)}

# Limits shared by every arm: same space for GP and for the LLM.
MAX_HEIGHT, MAX_NODES, MAX_NUMBERS = 6, 40, 8


# ------------------------------------------------- causal numeric kernels --
# Every kernel is causal (output[t] uses inputs[<= t]) and has finite memory,
# so a window of `lookback` bars reproduces the full-history value exactly.
_DECIMALS = 9   # comparisons see values rounded to 1e-9: both engines agree on ties


def _roll(x, w):
    w = int(w)
    return pd.Series(x).rolling(w, min_periods=w)


def _sma(x, w):
    return _roll(x, w).mean().to_numpy()


def _std(x, w):
    return _roll(x, w).std(ddof=0).to_numpy()


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(a, float) / np.asarray(b, float)
    out[~np.isfinite(out)] = np.nan
    return out


def _shift(x, k):
    """k > 0: value k bars ago. k < 0: value |k| bars ahead (look-ahead, tests only)."""
    n = len(x)
    out = np.full(n, np.nan)
    if 0 < k < n:
        out[k:] = x[: n - k]
    elif -n < k < 0:
        out[: n + k] = x[-k:]
    return out


def _gt(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = (np.round(a, _DECIMALS) > np.round(b, _DECIMALS)).astype(float)
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out


def _and(a, b):   # Kleene logic: a known False decides, otherwise unknown stays unknown
    return np.where((a == 0) | (b == 0), 0.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 1.0))


def _or(a, b):
    return np.where((a == 1) | (b == 1), 1.0, np.where(np.isnan(a) | np.isnan(b), np.nan, 0.0))


def _cross_up(a, b):
    was_above = _gt(_shift(a, 1), _shift(b, 1))
    return _and(_gt(a, b), 1.0 - was_above)


def _rsi(x, w):
    """RSI on simple averages (Cutler): finite memory, unlike Wilder's smoothing."""
    d = np.diff(x, prepend=np.nan)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    gain[np.isnan(d)] = np.nan
    loss[np.isnan(d)] = np.nan
    ag, al = _sma(gain, w), _sma(loss, w)
    tot = ag + al
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(tot > 0, 100.0 * ag / tot, 50.0)
    r[np.isnan(tot)] = np.nan
    return r


def _stoch(ctx, w):
    hh = _roll(ctx["high"], w).max().to_numpy()
    ll = _roll(ctx["low"], w).min().to_numpy()
    rng = hh - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(rng > 0, 100.0 * (ctx["close"] - ll) / rng, 50.0)
    k[np.isnan(rng)] = np.nan
    return k


def _ibs(ctx):
    rng = ctx["high"] - ctx["low"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rng > 0, 100.0 * (ctx["close"] - ctx["low"]) / rng, 50.0)


def _zscore(x, w):
    return _ratio(x - _sma(x, w), _std(x, w))


def _mom(x, w):
    lx = np.log(x)
    vol = _std(np.diff(lx, prepend=np.nan), w)
    return _ratio(lx - _shift(lx, int(w)), vol * math.sqrt(int(w)))


def _vol_regime(ctx, w1, w2):
    r1 = np.diff(np.log(ctx["close"]), prepend=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log(_std(r1, w1) / _std(r1, w2))
    v[~np.isfinite(v)] = np.nan
    return v


# --------------------------------------------------------------- registry --
@dataclass(frozen=True)
class Op:
    name: str
    args: tuple
    ret: type
    fn: Callable                       # fn(ctx, *args) -> array (a tuple for Rule)
    need: Callable                     # need(series_needs, numbers) -> bars of history
    doc: str
    uses: frozenset = frozenset()      # raw fields the kernel reads from ctx directly
    selectable: bool = True


def _need_roll(s, n):
    return s[0] + int(n[0]) - 1


def _need_shift(s, n):
    return s[0] + int(n[0])


def _need_max(s, n):
    return max(s)


_HLC = frozenset({"high", "low", "close"})

OPS = {op.name: op for op in [
    # price -> price
    Op("sma", (Price, Win), Price, lambda c, x, w: _sma(x, w), _need_roll, "simple moving average"),
    Op("rmax", (Price, Win), Price, lambda c, x, w: _roll(x, w).max().to_numpy(), _need_roll,
       "rolling maximum (channel top), includes the current bar"),
    Op("rmin", (Price, Win), Price, lambda c, x, w: _roll(x, w).min().to_numpy(), _need_roll,
       "rolling minimum (channel bottom), includes the current bar"),
    Op("lag", (Price, Lag), Price, lambda c, x, k: _shift(x, int(k)), _need_shift, "value LAG bars ago"),
    Op("bb_up", (Price, Win, KMult), Price, lambda c, x, w, k: _sma(x, w) + k * _std(x, w), _need_roll,
       "Bollinger upper band"),
    Op("bb_dn", (Price, Win, KMult), Price, lambda c, x, w, k: _sma(x, w) - k * _std(x, w), _need_roll,
       "Bollinger lower band"),
    Op("sma_v", (Volume, Win), Volume, lambda c, v, w: _sma(v, w), _need_roll, "volume moving average"),
    # oscillators, 0..100
    Op("rsi", (Price, Win), Osc, lambda c, x, w: _rsi(x, w), _need_shift, "RSI on simple averages (Cutler)"),
    Op("stoch", (Win,), Osc, lambda c, w: _stoch(c, w), lambda s, n: int(n[0]),
       "stochastic %K: close within the W-bar high-low range", _HLC),
    # dimensionless
    Op("zscore", (Price, Win), Z, lambda c, x, w: _zscore(x, w), _need_roll, "(x - mean) / std over W"),
    Op("mom", (Price, Win), Z, lambda c, x, w: _mom(x, w), _need_shift,
       "log return over W / (1-bar volatility * sqrt(W))"),
    Op("vol_z", (Volume, Win), Z, lambda c, v, w: _zscore(v, w), _need_roll, "volume z-score over W"),
    Op("vol_regime", (Win, Win), Z, lambda c, a, b: _vol_regime(c, a, b), lambda s, n: 1 + max(int(n[0]), int(n[1])),
       "log(volatility of close over the first W / over the second W)", frozenset({"close"})),
    # conditions
    Op("gt", (Price, Price), Bool, lambda c, a, b: _gt(a, b), _need_max, "a > b"),
    Op("cross_up", (Price, Price), Bool, lambda c, a, b: _cross_up(a, b), lambda s, n: max(s) + 1,
       "a crosses above b on this bar"),
    Op("gt_v", (Volume, Volume), Bool, lambda c, a, b: _gt(a, b), _need_max, "volume a > volume b"),
    Op("osc_above", (Osc, OscLevel), Bool, lambda c, o, lv: _gt(o, lv), _need_max, "oscillator above level"),
    Op("osc_below", (Osc, OscLevel), Bool, lambda c, o, lv: _gt(lv, o), _need_max, "oscillator below level"),
    Op("z_above", (Z, ZLevel), Bool, lambda c, z, lv: _gt(z, lv), _need_max, "z above level"),
    Op("z_below", (Z, ZLevel), Bool, lambda c, z, lv: _gt(lv, z), _need_max, "z below level"),
    Op("and_", (Bool, Bool), Bool, lambda c, a, b: _and(a, b), _need_max, "both conditions"),
    Op("or_", (Bool, Bool), Bool, lambda c, a, b: _or(a, b), _need_max, "either condition"),
    Op("not_", (Bool,), Bool, lambda c, a: 1.0 - a, _need_max, "negation; unknown stays unknown"),
    # archetype at the root
    Op("long_flat", (Bool, Bool), Rule, lambda c, e, x: (e, x), _need_max,
       "long/flat: enter on entry, leave on exit; exit wins on the same bar"),
    # planted leak: registered for tests, never selectable
    Op("lead", (Price, Lag), Price, lambda c, x, k: _shift(x, -int(k)), _need_shift,
       "value LAG bars AHEAD (look-ahead)", selectable=False),
]}
COMMUTATIVE = {"and_", "or_"}

TERMINALS = {   # name -> (type, kernel, raw fields it reads)
    "close": (Price, lambda c: c["close"], frozenset({"close"})),
    "open": (Price, lambda c: c["open"], frozenset({"open"})),
    "high": (Price, lambda c: c["high"], frozenset({"high"})),
    "low": (Price, lambda c: c["low"], frozenset({"low"})),
    "volume": (Volume, lambda c: c["volume"], frozenset({"volume"})),
    "ibs": (Osc, _ibs, _HLC),
}

# Classic setups written in the grammar: tests, LLM few-shot examples, optional GP seeds.
CLASSICS = {
    "rsi_reversion": "long_flat(osc_below(rsi(close, 14), 30), osc_above(rsi(close, 14), 70))",
    "ma_cross": "long_flat(cross_up(sma(close, 20), sma(close, 100)), cross_up(sma(close, 100), sma(close, 20)))",
    "donchian_breakout": "long_flat(gt(close, lag(rmax(high, 55), 1)), gt(lag(rmin(low, 20), 1), close))",
    "bollinger_reversion": "long_flat(gt(bb_dn(close, 20, 2), close), gt(close, sma(close, 20)))",
    "momentum": "long_flat(z_above(mom(close, 100), 1), z_below(mom(close, 100), 0))",
    "rsi_with_trend_filter": "long_flat(and_(gt(close, sma(close, 200)), osc_below(rsi(close, 5), 20)), "
                             "osc_above(rsi(close, 5), 60))",
    "volatility_breakout": "long_flat(and_(z_above(vol_regime(10, 100), 0.2), z_above(mom(close, 10), 1)), "
                           "z_below(mom(close, 10), 0))",
    "volume_spike": "long_flat(and_(z_above(vol_z(volume, 50), 2), gt(close, open)), gt(open, close))",
    "ibs_reversion": "long_flat(osc_below(ibs, 20), osc_above(ibs, 80))",
    "stochastic": "long_flat(osc_below(stoch(14), 20), osc_above(stoch(14), 80))",
}


# ------------------------------------------------------------------- DEAP --
def _build_pset():
    pset = gp.PrimitiveSetTyped("RULE", [], Rule)
    for op in OPS.values():
        if op.selectable:
            pset.addPrimitive(op.fn, list(op.args), op.ret, name=op.name)
    for name, (t, _, _) in TERMINALS.items():
        pset.addTerminal(name, t, name=name)
    for name, t in SLOT_TYPE_BY_NAME.items():
        pset.addTerminal(name, t, name=name)
    return pset


PSET = _build_pset()
_HIDDEN = {op.name: gp.Primitive(op.name, list(op.args), op.ret) for op in OPS.values() if not op.selectable}


def _min_depths(pset):
    """Smallest subtree height that can produce each type (0 if it has terminals)."""
    types = set(pset.primitives) | set(pset.terminals)
    depth = {t: 0 if pset.terminals[t] else math.inf for t in types}
    changed = True
    while changed:
        changed = False
        for t in types:
            for p in pset.primitives[t]:
                d = 1 + max((depth[a] for a in p.args), default=0)
                if d < depth[t]:
                    depth[t], changed = d, True
    return depth


MIN_DEPTH = _min_depths(PSET)


def gen_tree(pset=PSET, min_=2, max_=4, type_=None, p_leaf=0.4):
    """Typed half-and-half generator, usable as DEAP `expr` (also inside gp.mutUniform).

    DEAP's genFull/genGrow raise IndexError when a typed branch reaches the depth limit
    at a type without terminals (Bool, Z, Rule here); this one never picks a primitive
    whose arguments cannot be completed within the remaining depth.
    """
    type_ = type_ or pset.ret
    height = max(random.randint(min_, max_), MIN_DEPTH[type_])
    full = random.random() < 0.5
    expr, stack = [], [(0, type_)]
    while stack:
        depth, t = stack.pop()
        room = height - depth
        leaves = pset.terminals[t]
        prims = [p for p in pset.primitives[t] if all(MIN_DEPTH[a] < room for a in p.args)]
        if leaves and (not prims or room <= 0 or (not full and depth >= min_ and random.random() < p_leaf)):
            expr.append(random.choice(leaves))
        else:
            p = random.choice(prims)
            expr.append(p)
            stack.extend((depth + 1, a) for a in reversed(p.args))
    return expr


def n_numbers(tree):
    """Numeric arguments in the rule, placeholders included. Counted by position, so binding
    a template never changes it; the optimizer's dimension is len(slots(tree))."""
    return sum(1 for node in tree if isinstance(node, gp.Terminal) and node.ret in SLOTS)


def limit_violations(tree):
    out = []
    if tree.height > MAX_HEIGHT:
        out.append(f"height {tree.height} > {MAX_HEIGHT}")
    if len(tree) > MAX_NODES:
        out.append(f"{len(tree)} nodes > {MAX_NODES}")
    if n_numbers(tree) > MAX_NUMBERS:
        out.append(f"{n_numbers(tree)} numeric arguments > {MAX_NUMBERS}")
    return out


def limit_excess(tree):
    """Key for gp.staticLimit(key=limit_excess, max_value=0)."""
    return len(limit_violations(tree))


# ------------------------------------------------------------------ slots --
def _is_slot(node):
    return isinstance(node, gp.Terminal) and node.name in SLOT_TYPE_BY_NAME


def slot_names(tree):
    """Free parameters: distinct slot names in order of first use."""
    return list(dict.fromkeys(node.name for node in tree if _is_slot(node)))


def slots(tree):
    """[(name, Slot)] that Optuna or an LLM has to fill; a reused name is one parameter."""
    return [(name, SLOTS[SLOT_TYPE_BY_NAME[name]]) for name in slot_names(tree)]


def canonical_slots(tree):
    """Rename placeholders by order of first use within each type (W2, W1 -> W1, W2).
    Templates that differ only in placeholder names then share one text, one tuning
    run and one ledger history; without this GP re-tunes renamed copies."""
    renamed, used = {}, {}
    out = []
    for node in tree:
        if _is_slot(node):
            t = SLOT_TYPE_BY_NAME[node.name]
            if node.name not in renamed:
                used[t] = used.get(t, 0) + 1
                renamed[node.name] = f"{SLOTS[t].token}{used[t]}"
            node = PSET.mapping[renamed[node.name]]
        out.append(node)
    return gp.PrimitiveTree(out)


def _numbers(tree, params):
    """Map every slot occurrence (tree position) to its checked value.
    params: a list in slot_names() order, or a dict {name: value}."""
    names = slot_names(tree)
    if isinstance(params, dict):
        if sorted(params) != sorted(names):
            raise ValueError(f"expected values for {names}, got {sorted(params)}")
        values = params
    else:
        if len(params) != len(names):
            raise ValueError(f"tree has free slots {names}, got {len(params)} values")
        values = dict(zip(names, params))
    checked = {n: SLOTS[SLOT_TYPE_BY_NAME[n]].check(values[n], n) for n in names}
    return {i: checked[node.name] for i, node in enumerate(tree) if _is_slot(node)}


def bind(tree, params):
    """Copy of the tree with free slots replaced by numbers (a fully specified rule)."""
    nums = _numbers(tree, params)
    return gp.PrimitiveTree([gp.Terminal(nums[i], False, node.ret) if i in nums else node
                             for i, node in enumerate(tree)])


def suggest(trial, tree):
    """Optuna: one parameter per free slot name, same ranges as everywhere else."""
    vals = []
    for name, s in slots(tree):
        if s.kind == "int":
            vals.append(trial.suggest_int(name, int(s.lo), int(s.hi), log=s.log))
        else:
            vals.append(trial.suggest_float(name, s.lo, s.hi, step=s.step))
    return vals


def params_schema(tree):
    """JSON schema for an LLM tool call that fills the slots (arm 2: GP structure, LLM numbers)."""
    props = {name: {"type": "integer" if s.kind == "int" else "number",
                    "minimum": s.lo, "maximum": s.hi, "description": s.meaning}
             for name, s in slots(tree)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def random_params(tree, rng=random):
    """Arm 0 and tests: log-uniform windows, grid floats."""
    vals = []
    for _, s in slots(tree):
        if s.kind == "int":
            lo, hi = int(s.lo), int(s.hi)
            v = math.exp(rng.uniform(math.log(lo), math.log(hi + 1))) if s.log else rng.uniform(lo, hi + 1)
            vals.append(min(int(v), hi))
        else:
            vals.append(round(s.lo + s.step * rng.randrange(int(round((s.hi - s.lo) / s.step)) + 1), 6))
    return vals


# ------------------------------------------------------------- evaluation --
class Evaluator:
    """Vectorized evaluation of trees on one dataset, with a subexpression cache."""

    def __init__(self, data, cache_size=256):
        self.ctx = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def signals(self, tree, params=()):
        """(entry, exit) as 1/0/NaN arrays. The first lookback-1 bars are unknown."""
        nums = _numbers(tree, params)
        (entry, exit_), _, need, _ = self._eval(tree, 0, nums)
        entry, exit_ = np.array(entry, dtype=float), np.array(exit_, dtype=float)
        entry[: need - 1] = np.nan
        exit_[: need - 1] = np.nan
        return entry, exit_

    def _eval(self, tree, i, nums):
        node = tree[i]
        if isinstance(node, gp.Terminal):
            if i in nums:
                return nums[i], repr(nums[i]), 0, i + 1
            if node.name in TERMINALS:
                return TERMINALS[node.name][1](self.ctx), node.name, 1, i + 1
            return node.value, repr(node.value), 0, i + 1
        op = OPS[node.name]
        vals, keys, series_need, numbers, j = [], [], [], [], i + 1
        for t in op.args:
            v, k, nd, j = self._eval(tree, j, nums)
            vals.append(v)
            keys.append(k)
            (numbers if t in SLOTS else series_need).append(v if t in SLOTS else nd)
        key = f"{op.name}({','.join(keys)})"
        need = op.need(series_need, numbers)
        if op.ret is Rule:
            return op.fn(self.ctx, *vals), key, need, j
        out = self._cache.get(key)
        if out is None:
            out = op.fn(self.ctx, *vals)
            if self.cache_size:
                self._cache[key] = out
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return out, key, need, j


def lookback(tree, params=()):
    """Bars of history the rule needs: on a window this long, its value at the last bar
    equals the full-history value (every operator has finite memory)."""
    nums = _numbers(tree, params)

    def rec(i):
        node = tree[i]
        if isinstance(node, gp.Terminal):
            if i in nums:
                return 0, nums[i], i + 1
            return (1, None, i + 1) if node.name in TERMINALS else (0, node.value, i + 1)
        op, s, n, j = OPS[node.name], [], [], i + 1
        for t in op.args:
            nd, v, j = rec(j)
            (n if t in SLOTS else s).append(v if t in SLOTS else nd)
        return op.need(s, n), None, j

    return rec(0)[0]


def fields_used(tree):
    used = {"close"}
    for node in tree:
        if isinstance(node, gp.Terminal) and node.name in TERMINALS:
            used |= TERMINALS[node.name][2]
        elif isinstance(node, gp.Primitive):
            used |= OPS[node.name].uses
    return used


def prefix_invariant(tree, data, params=(), cuts=(0.3, 0.6, 0.9)):
    """True if signals computed on data[:c] equal the full-data signals on [:c]."""
    full = Evaluator(data, cache_size=0).signals(tree, params)
    n = len(data["close"])
    for frac in cuts:
        c = max(2, int(n * frac))
        part = Evaluator({k: v[:c] for k, v in data.items()}, cache_size=0).signals(tree, params)
        if not all(np.array_equal(a[:c], b, equal_nan=True) for a, b in zip(full, part)):
            return False
    return True


# -------------------------------------------- archetype and reference run --
def positions(entry, exit_):
    """Long/flat state after each close. Unknown conditions never act; exit wins on the same bar."""
    event = np.where(exit_ == 1.0, -1.0, np.where(entry == 1.0, 1.0, np.nan))
    return (pd.Series(event).ffill().to_numpy() == 1.0).astype(float)


def net_returns(pos, close, cost_bps):
    """Per-bar returns: position decided at a close, held over the next bar, costs per side."""
    close = np.asarray(close, dtype=float)
    r = np.zeros(len(close))
    r[1:] = close[1:] / close[:-1] - 1.0
    held = np.zeros(len(pos))
    held[1:] = pos[:-1]
    return held * r - np.abs(np.diff(pos, prepend=0.0)) * (cost_bps / 1e4)


def n_trades(pos):
    return int(np.sum(np.diff(pos, prepend=0.0) > 0))


def sharpe(r, periods_per_year):
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return 0.0 if sd == 0 else float(np.mean(r)) / sd * math.sqrt(periods_per_year)


# ------------------------------------------------------------- text form --
def to_text(tree, canonical=False):
    """Readable form, parseable back. canonical=True sorts the arguments of and_/or_
    (use on bound trees as a ledger key)."""

    def rec(i):
        node = tree[i]
        if isinstance(node, gp.Terminal):
            named = node.name in TERMINALS or _is_slot(node)
            return (node.name if named else repr(node.value)), i + 1
        parts, j = [], i + 1
        for _ in range(node.arity):
            s, j = rec(j)
            parts.append(s)
        if canonical and node.name in COMMUTATIVE:
            parts.sort()
        return f"{node.name}({', '.join(parts)})", j

    return rec(0)[0]


def _sig(op):
    return f"{op.name}({', '.join(a.__name__ for a in op.args)}) -> {op.ret.__name__}"


def _allowed(t):
    names = [op.name for op in OPS.values() if op.selectable and op.ret is t]
    names += [n for n, (tt, _, _) in TERMINALS.items() if tt is t]
    if t in SLOTS:
        s = SLOTS[t]
        placeholders = ", ".join(n for n, tt in SLOT_TYPE_BY_NAME.items() if tt is t)
        names.append(f"a number in [{s.lo:g}, {s.hi:g}] or a placeholder {placeholders}")
    return f"Allowed for {t.__name__}: {', '.join(names)}."


def catalog():
    """The grammar as shown to an LLM; generated from the same registry as PSET and parse."""
    lines = ["Write one expression rooted at long_flat(entry, exit).",
             "Types: Price, Volume, Osc (0..100), Z (dimensionless), Bool (condition), Rule.",
             "Operators:"]
    lines += [f"  {_sig(op)}: {op.doc}" for op in OPS.values() if op.selectable]
    lines.append("Series: " + ", ".join(f"{n} ({t.__name__})" for n, (t, _, _) in TERMINALS.items()))
    lines.append("Numbers: write a value, or a placeholder when an optimizer will choose it; "
                 "the same placeholder twice means the same value:")
    lines += [f"  {s.token}1..{s.token}{s.count}: {t.__name__}, {s.kind} in [{s.lo:g}, {s.hi:g}], {s.meaning}"
              for t, s in SLOTS.items()]
    lines.append(f"Limits: height <= {MAX_HEIGHT}, nodes <= {MAX_NODES}, "
                 f"numeric arguments (numbers and placeholders) <= {MAX_NUMBERS}.")
    return "\n".join(lines)


# ----------------------------------------------------------- strict parser --
_TOKEN = re.compile(r"\s*(?:(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
                    r"|(?P<name>[A-Za-z_]\w*)|(?P<sym>[(),]))")


def _tokens(text):
    text, pos, out = text.strip(), 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise GrammarError(f"unexpected text at {pos}: {text[pos:pos + 15]!r}")
        kind = m.lastgroup
        out.append((kind, m.group(kind), m.start(kind)))
        pos = m.end()
    return out


class _Parser:
    """Recursive descent over the registry. No eval: names come only from OPS,
    TERMINALS and SLOTS, numbers only from the number token."""

    def __init__(self, text, allow_slots, allow_hidden):
        self.toks, self.k, self.nodes = _tokens(text), 0, []
        self.allow_slots, self.allow_hidden = allow_slots, allow_hidden

    def peek(self):
        return self.toks[self.k] if self.k < len(self.toks) else ("end", "", len(self.toks))

    def symbol(self, ch, context):
        kind, val, pos = self.peek()
        if kind != "sym" or val != ch:
            raise GrammarError(f"at {pos}: expected {ch!r}, got {val or 'end of text'!r}. {context}")
        self.k += 1

    def expr(self, want):
        kind, val, pos = self.peek()
        if kind == "end":
            raise GrammarError(f"expression ends early: expected {want.__name__}. {_allowed(want)}")
        self.k += 1
        if kind == "num":
            if want not in SLOTS:
                raise GrammarError(f"at {pos}: a number cannot stand for {want.__name__}. {_allowed(want)}")
            self.nodes.append(gp.Terminal(SLOTS[want].check(val), False, want))
            return
        if kind == "sym":
            raise GrammarError(f"at {pos}: unexpected {val!r}, expected {want.__name__}. {_allowed(want)}")
        if val in SLOT_TYPE_BY_NAME:
            if not self.allow_slots:
                raise GrammarError(f"at {pos}: {val} is a placeholder; write a number. {_allowed(want)}")
            if SLOT_TYPE_BY_NAME[val] is not want:
                raise GrammarError(f"at {pos}: {val} fills {SLOT_TYPE_BY_NAME[val].__name__}, "
                                   f"expected {want.__name__}. {_allowed(want)}")
            self.nodes.append(PSET.mapping[val])
            return
        if val in TERMINALS:
            if TERMINALS[val][0] is not want:
                raise GrammarError(f"at {pos}: {val} is {TERMINALS[val][0].__name__}, "
                                   f"expected {want.__name__}. {_allowed(want)}")
            self.nodes.append(PSET.mapping[val])
            return
        op = OPS.get(val)
        if op is None or not (op.selectable or self.allow_hidden):
            raise GrammarError(f"at {pos}: unknown name {val!r}. {_allowed(want)}")
        if op.ret is not want:
            raise GrammarError(f"at {pos}: {_sig(op)}, but {want.__name__} is expected here. {_allowed(want)}")
        self.nodes.append(PSET.mapping[val] if op.selectable else _HIDDEN[val])
        self.symbol("(", f"Signature: {_sig(op)}.")
        for n, a in enumerate(op.args):
            if n:
                self.symbol(",", f"Signature: {_sig(op)}.")
            self.expr(a)
        self.symbol(")", f"Signature: {_sig(op)}.")


def parse(text, allow_slots=True, _allow_hidden=False):
    """Text -> PrimitiveTree, or GrammarError with a message meant for the LLM.
    allow_slots=False demands numbers everywhere (arms where the LLM sets the numbers)."""
    p = _Parser(text, allow_slots, _allow_hidden)
    p.expr(Rule)
    kind, val, pos = p.peek()
    if kind != "end":
        raise GrammarError(f"at {pos}: extra text after the expression: {val!r}")
    tree = gp.PrimitiveTree(p.nodes)
    problems = limit_violations(tree)
    if problems:
        raise GrammarError("rule too large: " + "; ".join(problems))
    return tree
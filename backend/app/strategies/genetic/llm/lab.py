"""Shared bench for every arm: data, the ledger of backtests, Optuna tuning, the verifier
that promotes or rejects rules, the final choice and the report.

Bars:   | train (70 % of search) | validation |  hold-out  |
        |<--------------- search ------------->|
Search-time decisions see train and validation only; the hold-out is read by report().

Budget: one unit per distinct rule backtested on the search window (a ledger row), whoever
asked for it: a GP individual, an LLM proposal, an Optuna trial or a jitter probe of the
verifier. Probes with doubled costs or a one-bar delay reuse the stored positions and are
free. The final choice (select_final) is the same for every arm and spends nothing.
"""
from __future__ import annotations

import hashlib
import math
import random
import warnings
import zlib
from collections import deque
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import optuna

import app.strategies.genetic.strategy_gp as sg

BARS_PER_YEAR = 24 * 365
COST_BPS = 12                          # fee + slippage per side
N_SEARCH, N_HOLD = 20_000, 8_000
TRAIN_FRACTION = 0.7
MIN_TRADES = {"train": 14, "val": 6}   # gates; a rule below them on train has fitness -1
EDGE = 0.15                            # planted drift, in units of bar volatility
PLANTED_RULE = "long_flat(osc_below(rsi(close, 14), 30), osc_above(rsi(close, 14), 30))"

optuna.logging.set_verbosity(optuna.logging.ERROR)


def synthetic(seed, edge, n=N_SEARCH + N_HOLD, sigma=0.01):
    """Tick-rounded hourly bars; drift +edge*sigma after RSI(14, Cutler) < 30 and
    -edge*sigma after > 70, so buy-and-hold earns nothing from the planted effect."""
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


def segment_stats(pos, close, sl, cost_bps=None):
    """Statistics of a long/flat position series on one window. Returns are computed on the
    whole series first, so a position carried into the window is priced correctly.
    cost_bps and the annualization follow lab.COST_BPS and lab.BARS_PER_YEAR at call time."""
    net = sg.net_returns(pos, close, COST_BPS if cost_bps is None else cost_bps)[sl]
    gross = sg.net_returns(pos, close, 0.0)[sl]
    p = pos[sl]
    trades = sg.n_trades(p)
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    return {"sharpe": sg.sharpe(net, BARS_PER_YEAR), "gross": sg.sharpe(gross, BARS_PER_YEAR),
            "trades": trades, "exposure": float(p.mean()), "holding": float(p.sum()) / max(trades, 1),
            "mdd": float(np.max(1.0 - equity / peak)), "ret": float(equity[-1] - 1.0)}


class Ledger:
    """One row per distinct rule backtested on the search window. A new row spends one unit
    of budget (unless charge=False); a repeated backtest is free and returns the stored row.

    row: rule, key, role ("candidate", "trial" or "probe"), charged, stats per window
    (train, val, search), sr_obs (per-bar Sharpe on search, for DSR), behaviour (hash of the
    position path) and bits (the positions, packed)."""

    def __init__(self, data, budget, n_search=N_SEARCH, train_fraction=TRAIN_FRACTION):
        if len(data["close"]) < n_search:
            raise ValueError(f"data has {len(data['close'])} bars, the search window needs {n_search}")
        self.budget, self.n_search, self.charged = budget, n_search, 0
        n_train = int(n_search * train_fraction)
        self.segments = {"train": slice(0, n_train), "val": slice(n_train, n_search), "search": slice(0, n_search)}
        self.data = {k: np.asarray(v[:n_search], dtype=float) for k, v in data.items()}
        self.search = sg.Evaluator(self.data)
        self.close = self.data["close"]
        self.rows = {}
        self.structures = set()               # templates tried, for the report

    @property
    def used(self):
        return self.charged

    @staticmethod
    def key(tree, params=()):
        return sg.to_text(sg.bind(tree, params), canonical=True)

    def row(self, tree, params=()):
        return self.rows.get(self.key(tree, params))

    def positions(self, row):
        return np.unpackbits(row["bits"], count=self.n_search).astype(float)

    def returns(self, row, cost_bps=None):
        return sg.net_returns(self.positions(row), self.close, COST_BPS if cost_bps is None else cost_bps)

    def run(self, tree, params=(), role="candidate"):
        """Search fitness of a rule (Sharpe on train): the objective of every arm."""
        return fitness(self.backtest(tree, params, role))

    def backtest(self, tree, params=(), role="candidate", charge=True):
        bound = sg.bind(tree, params)
        key = sg.to_text(bound, canonical=True)
        row = self.rows.get(key)
        if row is None:
            if charge and self.charged >= self.budget:
                raise BudgetExhausted
            pos = sg.positions(*self.search.signals(bound))
            r = sg.net_returns(pos, self.close, COST_BPS)
            sd = float(r.std(ddof=1))
            row = self.rows[key] = {
                "rule": bound, "key": key, "role": role, "charged": bool(charge),
                "stats": {name: segment_stats(pos, self.close, sl) for name, sl in self.segments.items()},
                "sr_obs": 0.0 if sd == 0 else float(r.mean()) / sd,
                "behaviour": hashlib.blake2b(pos.tobytes(), digest_size=8).hexdigest(),
                "bits": np.packbits(pos.astype(bool))}
            self.charged += bool(charge)
        elif row["role"] == "probe" and role != "probe":
            row["role"] = role                # a probe that a generator proposed later is its candidate
        return row


def fitness(row):
    t = row["stats"]["train"]
    return t["sharpe"] if t["trades"] >= MIN_TRADES["train"] else -1.0


def tune(tree, ledger, k, start=None, role="trial"):
    """Optuna fills the free slots of one template on train: k trials, at most k units.
    start: {name: value} tried first (arm 4 starts from the LLM's numbers; that trial repeats
    a backtest already in the ledger, so it is free). Returns (best fitness, best values in
    slot_names order)."""
    ledger.structures.add(sg.to_text(tree))
    names = sg.slot_names(tree)
    if not names:
        return ledger.run(tree, (), role="candidate"), []
    sampler = optuna.samplers.TPESampler(seed=zlib.crc32(sg.to_text(tree).encode()), n_startup_trials=4)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if start:
        study.enqueue_trial({n: start[n] for n in names if n in start})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # a start value off Optuna's float grid is still used as is
        study.optimize(lambda trial: ledger.run(tree, sg.suggest(trial, tree), role=role), n_trials=k)
    best = max((t for t in study.trials if t.value is not None), key=lambda t: t.value)
    return best.value, [best.params[n] for n in names]


# ------------------------------------------------------------- verifier --
def jittered(rule, n, size, rng):
    """Up to n copies of a concrete rule with every number moved by about +-size (tied
    numbers move together), kept inside the slot ranges."""
    template, values = sg.lift(rule)
    if not values:
        return []
    out, seen = [], {sg.to_text(rule)}
    for _ in range(4 * n):
        if len(out) == n:
            break
        moved = {}
        for name, v in values.items():
            slot = sg.SLOTS[sg.SLOT_TYPE_BY_NAME[name]]
            sign = rng.choice((-1, 1))
            if slot.kind == "int":
                step = max(1, round(abs(v) * size))
                moved[name] = int(min(max(v + sign * step, slot.lo), slot.hi))
            else:
                x = min(max(v + sign * max(slot.step, abs(v) * size), slot.lo), slot.hi)
                x = slot.lo + round((x - slot.lo) / slot.step) * slot.step
                moved[name] = round(min(max(x, slot.lo), slot.hi), 6)
        bound = sg.bind(template, moved)
        text = sg.to_text(bound)
        if text not in seen:
            seen.add(text)
            out.append(bound)
    return out


def sharpe_diff_se(a, b, periods_per_year=None):
    """Standard error of the difference of two annualized Sharpe ratios measured on the same
    bars (Jobson & Korkie 1981, with Memmel's 2003 correction). Small for near copies of a
    rule, large for rules that behave differently. Annualized with lab.BARS_PER_YEAR by default."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    rho = float(np.corrcoef(a, b)[0, 1])
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    var = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / len(a)
    return math.sqrt(max(var, 0.0) * (periods_per_year or BARS_PER_YEAR))


@dataclass
class Verdict:
    decision: str                     # ADOPT, INCUBATE or REJECT
    reason: str
    q: float = -math.inf              # soft score; -inf when a gate failed
    threshold: float = math.nan       # what the candidate had to reach
    probes: dict = field(default_factory=dict)   # validation Sharpe under each probe


class Verifier:
    """Gates, stress probes and the soft score Q on validation, shared by every arm.

    Q = S_val - gap * max(0, S_train - S_val) - stress * max(0, S_val - worst probe)
        - dd * max(0, MDD_val - dd_free)
    Probes: doubled costs and a one-bar delay (recomputed from the stored positions) and
    `jitter` copies of the rule with every number moved by about +-jitter_size (new rules,
    so ledger rows that spend budget when charge=True).
    Against an incumbent, a candidate is ADOPTED when its Q reaches the incumbent's Q plus
    z standard errors of the difference of the two Sharpe ratios, INCUBATED when it beats
    the incumbent by less, REJECTED otherwise.
    """

    def __init__(self, ledger, *, z=0.5, gap=0.5, stress=0.5, dd=2.0, dd_free=0.2, max_dd=0.35,
                 jitter=2, jitter_size=0.1, charge=True):
        self.ledger, self.z, self.gap, self.stress, self.dd = ledger, z, gap, stress, dd
        self.dd_free, self.max_dd, self.jitter, self.jitter_size = dd_free, max_dd, jitter, jitter_size
        self.charge = charge

    def gates(self, row):
        s, out = row["stats"], []
        for window in ("train", "val"):
            if s[window]["trades"] < MIN_TRADES[window]:
                out.append(f"{s[window]['trades']} trades on {window} (needs {MIN_TRADES[window]})")
        if s["val"]["mdd"] > self.max_dd:
            out.append(f"validation drawdown {s['val']['mdd']:.0%} (limit {self.max_dd:.0%})")
        return out

    def probes(self, row):
        val, pos, close = self.ledger.segments["val"], self.ledger.positions(row), self.ledger.close
        out = {"costs x2": sg.sharpe(sg.net_returns(pos, close, 2 * COST_BPS)[val], BARS_PER_YEAR),
               "1-bar delay": sg.sharpe(sg.net_returns(np.r_[0.0, pos[:-1]], close, COST_BPS)[val], BARS_PER_YEAR)}
        rng = random.Random(zlib.crc32(row["key"].encode()))      # the same probes every time
        for i, rule in enumerate(jittered(row["rule"], self.jitter, self.jitter_size, rng), 1):
            probe = self.ledger.backtest(rule, role="probe", charge=self.charge)
            out[f"jitter {i}"] = probe["stats"]["val"]["sharpe"]
        return out

    def score(self, row, probes):
        s = row["stats"]
        val, train = s["val"]["sharpe"], s["train"]["sharpe"]
        worst = min(probes.values()) if probes else val
        return (val - self.gap * max(0.0, train - val) - self.stress * max(0.0, val - worst)
                - self.dd * max(0.0, s["val"]["mdd"] - self.dd_free))

    def verify(self, row, incumbent_row=None, incumbent_q=-math.inf):
        failed = self.gates(row)
        if failed:
            return Verdict("REJECT", "; ".join(failed))
        probes = self.probes(row)
        q = self.score(row, probes)
        if incumbent_row is None or not math.isfinite(incumbent_q):
            return Verdict("ADOPT", "passed the gates", q, -math.inf, probes)
        val = self.ledger.segments["val"]
        se = sharpe_diff_se(self.ledger.returns(row)[val], self.ledger.returns(incumbent_row)[val])
        threshold = incumbent_q + self.z * se
        if q >= threshold:
            return Verdict("ADOPT", f"Q {q:.2f} reaches {threshold:.2f}", q, threshold, probes)
        if q > incumbent_q:
            return Verdict("INCUBATE", f"Q {q:.2f} beats the current {incumbent_q:.2f} but not {threshold:.2f}",
                           q, threshold, probes)
        return Verdict("REJECT", f"Q {q:.2f} does not beat the current {incumbent_q:.2f}", q, threshold, probes)


def select_final(ledger, m=10, **verifier_options):
    """The rule an arm hands in: of the m rules with the best train fitness, the one with the
    best Q on validation among those that pass the gates. Same for every arm; spends no budget.
    Returns (row, verdict) or None."""
    verifier = Verifier(ledger, charge=False, **verifier_options)
    pool = sorted((r for r in ledger.rows.values() if r["role"] != "probe"), key=fitness, reverse=True)[:m]
    best = None
    for row in pool:
        verdict = verifier.verify(row)
        if verdict.decision == "ADOPT" and (best is None or verdict.q > best[1].q):
            best = (row, verdict)
    return best


# --------------------------------------------------------------- report --
def deflated_sharpe(sr, n, v, returns):
    """Bailey & Lopez de Prado (2014). sr: per-observation Sharpe of the selected rule,
    n: number of trials (>= 2), v: variance of the trials' per-observation Sharpe ratios."""
    g, nd = 0.5772156649015329, NormalDist()
    sr0 = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * math.e)))
    z = (returns - returns.mean()) / returns.std()
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    return nd.cdf((sr - sr0) * math.sqrt(len(returns) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


def hold_out(data, rule, n_search=N_SEARCH):
    pos = sg.positions(*sg.Evaluator(data).signals(rule))
    r = sg.net_returns(pos, data["close"], COST_BPS)[n_search:]
    return sg.sharpe(r, BARS_PER_YEAR), sg.n_trades(pos[n_search:])


def summarize(data, ledger, reference=None):
    """What an arm hands in and how it holds up, as a flat dict: the chosen rule (select_final),
    its train, validation and hold-out Sharpe, and its DSR under two conventions for (N, V).
    The ledger variance mixes estimation noise with real differences between rules (cost-crushed
    junk), which raises the bar for everyone; V = 1/T is the variance under the null.
    reference: an optional rule text measured the same way (the planted rule of synthetic data).
    Spends no budget."""
    rows = [r for r in ledger.rows.values() if r["role"] != "probe"]
    out = {"structures": len(ledger.structures), "rules": len(rows), "budget used": ledger.used, "rule": None}
    chosen = select_final(ledger)
    if chosen is None:
        return out
    row, verdict = chosen
    n = ledger.n_search
    distinct = {}                     # one Sharpe per distinct position path among tradable rules
    for r in rows:
        if r["stats"]["train"]["trades"] >= MIN_TRADES["train"]:
            distinct.setdefault(r["behaviour"], r["sr_obs"])
    r_best = ledger.returns(row)
    sr_best = float(r_best.mean() / r_best.std(ddof=1))
    ho, ho_trades = hold_out(data, row["rule"], n)
    s = row["stats"]
    out.update({"rule": sg.to_text(row["rule"]), "train sharpe": s["train"]["sharpe"],
                "train trades": s["train"]["trades"], "val sharpe": s["val"]["sharpe"],
                "val trades": s["val"]["trades"], "Q": verdict.q, "hold-out sharpe": ho,
                "hold-out trades": ho_trades, "distinct tradable": len(distinct)})
    conventions = {}
    if len(distinct) >= 2:
        conventions["null V"] = (len(distinct), 1.0 / (n - 1))
    if len(rows) >= 2:
        v = float(np.var([r["sr_obs"] for r in rows], ddof=1))
        if v > 0:
            conventions["ledger V"] = (len(rows), v)
    for label, (k, v) in conventions.items():
        out[f"DSR {label}"] = deflated_sharpe(sr_best, k, v, r_best)
        out[f"N {label}"] = k
    if reference:
        ref = sg.parse(reference)
        r_ref = sg.net_returns(sg.positions(*ledger.search.signals(ref)), ledger.close, COST_BPS)
        out["reference search sharpe"] = sg.sharpe(r_ref, BARS_PER_YEAR)
        out["reference hold-out sharpe"] = hold_out(data, ref, n)[0]
        if r_ref.std() > 0:
            for label, (k, v) in conventions.items():
                out[f"reference DSR {label}"] = deflated_sharpe(float(r_ref.mean() / r_ref.std(ddof=1)), k, v, r_ref)
    return out


def report(name, data, ledger, note="", reference=PLANTED_RULE):
    """Print summarize(): the rule the arm hands in, its hold-out Sharpe and its DSR."""
    s = summarize(data, ledger, reference)
    print(f"\n== {name}: {note + ', ' if note else ''}{s['structures']} structures, "
          f"{s['rules']} rules backtested, {s['budget used']} budget units")
    if s["rule"] is None:
        print("no rule passed the gates")
        return
    print(f"chosen rule:  {s['rule']}")
    print(f"train Sharpe {s['train sharpe']:.2f} ({s['train trades']} trades), validation "
          f"{s['val sharpe']:.2f} ({s['val trades']} trades), Q {s['Q']:.2f}")
    print(f"hold-out Sharpe {s['hold-out sharpe']:.2f} ({s['hold-out trades']} trades)")
    if reference:
        print(f"reference rule: search Sharpe {s['reference search sharpe']:.2f}, "
              f"hold-out Sharpe {s['reference hold-out sharpe']:.2f}")
    for label in ("null V", "ledger V"):
        if f"DSR {label}" in s:
            ref = f", reference rule {s[f'reference DSR {label}']:.3f}" if f"reference DSR {label}" in s else ""
            print(f"DSR [{label}, N={s[f'N {label}']}]: chosen {s[f'DSR {label}']:.3f}{ref}")

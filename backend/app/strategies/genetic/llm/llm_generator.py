"""LLM strategy generator for arms 1, 3 and 4, after EvoQuant (arXiv 2607.12455).

One line of descent per classic strategy (strategy_gp.CLASSICS). Each round, for one line:

  evidence of the current rule -> diagnosis: up to two flags from a fixed list, in code
  -> plan: the highest edit level allowed (repair -> bridge -> redesign -> migration, raised
     after `escalate_after` rounds without an adoption)
  -> the model proposes candidates as JSON: rule, layer, level, mechanism, expected moves
  -> sg.parse (grammar, types, limits) -> edit contract (declared vs actual level and layer)
  -> repeat check -> fit: one backtest (arms 1, 4) or Optuna on train (arm 3)
  -> behavioural drift limit of the level -> lab.Verifier: gates, probes, Q on validation
  -> ADOPT replaces the current rule, INCUBATE keeps a variant to edit later, REJECT
  -> memory: what worked for which flag, what failed on this line, and whether the moves
     the model predicted came true.
Parse errors, contract breaks and repeats go back to the model in one repair turn.

Arm 1: the model writes the numbers; all four levels.
Arm 3: the model writes templates with placeholders; Optuna tunes them on train; no repair.
Arm 4: as arm 1 until (1 - reserve) of the budget is spent, then Optuna recalibrates the
       numbers of the best current rules, starting from the model's values.

The model sees rules and their train/validation evidence, never prices, dates, the asset or
the hold-out window; its text goes through sg.parse and is never executed. Any ChatModel
from llm_models works; MockModel below runs the pipeline offline.

Run:  python llm_generator.py --arm 1                      (Claude, needs ANTHROPIC_API_KEY)
      python llm_generator.py --arm 3 --model claude-haiku-5-5 --effort medium
      python llm_generator.py --arm 1 --provider openai --base-url http://localhost:8080/v1 --model qwen
      python llm_generator.py --arm 4 --provider mock      (offline plumbing check)
      python llm_generator.py --arm 1 --replay             (re-run from the call log, no calls)
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from deap import gp

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_models as lm
import app.strategies.genetic.strategy_gp as sg

LEVELS = ("repair", "bridge", "redesign", "migration")
LAYERS = ("entry", "exit")
MOVES = ("trades", "holding", "exposure", "val_sharpe")
DRIFT_LIMIT = {"repair": 0.6, "bridge": 1.0, "redesign": 2.0, "migration": 2.0}   # 1 - corr of positions
TREND_WINDOW = 200

# ------------------------------------------------------------ semantics --
# Mechanism of a rule = mechanism of its entry, read from the operators it uses.
FAMILY_OF = {"rsi": "reversion", "stoch": "reversion", "ibs": "reversion", "zscore": "reversion",
             "bb_dn": "reversion", "cross_up": "trend", "mom": "trend", "rmax": "breakout",
             "rmin": "breakout", "bb_up": "breakout", "vol_regime": "volatility", "vol_z": "volume",
             "gt_v": "volume", "sma_v": "volume"}
FAMILIES = ("volatility", "volume", "breakout", "reversion", "trend")     # this order breaks ties


def layers(tree):
    """(entry, exit) of a long_flat rule as separate trees."""
    first = tree.searchSubtree(1)
    second = tree.searchSubtree(first.stop)
    return gp.PrimitiveTree(tree[first]), gp.PrimitiveTree(tree[second])


def family(tree):
    """Mechanism of the entry condition (a heuristic over operator names); a plain moving
    average counts as trend only when nothing more specific is there."""
    names = [node.name for node in layers(tree)[0]]
    counts = Counter(FAMILY_OF[n] for n in names if n in FAMILY_OF)
    if counts:
        return max(FAMILIES, key=lambda f: (counts[f], -FAMILIES.index(f)))
    return "trend" if "sma" in names else "other"


def skeleton(tree):
    """Prefix tokens of a rule with every number or placeholder replaced by its type: rules
    that differ only in numbers share a skeleton."""
    return " ".join(f"<{node.ret.__name__}>" if isinstance(node, gp.Terminal) and node.ret in sg.SLOTS
                    else node.name for node in tree)


@dataclass(frozen=True)
class Edit:
    layers: tuple        # parts whose text changed, in the order entry, exit
    level: str           # the smallest level that explains the change
    syntactic: float     # 1 - similarity of the skeleton token sequences


def analyse_edit(old, new):
    """What a proposal actually changes relative to the rule it edits. repair: numbers only;
    bridge: the structure of one part; redesign: of both parts; migration: another mechanism."""
    pairs = list(zip(LAYERS, layers(old), layers(new)))
    changed = tuple(name for name, a, b in pairs if sg.to_text(a) != sg.to_text(b))
    structural = [name for name, a, b in pairs if skeleton(a) != skeleton(b)]
    if not structural:
        level = "repair"
    elif family(old) != family(new):
        level = "migration"
    elif len(structural) == 1:
        level = "bridge"
    else:
        level = "redesign"
    ratio = difflib.SequenceMatcher(None, skeleton(old).split(), skeleton(new).split(), autojunk=False).ratio()
    return Edit(changed, level, 1.0 - ratio)


@dataclass
class Proposal:
    rule: str
    layer: str = ""
    level: str = ""
    mechanism: str = ""
    expected: dict = field(default_factory=dict)
    problem: str = ""


@dataclass(frozen=True)
class Plan:
    level: str           # highest level allowed this round
    focus: tuple         # parts the diagnosis points at (guidance, not enforced)
    flags: tuple


def contract_problems(p, edit, plan, arm):
    """Reasons a proposal breaks its edit contract; empty if it keeps it."""
    if not edit.layers:
        return ["the rule is the current rule itself"]
    rank, out = LEVELS.index, []
    if rank(edit.level) > rank(plan.level):
        out.append(f"this is a {edit.level} edit, but the plan allows {plan.level} at most")
    if p.level not in LEVELS:
        out.append(f"level must be one of {', '.join(LEVELS)}")
    elif rank(edit.level) > rank(p.level):
        out.append(f"declared {p.level}, but the change is a {edit.level} edit")
    declared = {"entry": ("entry",), "exit": ("exit",), "both": LAYERS}.get(p.layer)
    if declared is None:
        out.append("layer must be entry, exit or both")
    elif any(part not in declared for part in edit.layers):
        out.append(f"changes {' and '.join(edit.layers)}, but declares {p.layer}")
    if arm == 3 and edit.level == "repair":
        out.append("numbers are tuned by the optimizer; change the structure instead")
    return out


def parse_reply(text):
    """Proposals from the model's answer: the first JSON list of objects in it (code fences and
    surrounding prose are tolerated). Raises ValueError if there is none."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [_proposal(i, x) for i, x in enumerate(data, 1)]
    raise ValueError("no JSON list of proposals found")


def _proposal(i, item):
    rule = item.get("rule")
    if not isinstance(rule, str) or not rule.strip():
        return Proposal("", problem=f"proposal {i} has no 'rule' string")
    expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
    expected = {m: str(expected[m]).lower() for m in MOVES
                if m in expected and str(expected[m]).lower() in ("up", "down", "same")}
    return Proposal(rule.strip().strip("`").strip(), str(item.get("layer", "")).lower().strip(),
                    str(item.get("level", "")).lower().strip(), str(item.get("mechanism", ""))[:300], expected)


# ------------------------------------------------------------ diagnosis --
def trade_diagnostics(pos, close, sl, trend_window=TREND_WINDOW):
    """(give-back, against-trend share) on one window: the mean share of its best open profit a
    trade gives back before the exit (needs 10 trades), and the share of the loss of losing
    trades that comes from entries below the trend_window-bar average (needs 5 losers)."""
    trend = pd.Series(close).rolling(trend_window, min_periods=trend_window).mean().to_numpy()[sl]
    p, c = pos[sl], close[sl]
    d = np.diff(np.r_[0.0, p, 0.0])
    starts, ends = np.flatnonzero(d > 0), np.minimum(np.flatnonzero(d < 0), len(c) - 1)
    givebacks, loss, against, losers = [], 0.0, 0.0, 0
    for e, x in zip(starts, ends):
        path = c[e:x + 1] / c[e] - 1.0
        best, final = float(path.max()), float(path[-1])
        if best > 0:
            givebacks.append(min((best - final) / best, 1.0))
        if final < 0:
            losers += 1
            loss -= final
            if np.isfinite(trend[e]) and c[e] <= trend[e]:
                against -= final
    giveback = float(np.mean(givebacks)) if len(starts) >= 10 and givebacks else 0.0
    share = against / loss if losers >= 5 and loss > 0 else 0.0
    return giveback, share


# flag: (what the model is told, severity; the flag fires when severity > 0)
DIAGNOSIS = {
    "sparse_entry": ("too few trades: the entry is too strict or fires too rarely",
                     lambda ev: max((30 - ev["train"]["trades"]) / 30, (0.02 - ev["train"]["exposure"]) / 0.02)),
    "delayed_exit": ("trades give back most of their best open profit before the exit",
                     lambda ev: ev["giveback"] - 0.6),
    "excessive_drawdown": ("the drawdown is large relative to the return",
                           lambda ev: min(ev["train"]["mdd"] - 0.2, ev["train"]["mdd"] - max(ev["train"]["ret"], 0.0))),
    "low_trend_filter": ("most of the loss comes from entries below the long-term average",
                         lambda ev: ev["against_trend"] - 0.65),
    "overfit_validation": ("much better on train than on validation",
                           lambda ev: ev["train"]["sharpe"] - ev["val"]["sharpe"] - 1.0 if ev["train"]["sharpe"] > 0 else -1.0),
    "parameter_sensitivity": ("slightly changed numbers lose the validation edge",
                              lambda ev: ev["jitter_drop"] - 1.0),
    "cost_drag": ("costs eat most of the gross edge: too many or too short trades",
                  lambda ev: max(ev["train"]["gross"] - ev["train"]["sharpe"] - 0.75,
                                 0.5 if ev["train"]["gross"] > 0 >= ev["train"]["sharpe"] else -1.0)),
}
FOCUS = {"sparse_entry": ("entry",), "low_trend_filter": ("entry",),
         "delayed_exit": ("exit",), "excessive_drawdown": ("exit",)}


def diagnose(ev, top=2):
    """Up to `top` flags that fire, most severe first. Computed in code, not by the model."""
    fired = [(fn(ev), flag) for flag, (_, fn) in DIAGNOSIS.items()]
    fired = sorted((x for x in fired if x[0] > 0), key=lambda x: -x[0])
    return [flag for _, flag in fired[:top]]


def evidence(ledger, entry):
    """What the diagnosis and the prompt see about a rule: train and validation statistics,
    trade give-back, against-trend losses, the probes of its verdict and the jitter drop."""
    row, verdict = entry.row, entry.verdict
    giveback, against = trade_diagnostics(ledger.positions(row), ledger.close, ledger.segments["train"])
    jitter = [v for k, v in verdict.probes.items() if k.startswith("jitter")]
    drop = row["stats"]["val"]["sharpe"] - min(jitter) if jitter else 0.0
    return {"train": row["stats"]["train"], "val": row["stats"]["val"], "giveback": giveback,
            "against_trend": against, "jitter_drop": max(drop, 0.0), "q": verdict.q,
            "gates": "" if math.isfinite(verdict.q) else verdict.reason, "probes": verdict.probes}


def actual_moves(old, new, tolerance=0.1):
    """How a change moved trades, holding and exposure on the search window and the validation
    Sharpe: 'same' within tolerance (relative; absolute 0.1 for the Sharpe)."""
    def move(a, b, scale):
        if abs(b - a) <= tolerance * scale:
            return "same"
        return "up" if b > a else "down"
    so, sn = old["stats"]["search"], new["stats"]["search"]
    return {"trades": move(so["trades"], sn["trades"], max(so["trades"], 1)),
            "holding": move(so["holding"], sn["holding"], max(so["holding"], 1.0)),
            "exposure": move(so["exposure"], sn["exposure"], max(so["exposure"], 0.01)),
            "val_sharpe": move(old["stats"]["val"]["sharpe"], new["stats"]["val"]["sharpe"], 1.0)}


def behaviour_drift(ledger, old, new):
    """1 - correlation of the two position series on the search window (0 same behaviour,
    1 unrelated, 2 opposite); 0 when the old rule never trades."""
    a, b = ledger.positions(old), ledger.positions(new)
    if a.std() == 0:
        return 0.0
    if b.std() == 0:
        return 1.0
    return float(1.0 - np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------- memory --
@dataclass
class Case:
    line: str
    family: str
    flags: tuple
    level: str
    layers: tuple
    rule: str
    mechanism: str
    decision: str
    reason: str
    q: float
    gain: float | None        # Q minus the current rule's Q; None when either failed the gates
    hit_rate: float | None    # share of the model's predicted moves that came true


class Memory:
    """Experience of a run: proven edits per flag, and the failures and tests of each line."""

    def __init__(self):
        self.cases = []

    def add(self, case):
        self.cases.append(case)

    def worked(self, flags, family_, n=3):
        hits = [c for c in self.cases if c.decision == "ADOPT" and set(c.flags) & set(flags)]
        return sorted(hits, key=lambda c: (c.family != family_, -(c.gain or 0.0)))[:n]

    def failed(self, line, n=3):
        return [c for c in reversed(self.cases) if c.line == line and c.decision == "REJECT"][:n]

    def tried(self, line, n=6):
        return [c.rule for c in reversed(self.cases) if c.line == line][:n]


# -------------------------------------------------------------- prompts --
def system_prompt(arm, max_dd=0.35):
    levels = [("repair", "change numbers only"),
              ("bridge", "change the structure of one part, the entry or the exit: add, remove or replace a condition"),
              ("redesign", "change the structure of both parts, keeping the mechanism"),
              ("migration", "switch the entry to another mechanism")]
    level_lines = ["- repair: not available; an optimizer sets the numbers" if arm == 3 and name == "repair"
                   else f"- {name}: {text}" for name, text in levels]
    numbers = ("Do not choose numbers: write placeholders (W1, W2, LVL1, ...) wherever a number goes. An optimizer "
               "tunes them on train for every proposal, and you will see the values it found. The same "
               "placeholder twice means one shared value."
               if arm == 3 else "Write every number yourself; placeholders such as W1 are not allowed.")
    return "\n".join([
        "You improve long/flat trading rules for one market traded on hourly bars. You never see the data: "
        "a program backtests every rule you propose on a search period split into an earlier train part and a "
        "later validation part, and reports evidence. A final period is kept for the test and never shown.",
        "",
        "A rule is one expression in the grammar below. long_flat(entry, exit) buys at the close of a bar where "
        "`entry` is true and sells at the close of a bar where `exit` is true; in between it holds, otherwise it "
        f"is flat. Costs are {lab.COST_BPS / 1e4:.2%} per side.",
        "",
        "How a proposal is judged:",
        f"- gates: at least {lab.MIN_TRADES['train']} trades on train and {lab.MIN_TRADES['val']} on validation, "
        f"validation drawdown at most {max_dd:.0%};",
        "- score Q: validation Sharpe, minus penalties for a train/validation gap, for losing the edge under "
        "doubled costs, a one-bar delay or slightly changed numbers, and for drawdown;",
        "- a proposal replaces the current rule only if its Q beats the current Q by a margin that grows when "
        "the proposal behaves differently from the current rule.",
        "",
        "Edit levels (the plan names the highest level allowed in a round):",
        *level_lines,
        "Mechanisms: reversion, trend, breakout, volatility, volume; the entry decides the mechanism.",
        numbers,
        "",
        "GRAMMAR",
        sg.catalog(),
        "",
        "ANSWER FORMAT",
        "Return only a JSON list with one object per proposal:",
        '[{"rule": "<the expression on one line>",',
        '  "layer": "entry" | "exit" | "both",',
        '  "level": "repair" | "bridge" | "redesign" | "migration",',
        '  "mechanism": "<one sentence: which market behaviour the change exploits>",',
        '  "expected": {"trades": "up" | "down" | "same", "holding": "up" | "down" | "same",',
        '               "exposure": "up" | "down" | "same", "val_sharpe": "up" | "down" | "same"}}]',
        "`layer` names every part you changed. `expected` says how the change should move the number of "
        "trades, the mean holding time, the time in the market and the validation Sharpe, compared with the "
        "current rule.",
    ])


def repair_prompt(problems, wanted):
    lines = ["Some proposals were not accepted:"]
    lines += [f"- `{rule}`: {why}" if rule else f"- {why}" for rule, why in problems]
    lines.append(f"Send {wanted} replacement proposals as a JSON list in the same format.")
    return "\n".join(lines)


def evidence_table(ev):
    t, v = ev["train"], ev["val"]
    rows = [("Sharpe after costs", f"{t['sharpe']:.2f}", f"{v['sharpe']:.2f}"),
            ("Sharpe before costs", f"{t['gross']:.2f}", f"{v['gross']:.2f}"),
            ("trades", str(t["trades"]), str(v["trades"])),
            ("mean holding, bars", f"{t['holding']:.0f}", f"{v['holding']:.0f}"),
            ("time in market", f"{t['exposure']:.0%}", f"{v['exposure']:.0%}"),
            ("max drawdown", f"{t['mdd']:.0%}", f"{v['mdd']:.0%}")]
    out = [f"  {name:<20} {a:>7} | {b:>7}" for name, a, b in rows]
    if math.isfinite(ev["q"]):
        stress = ", ".join(f"{k} {x:.2f}" for k, x in ev["probes"].items())
        out.append(f"  Q {ev['q']:.2f}; validation Sharpe under stress: {stress}")
    else:
        out.append(f"  fails the gates: {ev['gates']}")
    out.append(f"  trades give back {ev['giveback']:.0%} of their best open profit; {ev['against_trend']:.0%} of "
               f"the loss of losing trades comes from entries below the {TREND_WINDOW}-bar average")
    return out


def _fmt(q):
    return f"{q:.2f}" if math.isfinite(q) else "n/a (fails the gates)"


def _gain(case):
    return f"Q {case.gain:+.2f}" if case.gain is not None else "fixed a rule that failed the gates"


def _r(x):
    return round(float(x), 4) if x is not None and math.isfinite(x) else None


# ------------------------------------------------------------ generator --
@dataclass
class Entry:
    tree: gp.PrimitiveTree          # the concrete rule (what is traded)
    template: gp.PrimitiveTree      # what the model edits: the rule (arms 1, 4) or its template (arm 3)
    values: dict                    # arm 3: tuned values of the template's placeholders
    row: dict                       # ledger row of the concrete rule
    verdict: lab.Verdict | None = None

    @property
    def q(self):
        return self.verdict.q if self.verdict else -math.inf


@dataclass
class Line:
    name: str
    incumbent: Entry
    level: int                      # index into the generator's levels
    failures: int = 0               # rounds without an adoption at this level
    visits: int = 0
    adopted: int = 0
    incubated: list = field(default_factory=list)


class Generator:
    """One run of arm 1, 3 or 4 on one dataset. run() spends the budget and returns self;
    lines, ledger, memory, events and stats stay available for the report."""

    def __init__(self, arm, model, data, budget=2400, *, k=10, candidates=4, repairs=1, reserve=0.2,
                 finalists=5, escalate_after=2, lines=None, first_level=None, max_calls=None,
                 patience=12, seed=0, n_search=lab.N_SEARCH, verifier_options=None, bars="hourly bars"):
        if arm not in (1, 3, 4):
            raise ValueError("the LLM arms are 1, 3 and 4")
        self.arm, self.model, self.budget, self.k, self.candidates = arm, model, budget, k, candidates
        self.repairs, self.reserve, self.finalists = repairs, reserve, finalists
        self.escalate_after, self.patience, self.seed = escalate_after, patience, seed
        self.levels = LEVELS[1:] if arm == 3 else LEVELS
        self.first_level = first_level or self.levels[0]
        if self.first_level not in self.levels:
            raise ValueError(f"arm {arm} starts at one of {self.levels}")
        self.line_names = list(lines or sg.CLASSICS)
        unknown = [n for n in self.line_names if n not in sg.CLASSICS]
        if unknown:
            raise ValueError(f"unknown classic strategies: {unknown}")
        self.ledger = lab.Ledger(data, budget, n_search=n_search)
        self.verifier = lab.Verifier(self.ledger, **(verifier_options or {}))
        self.system = system_prompt(arm, self.verifier.max_dd, bars)
        self.memory = Memory()
        self.lines, self.events, self.proposed = [], [], {}
        self.stats = Counter()
        self.max_calls = max_calls or max(40, 2 * budget // max(candidates, 1))
        self.stop_reason, self.turn, self.idle, self.search_used = "", 0, 0, 0

    # ---- main loop
    def run(self):
        self.ledger.budget = self.budget - int(round(self.reserve * self.budget)) if self.arm == 4 else self.budget
        try:
            if not self.lines:
                self._start_lines()
            while self.stats["calls"] < self.max_calls and self.idle < self.patience:
                line = self.lines[self.turn % len(self.lines)]
                self.turn += 1
                self._round(line)
            self.stop_reason = "call limit" if self.stats["calls"] >= self.max_calls else "no new backtests"
        except lab.BudgetExhausted:
            self.stop_reason = "budget"
        self.search_used = self.ledger.used
        if self.arm == 4:
            self.ledger.budget = self.budget
            self._calibrate()
        return self

    def _start_lines(self):
        for name in self.line_names:
            entry = self._fit(sg.parse(sg.CLASSICS[name]))
            entry.verdict = self.verifier.verify(entry.row)
            self.lines.append(Line(name, entry, self.levels.index(self.first_level), start_q=entry.q))

    def _round(self, line):
        line.visits += 1
        base = line.incumbent
        if line.incubated and line.visits % 3 == 0:  # now and then, edit a promising variant
            base = max(line.incubated, key=lambda e: e.q)
        ev = evidence(self.ledger, base)
        flags = tuple(diagnose(ev))
        plan = Plan(self.levels[line.level], FOCUS.get(flags[0], LAYERS) if flags else LAYERS, flags)
        messages = [lm.Message("user", self.user_prompt(line, base, ev, plan))]
        before, adopted = self.ledger.used, False
        for attempt in range(self.repairs + 1):
            reply = self._ask(messages)
            fixable, wanted, adopted_now = self._handle(reply.text, line, base, plan, attempt)
            adopted = adopted or adopted_now
            if not fixable or attempt == self.repairs or not reply.text.strip():
                break
            messages = messages + [lm.Message("assistant", reply.text),
                                   lm.Message("user", repair_prompt(fixable, wanted))]
        if not adopted:
            line.failures += 1
            if line.failures >= self.escalate_after and line.level < len(self.levels) - 1:
                line.level, line.failures = line.level + 1, 0
        self.idle = 0 if self.ledger.used > before else self.idle + 1

    def _ask(self, messages):
        self.stats["calls"] += 1
        reply = self.model.complete(self.system, messages, seed=self.seed * 1_000_003 + self.stats["calls"])
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            self.stats[name] += getattr(reply, name)
        if reply.stop_reason == "refusal":
            self.stats["refusals"] += 1
        return reply

    def _handle(self, text, line, base, plan, attempt):
        """Process one reply. Returns (problems to send back, how many replacements to ask
        for, whether a proposal was adopted)."""
        try:
            proposals = parse_reply(text)
        except ValueError as err:
            self.stats["format_errors"] += 1
            return [("", f"the answer was not usable ({err}); return only a JSON list")], self.candidates, False
        fixable, adopted = [], False
        for p in proposals:
            why, did_adopt = self._consider(p, line, base, plan, attempt)
            if why:
                fixable.append((p.rule, why))
            adopted = adopted or did_adopt
        return fixable, len(fixable), adopted

    def _consider(self, p, line, base, plan, attempt):
        """Check, test and judge one proposal. Returns (reason to send back or None, adopted)."""
        self.stats["proposals"] += 1
        event = {"call": self.stats["calls"], "attempt": attempt, "line": line.name, "plan": plan.level,
                 "flags": list(plan.flags), "rule": p.rule, "layer": p.layer, "declared": p.level,
                 "mechanism": p.mechanism, "expected": p.expected, "status": "cut by budget"}
        self.events.append(event)
        if p.problem:
            self.stats["format_errors"] += 1
            event.update(status="format", error=p.problem)
            return p.problem, False
        try:
            tree = sg.parse(p.rule, allow_slots=self.arm == 3)
        except sg.GrammarError as err:
            self.stats["invalid"] += 1
            event.update(status="invalid", error=str(err))
            return str(err), False
        if self.arm == 3:
            tree, _ = sg.lift(tree)  # numbers written anyway are tuned as well
        edit = analyse_edit(base.template, tree)
        event.update(level=edit.level, changed=list(edit.layers), syntactic_drift=_r(edit.syntactic))
        problems = contract_problems(p, edit, plan, self.arm)
        if problems:
            self.stats["contract"] += 1
            event.update(status="contract", error="; ".join(problems))
            return "; ".join(problems), False
        key = sg.to_text(tree) if self.arm == 3 else lab.Ledger.key(tree)
        if key in self.proposed:
            self.stats["repeats"] += 1
            event["status"] = "repeat"
            return "already proposed earlier; propose something different", False
        self.proposed[key] = event
        entry = self._fit(tree)
        self.stats["tested"] += 1
        drift = behaviour_drift(self.ledger, base.row, entry.row)
        moves = actual_moves(base.row, entry.row)
        hits = [p.expected[m] == moves[m] for m in p.expected]
        self.stats["predicted"] += len(hits)
        self.stats["predicted_right"] += sum(hits)
        if drift > DRIFT_LIMIT[edit.level]:
            entry.verdict = lab.Verdict("REJECT", f"behaves too differently for a {edit.level} edit "
                                                  f"(drift {drift:.2f} > {DRIFT_LIMIT[edit.level]})")
        else:
            entry.verdict = self.verifier.verify(entry.row, line.incumbent.row, line.incumbent.q)
        verdict = entry.verdict
        self.stats[verdict.decision.lower()] += 1
        gain = verdict.q - line.incumbent.q if math.isfinite(verdict.q) and math.isfinite(line.incumbent.q) else None
        event.update(status=verdict.decision, reason=verdict.reason, q=_r(verdict.q), threshold=_r(verdict.threshold),
                     tested=sg.to_text(entry.tree), behaviour_drift=_r(drift), actual=moves,
                     trades=entry.row["stats"]["search"]["trades"])
        self.memory.add(Case(line.name, family(base.template), plan.flags, edit.level, edit.layers,
                             sg.to_text(tree), p.mechanism, verdict.decision, verdict.reason, verdict.q, gain,
                             sum(hits) / len(hits) if hits else None))
        if verdict.decision == "ADOPT":
            self._promote(line, entry, base)
            return None, True
        if verdict.decision == "INCUBATE":
            line.incubated = sorted(line.incubated + [entry], key=lambda e: -e.q)[:3]
        return None, False

    def _fit(self, tree):
        """Arms 1 and 4 backtest the rule as written; arm 3 tunes the template on train."""
        if self.arm == 3:
            template, _ = sg.lift(tree)
            _, values = lab.tune(template, self.ledger, self.k)
            bound = sg.bind(template, values)
            row = self.ledger.backtest(bound, role="trial")
            return Entry(bound, template, dict(zip(sg.slot_names(template), values)), row)
        self.ledger.structures.add(sg.to_text(sg.lift(tree)[0]))  # rules that differ only in numbers
        return Entry(tree, tree, {}, self.ledger.backtest(tree, role="candidate"))

    def _promote(self, line, entry, base):
        line.incumbent, line.adopted = entry, line.adopted + 1
        line.level, line.failures = self.levels.index(self.first_level), 0
        line.incubated = [e for e in line.incubated if e is not base]

    def _calibrate(self):
        """Arm 4: Optuna retunes the numbers of the best current rules on train with the reserve,
        starting from the model's own values; the verifier decides whether the result replaces them."""
        ranked = sorted((ln for ln in self.lines if math.isfinite(ln.incumbent.q)), key=lambda ln: -ln.incumbent.q)
        jobs = []
        for line in ranked[: self.finalists]:
            template, start = sg.lift(line.incumbent.tree)
            if start:
                jobs.append((line, template, start))
        if not jobs:
            return
        trials = (self.budget - self.ledger.used) // len(jobs) + 1  # the first trial is the model's rule
        try:
            for line, template, start in jobs:
                self.stats["calibrated"] += 1
                _, values = lab.tune(template, self.ledger, trials, start=start)
                bound = sg.bind(template, values)
                if lab.Ledger.key(bound) == line.incumbent.row["key"]:
                    continue
                entry = Entry(bound, bound, {}, self.ledger.backtest(bound, role="trial"))
                entry.verdict = self.verifier.verify(entry.row, line.incumbent.row, line.incumbent.q)
                if entry.verdict.decision == "ADOPT":
                    line.incumbent = entry
                    self.stats["calibration_adopted"] += 1
        except lab.BudgetExhausted:
            pass

    # ---- prompts and output
    def user_prompt(self, line, base, ev, plan):
        fam = family(base.template)
        out = [f"Line: {line.name} (mechanism: {fam})"]
        if self.arm == 3:
            out.append(f"Current template: {sg.to_text(base.template)}")
            if base.values:
                out.append("Tuned values: " + ", ".join(f"{k}={v:g}" for k, v in base.values.items()))
        else:
            out.append(f"Current rule: {sg.to_text(base.tree)}")
        if base is not line.incumbent:
            out.append(f"(A promising variant of this line; the line's current rule has Q {_fmt(line.incumbent.q)}.)")
        out += ["", "Evidence (train | validation):", *evidence_table(ev), "", "Diagnosis:"]
        out += [f"- {f}: {DIAGNOSIS[f][0]}" for f in plan.flags] or ["- nothing flagged; raise Q"]
        out += ["", f"Plan: {plan.level} edit at most; focus on the {' and the '.join(plan.focus)}.",
                f"Keep at least {lab.MIN_TRADES['train']} trades on train and {lab.MIN_TRADES['val']} on validation."]
        worked = self.memory.worked(plan.flags, fam)
        if worked:
            out += ["", "Worked before for these problems:"]
            out += [f"- {c.level} ({_gain(c)}): {c.mechanism} -> {c.rule}" for c in worked]
        failed = self.memory.failed(line.name)
        if failed:
            out += ["", "Rejected on this line:"] + [f"- {c.rule}: {c.reason}" for c in failed]
        tried = self.memory.tried(line.name)
        if tried:
            out += ["", "Already tested on this line (do not repeat):"] + [f"- {r}" for r in tried]
        out += ["", f"Propose {self.candidates} candidates."]
        return "\n".join(out)

    def lines_table(self):
        """One row per line: the classic it started from, its current rule, Q then and now."""
        return [{"line": ln.name, "start rule": sg.CLASSICS[ln.name], "start Q": _r(ln.start_q),
                 "rule": sg.to_text(ln.incumbent.tree), "Q": _r(ln.incumbent.q), "adopted": ln.adopted,
                 "level": self.levels[ln.level], "incubated": len(ln.incubated)} for ln in self.lines]

    def best(self):
        """The best current rule over all lines (by Q)."""
        return max((ln.incumbent for ln in self.lines), key=lambda e: e.q, default=None)

    def summary(self):
        s = self.stats
        predicted = f"{s['predicted_right']}/{s['predicted']}" if s["predicted"] else "n/a"
        out = (f"arm {self.arm}, {self.model.name}: {s['calls']} calls ({s['input_tokens']:,} input, "
               f"{s['cache_read_tokens']:,} cached, {s['output_tokens']:,} output tokens); "
               f"{s['proposals']} proposals: {s['tested']} tested, {s['invalid']} invalid, {s['contract']} broke "
               f"the contract, {s['repeats']} repeats, {s['format_errors']} format errors; "
               f"{s['adopt']} adopted, {s['incubate']} incubated, {s['reject']} rejected; "
               f"predicted moves right: {predicted}; budget {self.ledger.used}/{self.budget} "
               f"({self.search_used} in search); stopped by {self.stop_reason}")
        if self.arm == 4:
            out += f"; {s['calibrated']} rules recalibrated, {s['calibration_adopted']} improved"
        return out

    def save(self, path):
        """Per-proposal events (JSONL) and the final rule of every line (JSON next to it)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        lines = [{"line": ln.name, "rule": sg.to_text(ln.incumbent.tree), "q": _r(ln.incumbent.q),
                  "adopted": ln.adopted, "level": self.levels[ln.level]} for ln in self.lines]
        path.with_name(path.stem + ".lines.json").write_text(json.dumps(lines, indent=2), encoding="utf-8")


# ------------------------------------------------------------ analysis --
def status_counts(events):
    """How the proposals of a run ended: invalid, format, contract, repeat, ADOPT, INCUBATE, ..."""
    return Counter(e["status"] for e in events)


def first_attempt_validity(events):
    """Share of the proposals of first replies (not repair turns) that the grammar accepted."""
    first = [e for e in events if e["attempt"] == 0]
    return sum(e["status"] not in ("invalid", "format") for e in first) / len(first) if first else math.nan


def prediction_accuracy(events):
    """Per predicted move, the share of tested proposals where the model's prediction came true."""
    right, total = Counter(), Counter()
    for e in events:
        actual = e.get("actual")
        if actual:
            for move, guess in e["expected"].items():
                total[move] += 1
                right[move] += guess == actual[move]
    return {m: right[m] / total[m] for m in MOVES if total[m]}


# ------------------------------------------------------- offline model --
def replace_condition(tree, layer, rng):
    """Copy of a rule with one condition inside the entry or the exit replaced by a random one."""
    first = tree.searchSubtree(1)
    span = range(1, first.stop) if layer == "entry" else range(first.stop, len(tree))
    i = rng.choice([j for j in span if tree[j].ret is sg.Bool])
    sl = tree.searchSubtree(i)
    new = sg.gen_tree(min_=1, max_=2, type_=sg.Bool, rng=rng)
    return gp.PrimitiveTree(list(tree[: sl.start]) + new + list(tree[sl.stop:]))


class MockModel(lm.ChatModel):
    """Offline stand-in for plumbing checks: edits the current rule at a level the plan allows
    (numbers, one part, both parts or a random rule) and makes the occasional mistake.
    Its results say nothing about language models."""

    name = "mock"

    def __init__(self, arm, seed=0, p_mistake=0.15):
        self.arm, self.rng, self.p_mistake = arm, random.Random(seed), p_mistake

    def complete(self, system, messages, seed=None):
        prompt = messages[0].content
        current = re.search(r"^Current (?:rule|template): (.+)$", prompt, re.M).group(1)
        level = re.search(r"^Plan: (\w+) edit", prompt, re.M).group(1)
        wanted = re.findall(r"(?:Propose|Send) (\d+)", messages[-1].content)
        base = sg.parse(current)
        items = [self._propose(base, level) for _ in range(int(wanted[-1]) if wanted else 1)]
        text = json.dumps(items)
        return lm.Completion(text, (len(system) + sum(len(m.content) for m in messages)) // 4, len(text) // 4)

    def _propose(self, base, level):
        allowed = list(LEVELS[: LEVELS.index(level) + 1])
        if self.arm == 3 and "repair" in allowed:
            allowed.remove("repair")
        for _ in range(20):
            tree = self._edit(base, self.rng.choice(allowed))
            if tree is None or sg.limit_excess(tree):
                continue
            edit = analyse_edit(base, tree)
            if not edit.layers:
                continue
            text, layer = sg.to_text(tree), "both" if len(edit.layers) == 2 else edit.layers[0]
            if self.rng.random() < self.p_mistake:
                text = text.replace("sma(", "ema(", 1) if "sma(" in text else text + ")"
            elif self.rng.random() < self.p_mistake:
                layer = "exit" if layer == "entry" else "entry"
            return {"rule": text, "layer": layer, "level": edit.level, "mechanism": f"mock {edit.level} edit",
                    "expected": {m: self.rng.choice(("up", "down", "same")) for m in MOVES}}
        return {"rule": sg.to_text(base), "layer": "both", "level": "repair", "mechanism": "mock", "expected": {}}

    def _edit(self, base, kind):
        if kind == "repair":
            moved = lab.jittered(base, 1, 0.3, self.rng)
            return moved[0] if moved else None
        if kind == "migration":
            tree = gp.PrimitiveTree(sg.gen_tree(min_=2, max_=4, rng=self.rng))
        else:
            parts = [self.rng.choice(LAYERS)] if kind == "bridge" else list(LAYERS)
            tree = base
            for part in parts:
                tree = replace_condition(tree, part, self.rng)
        if self.arm == 3:
            return sg.canonical_slots(tree)
        return sg.bind(tree, sg.random_params(tree, self.rng))  # numbers for new placeholders


# ------------------------------------------------------------------ CLI --
def make_model(args):
    if args.provider == "mock":
        return MockModel(args.arm, seed=args.seed)
    if args.provider == "claude":
        return lm.ClaudeModel(args.model or "claude-sonnet-5-5", effort=args.effort)
    if not args.model:
        raise SystemExit("--model is required with --provider openai")
    return lm.OpenAICompatibleModel(args.model, base_url=args.base_url, temperature=args.temperature)


def main(argv=None):
    p = argparse.ArgumentParser(description="LLM strategy generator, arms 1, 3 and 4, on synthetic data")
    p.add_argument("--arm", type=int, choices=(1, 3, 4), required=True)
    p.add_argument("--provider", choices=("claude", "openai", "mock"), default="claude")
    p.add_argument("--model", help="model id; claude-sonnet-5-5 by default for --provider claude")
    p.add_argument("--effort", default="low", help="Claude effort: low, medium, high")
    p.add_argument("--base-url", help="OpenAI-compatible server, e.g. http://localhost:8080/v1")
    p.add_argument("--temperature", type=float, default=0.7, help="OpenAI-compatible servers only")
    p.add_argument("--data", choices=("planted", "noise"), default="planted")
    p.add_argument("--budget", type=int, default=2400)
    p.add_argument("--k", type=int, default=10, help="Optuna trials per template (arm 3)")
    p.add_argument("--candidates", type=int, default=4, help="proposals asked per call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", help="JSONL log of model calls (default runs/arm<A>_<provider>_<data>_seed<S>.jsonl)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true", help="answer every call from the log, no model calls")
    mode.add_argument("--resume", action="store_true", help="replay the log, then continue with the model")
    args = p.parse_args(argv)

    data = lab.synthetic(seed=7, edge=lab.EDGE if args.data == "planted" else 0.0)
    log = Path(args.log or f"runs/arm{args.arm}_{args.provider}_{args.data}_seed{args.seed}.jsonl")
    model = lm.RecordedModel(None if args.replay else make_model(args), log,
                             "replay" if args.replay else "resume" if args.resume else "record")
    gen = Generator(args.arm, model, data, args.budget, k=args.k, candidates=args.candidates, seed=args.seed).run()
    gen.save(log.with_name(log.stem + ".events.jsonl"))
    print(gen.summary())
    lab.report(f"arm {args.arm} ({model.name}, {args.data})", data, gen.ledger)


if __name__ == "__main__":
    main()

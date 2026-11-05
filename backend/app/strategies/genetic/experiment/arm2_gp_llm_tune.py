"""Arm 2: GP builds the rule structure, a language model tunes its numbers.

This is arm 5 with one part swapped. Arm 5 lets Optuna fill the free slots of each GP
structure (lab.tune); arm 2 hands that same job to a ChatModel. The GP engine, the shared
ledger, the per-structure trial budget `k` and the fitness (train Sharpe) are identical, so a
side-by-side run isolates the tuner: Optuna vs the model, on the very same structures.

The tuner is a small black-box optimiser driven by the model. For one structure it sees the
rule with named slots and their ranges, proposes sets of numbers as JSON, reads the train
Sharpe each set scored, and refines -- at most `k` backtests, exactly the budget Optuna gets.
It returns (best fitness, best values in slot_names order): the contract lab.tune and arm 5's
evaluate() already expect, so arm 2 just passes it as evolve(..., tune_fn=...).

The model only ever sees the rule's text and the ranges of its slots -- never prices, dates or
the asset. sg.params_schema(template) describes exactly the object it must fill.

Run: python arm2_gp_llm_tune.py                 (offline MockTuneModel, a plumbing check)
     python arm2_gp_llm_tune.py --claude        (Claude, needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import math
import random
import re
import zlib
from collections import Counter

import app.strategies.genetic.experiment.arm5_demo_gp_optuna as arm5
import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_models as lm
import app.strategies.genetic.strategy_gp as sg

CANDIDATES = 4           # value sets asked per model call
HISTORY = 8              # best past sets shown back to the model


# ------------------------------------------------------------- coercion --
def _coerce(slot, v):
    """A model number pulled onto the slot's grid and range, or None if it is not a number.
    Clamping (rather than rejecting) keeps a near-miss useful, like Optuna's own rounding."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    v = min(max(v, slot.lo), slot.hi)
    if slot.kind == "int":
        return int(round(v))
    if slot.step:
        v = slot.lo + round((v - slot.lo) / slot.step) * slot.step
    return round(min(max(v, slot.lo), slot.hi), 6)


def _values(item, specs):
    """A proposed {name: number} object turned into a values list in slot order, or None if a
    slot is missing or not a number."""
    out = []
    for name, slot in specs:
        if name not in item:
            return None
        v = _coerce(slot, item[name])
        if v is None:
            return None
        out.append(v)
    return out


def _parse_sets(text):
    """The first JSON list of objects in the model's reply (code fences and prose tolerated)."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\[", text):
        try:
            data, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data
    return []


# -------------------------------------------------------------- prompts --
def _system_prompt():
    return (
        "You tune the numbers of a fixed long/flat trading rule to maximise its Sharpe ratio on "
        "a training window. The structure never changes: you only choose a value for each named "
        "slot, within the range given. You never see the data -- a program backtests every set of "
        "numbers you propose and reports the train Sharpe it scored. Reuse what the scores teach "
        "you: move the numbers that seem to help, and explore when the scores are flat.\n\n"
        "Return only a JSON list, one object per proposed set, each giving every slot a number, "
        'for example [{"W1": 14, "LVL1": 30.0}]. Integers for int slots. Do not repeat a set '
        "already tried.")


def _user_prompt(template, specs, history, wanted):
    lines = [f"Rule: {sg.to_text(template)}", "", "Slots to choose:"]
    lines += [f"- {name}: {slot.kind} in [{slot.lo:g}, {slot.hi:g}], {slot.meaning}"
              for name, slot in specs]
    if history:
        lines += ["", "Tried so far (train Sharpe, best first):"]
        lines += [f"- {json.dumps(vals)} -> {fit:.3f}" for vals, fit in history]
    else:
        lines += ["", "Nothing tried yet."]
    lines += ["", f"Propose {wanted} new value sets as a JSON list."]
    return "\n".join(lines)


# ---------------------------------------------------------------- tuner --
def llm_tune(template, ledger, k, model, *, start=None, candidates=CANDIDATES, seed=0, stats=None):
    """Fill the free slots of one GP structure with a model, at most `k` backtests. Same return
    as lab.tune: (best train Sharpe, best values in slot_names order). May raise
    lab.BudgetExhausted when the shared budget runs out mid-tuning, exactly as lab.tune does."""
    stats = stats if stats is not None else Counter()
    specs = sg.slots(template)
    names = [name for name, _ in specs]
    if not names:
        return ledger.run(template, (), role="candidate"), []
    ledger.structures.add(sg.to_text(template))
    rng = random.Random(zlib.crc32(sg.to_text(template).encode()) ^ (seed & 0xFFFFFFFF))

    tried = {}                          # tuple(values) -> fitness
    best = {"fit": -math.inf, "vals": None}

    def score(values):                  # values: list in slot order; counts one trial
        key = tuple(values)
        if key not in tried:
            tried[key] = ledger.run(template, values, role="trial")   # may raise BudgetExhausted
            if tried[key] > best["fit"]:
                best["fit"], best["vals"] = tried[key], list(values)
        return tried[key]

    # a baseline trial (the model's warm start if any, else random), so a result always exists
    score(list(start) if start is not None else sg.random_params(template, rng))

    while len(tried) < k:
        wanted = min(candidates, k - len(tried))
        history = sorted(((list(v), f) for v, f in tried.items()), key=lambda vf: -vf[1])[:HISTORY]
        messages = [lm.Message("user", _user_prompt(template, specs, history, wanted))]
        stats["calls"] += 1
        reply = model.complete(_system_prompt(), messages, seed=seed * 1_000_003 + stats["calls"])
        for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            stats[name] += getattr(reply, name, 0)
        if reply.stop_reason == "refusal":
            stats["refusals"] += 1

        progressed = False
        for item in _parse_sets(reply.text):
            values = _values(item, specs)
            if values is None:
                stats["invalid"] += 1
                continue
            before = len(tried)
            score(values)
            progressed = progressed or len(tried) > before
            if len(tried) >= k:
                break
        if not progressed:               # unusable reply: spend one trial at random, never stall
            score(sg.random_params(template, rng))

    return best["fit"], best["vals"]


def make_llm_tuner(model, *, candidates=CANDIDATES, seed=0, stats=None):
    """A tune_fn(template, ledger, k) bound to one model, for arm5.evolve(..., tune_fn=...).
    Its .stats Counter accumulates calls and token usage across the whole run."""
    stats = stats if stats is not None else Counter()

    def tune_fn(template, ledger, k):
        return llm_tune(template, ledger, k, model, candidates=candidates, seed=seed, stats=stats)

    tune_fn.stats = stats
    return tune_fn


def evolve(data, model, seed=0, budget=2400, pop_size=30, k=10, n_search=lab.N_SEARCH, *,
           candidates=CANDIDATES):
    """Arm 2 on one dataset: GP structures, LLM-tuned numbers. Returns (ledger, generations,
    tuner stats) -- the ledger and generation count are arm 5's; the stats count model calls
    and tokens, as the LLM arms report them."""
    tuner = make_llm_tuner(model, candidates=candidates, seed=seed)
    ledger, generations = arm5.evolve(data, seed, budget=budget, pop_size=pop_size, k=k,
                                      n_search=n_search, tune_fn=tuner)
    return ledger, generations, tuner.stats


# ---------------------------------------------------------- offline model --
class MockTuneModel(lm.ChatModel):
    """Offline stand-in for plumbing checks: reads the slot ranges out of the prompt and returns
    random in-range value sets. Says nothing about how a real model tunes."""

    name = "mock-tune"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def complete(self, system, messages, seed=None):
        prompt = messages[-1].content
        specs = re.findall(r"^- (\w+): (int|float) in \[([^,]+), ([^\]]+)\]", prompt, re.M)
        m = re.search(r"Propose (\d+)", prompt)
        wanted = int(m.group(1)) if m else 1
        items = []
        for _ in range(wanted):
            item = {}
            for name, kind, lo, hi in specs:
                lo, hi = float(lo), float(hi)
                item[name] = self.rng.randint(int(lo), int(hi)) if kind == "int" \
                    else round(self.rng.uniform(lo, hi), 3)
            items.append(item)
        text = json.dumps(items)
        return lm.Completion(text, (len(system) + len(prompt)) // 4, len(text) // 4)


# ------------------------------------------------------------------ CLI --
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arm 2: GP structures, LLM-tuned numbers")
    parser.add_argument("--claude", action="store_true", help="use Claude instead of the offline mock")
    parser.add_argument("--model", help="model id, e.g. claude-haiku-5-5")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--pop", type=int, default=12)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name, edge in (("planted", lab.EDGE), ("noise", 0.0)):
        data = lab.synthetic(seed=7, edge=edge)
        model = lm.ClaudeModel(args.model or "claude-haiku-5-5") if args.claude else MockTuneModel(seed=args.seed)
        ledger, generations, stats = evolve(data, model, seed=args.seed, budget=args.budget,
                                            pop_size=args.pop, k=args.k)
        lab.report(f"arm 2 ({model.name}, {name})", data,
                   ledger, f"{generations} generations, {stats['calls']} model calls")

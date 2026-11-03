"""Arm 1: the search is fully the language model's.

Where arm 5 splits the work (GP builds rule structures, Optuna fills their numbers), arm 1
hands all of it to one ChatModel. Starting from each classic strategy (strategy_gp.CLASSICS)
the model proposes complete rules -- structure *and* numbers -- reads the backtest evidence
of its current rule, and proposes edits at the level the plan allows (repair, bridge,
redesign, migration). Every proposal is one backtest in the shared ledger; the same verifier
as every other arm (lab.Verifier) adopts, incubates or rejects it, and the rule handed in is
chosen from the ledger by lab.select_final.

The whole cycle lives in llm_generator.Generator (arm 1):

  1. propose     the model writes candidate rules as JSON (structure and numbers);
  2. backtest    each parsed rule is run once on the search window (lab.Ledger);
  3. tune        arm 1 has the model write and revise the numbers itself -- no Optuna;
  4. evaluate    the verifier scores each rule on validation and promotes the best per line.

This module only wires that up as an experiment arm with the same (ledger, ...) return shape
as arm0_random and arm5_demo_gp_optuna, so the comparison notebook drives every arm alike.

Run: python arm1_llm_fully.py                      (offline MockModel, a plumbing check)
     python arm1_llm_fully.py --claude             (Claude, needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import app.strategies.genetic.llm.lab as lab
import app.strategies.genetic.llm.llm_generator as llm
import app.strategies.genetic.llm.llm_models as lm


def fully_llm_search(data, model, seed=0, budget=2400, *, candidates=4, n_search=lab.N_SEARCH, **options):
    """Run the fully-LLM search (arm 1) on one dataset. `model` is any llm_models.ChatModel
    (ClaudeModel, OpenAICompatibleModel, GeminiModel, or MockModel for an offline check).
    `options` pass through to llm_generator.Generator (k, repairs, patience, max_calls, lines,
    first_level, verifier_options, ...).

    Returns (ledger, generator): the shared ledger of backtests (search window = the first
    n_search bars) and the finished Generator, whose .summary() and .lines_table() describe
    the run. The same seed and model give the same run."""
    generator = llm.Generator(1, model, data, budget, candidates=candidates, seed=seed,
                              n_search=n_search, **options)
    generator.run()
    return generator.ledger, generator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arm 1: fully-LLM strategy search on synthetic data")
    parser.add_argument("--claude", action="store_true", help="use Claude instead of the offline MockModel")
    parser.add_argument("--model", help="model id, e.g. claude-sonnet-5-5")
    parser.add_argument("--budget", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    for name, edge in (("planted", lab.EDGE), ("noise", 0.0)):
        data = lab.synthetic(seed=7, edge=edge)
        model = lm.ClaudeModel(args.model or "claude-sonnet-5-5") if args.claude else llm.MockModel(1, seed=0)
        ledger, generator = fully_llm_search(data, model, seed=args.seed, budget=args.budget)
        print(generator.summary())
        lab.report(f"arm 1 ({model.name}, {name})", data, ledger)

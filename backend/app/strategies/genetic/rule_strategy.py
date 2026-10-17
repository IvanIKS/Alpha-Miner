"""Event-driven wrapper: runs a grammar rule inside the platform's Strategy interface.

The rule is plain text, so the existing get_parameters_schema() exposes it as a string
field in the UI, and the same text is what GP, Optuna and the LLM arms produce.
Only the last `lookback` bars are kept; because every operator has finite memory,
decisions match strategy_gp.Evaluator + positions() bar for bar (see test_engine_parity).
"""
from collections import deque

import numpy as np

import app.strategies.genetic.strategy_gp as sg
# TODO: check the first two paths against the project (the third is the one RSIStrategy uses).
from app.strategies.Strategy import Strategy
from app.execution.signal import SimpleSignal
from app.strategies.rules.strategy_requirements import MustBePandasDataFrame, MustBeBinanceOHLCVData

DEFAULT_RULE = "long_flat(osc_below(rsi(close, 8), 36), osc_above(rsi(close, 8), 62))"


class RuleStrategy(Strategy):
    """Long/flat rule from the strategy grammar, e.g. the RSIStrategy defaults above."""

    def __init__(self, rule: str = DEFAULT_RULE, symbol: str = "BTC/USDT",
                 exchange: str = "binance", timeframe: str = "1m"):
        super().__init__("RuleStrategy", rule)
        self.rule = rule
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.data_requirements = [MustBePandasDataFrame(), MustBeBinanceOHLCVData()]
        self.tree = sg.parse(rule, allow_slots=False)    # every number must be set
        self.window = sg.lookback(self.tree)
        self.fields = sorted(sg.fields_used(self.tree))
        self.bars = {f: deque(maxlen=self.window) for f in self.fields}

    def on_tick(self, context):
        bar = context.market_data
        for f in self.fields:
            self.bars[f].append(float(bar[f]))
        if len(self.bars["close"]) < self.window:
            return []

        data = {f: np.fromiter(self.bars[f], dtype=float) for f in self.fields}
        entry, exit_ = sg.Evaluator(data, cache_size=0).signals(self.tree)
        enter, leave = entry[-1] == 1.0, exit_[-1] == 1.0     # unknown (NaN) never acts
        price = data["close"][-1]
        portfolio = context.portfolio

        if len(portfolio.positions) > 0:
            if leave:
                return [SimpleSignal(symbol=self.symbol, quantity=portfolio.positions[0].quantity,
                                     price=price, side="sell")]
        elif enter and not leave:                              # exit wins on the same bar
            return [SimpleSignal(symbol=self.symbol, quantity=portfolio.base_token_amount / price,
                                 price=price, side="buy")]
        return []

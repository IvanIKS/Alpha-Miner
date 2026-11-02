import json
import math
import statistics
from datetime import datetime


class BacktestResult:
    def __init__(self, trade_history, equity_curve, initial_cash, start_ts=None, end_ts=None,
                 buy_and_hold_curve=None):
        self.trade_history = trade_history
        self.equity_curve = [float(e) for e in equity_curve]
        self._initial_cash = float(initial_cash)
        self._start_ts = start_ts
        self._end_ts = end_ts
        self.buy_and_hold_curve = [float(e) for e in buy_and_hold_curve] if buy_and_hold_curve else []

    @property
    def overall_profit_percent(self):
        if not self.equity_curve or self._initial_cash == 0:
            return 0.0
        return (self.equity_curve[-1] - self._initial_cash) / self._initial_cash * 100

    @property
    def win_rate_percent(self):
        if not self.trade_history:
            return 0.0
        wins = sum(1 for pnl in self.trade_history if pnl > 0)
        return wins / len(self.trade_history) * 100

    @property
    def max_drawdown_percent(self):
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = (peak - equity) / peak * 100
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    @property
    def mean_monthly_profit_percent(self):
        if not self.equity_curve or self._initial_cash == 0:
            return 0.0
        months = 1.0
        if (
            isinstance(self._start_ts, datetime)
            and isinstance(self._end_ts, datetime)
            and self._end_ts > self._start_ts
        ):
            months = max(1.0, (self._end_ts - self._start_ts).days / 30.44)
        return self.overall_profit_percent / months

    @property
    def sharpe_ratio(self):
        if len(self.equity_curve) < 2:
            return 0.0
        returns = [
            (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
            for i in range(1, len(self.equity_curve))
            if self.equity_curve[i - 1] != 0
        ]
        if len(returns) < 2:
            return 0.0
        std = statistics.stdev(returns)
        if std == 0:
            return 0.0
        annualised = math.sqrt(252) * statistics.mean(returns) / std
        return round(annualised, 4)

    def _capm(self):
        """(alpha_per_step, beta) via OLS vs buy-and-hold (Rf=0). Returns (0.0, 0.0) when unavailable."""
        n = min(len(self.equity_curve), len(self.buy_and_hold_curve))
        if n < 3:
            return 0.0, 0.0
        strat_ret = [
            (self.equity_curve[i] - self.equity_curve[i - 1]) / self.equity_curve[i - 1]
            for i in range(1, n) if self.equity_curve[i - 1] != 0
        ]
        bah_ret = [
            (self.buy_and_hold_curve[i] - self.buy_and_hold_curve[i - 1]) / self.buy_and_hold_curve[i - 1]
            for i in range(1, n) if self.buy_and_hold_curve[i - 1] != 0
        ]
        k = min(len(strat_ret), len(bah_ret))
        if k < 2:
            return 0.0, 0.0
        strat_ret, bah_ret = strat_ret[:k], bah_ret[:k]
        mean_s = statistics.mean(strat_ret)
        mean_b = statistics.mean(bah_ret)
        var_b = statistics.variance(bah_ret)
        if var_b == 0:
            return 0.0, 0.0
        cov = sum((s - mean_s) * (b - mean_b) for s, b in zip(strat_ret, bah_ret)) / (k - 1)
        beta = cov / var_b
        return mean_s - beta * mean_b, beta

    @property
    def alpha(self):
        """Jensen's alpha vs buy-and-hold (Rf=0), annualised × 252."""
        alpha_step, _ = self._capm()
        return round(alpha_step * 252, 4)

    @property
    def beta(self):
        """Market beta vs buy-and-hold: Cov(Rs, Rb) / Var(Rb)."""
        _, b = self._capm()
        return round(b, 4)

    def to_json(self) -> str:
        return json.dumps({
            "overall_profit_percent": self.overall_profit_percent,
            "mean_monthly_profit_percent": self.mean_monthly_profit_percent,
            "max_drawdown_percent": self.max_drawdown_percent,
            "win_rate_percent": self.win_rate_percent,
            "sharpe_ratio": self.sharpe_ratio,
            "alpha": self.alpha,
            "beta": self.beta,
        })


class BacktestEngine:
    def run(self, strategy, data_provider, money_amount):
        from app.execution.execution_engine import BacktestExecutionEngine
        from app.portfolio.PortfolioProvider import BacktestPortfolioProvider
        from app.execution.bot import Bot

        portfolio_provider = BacktestPortfolioProvider(quantity=money_amount)
        execution_engine = BacktestExecutionEngine()

        bot = Bot(
            name="backtest_bot",
            strategy=strategy,
            execution_engine=execution_engine,
            portfolio_provider=portfolio_provider,
        )

        start_ts = None
        end_ts = None

        bot.on_start()

        while True:
            market_data = data_provider.get_market_data(strategy)
            if market_data is None:
                break

            ts = market_data.get("timestamp") if isinstance(market_data, dict) else None
            if start_ts is None:
                start_ts = ts
            end_ts = ts

            bot.on_tick(market_data)

            prices = data_provider.get_asset_prices(strategy)
            cash = portfolio_provider.get_portfolio(strategy).base_token_amount
            execution_engine.populate_equity_curve(prices, float(cash))

        bot.on_stop()

        return BacktestResult(
            trade_history=execution_engine.trade_history,
            equity_curve=execution_engine.equity_curve,
            initial_cash=money_amount,
            start_ts=start_ts,
            end_ts=end_ts,
        )

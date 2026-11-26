"""Commission and slippage for the lab tasks.

The lab search runs the fast vectorised backtest (sg.net_returns), which charges a single
per-side cost in basis points, while the finalist runs the full BacktestEngine, which charges
the Commission and SimpleSlippage classes order by order. Both are built here from the same run
config so the search and the finalist backtest agree on costs.
"""
from __future__ import annotations

from app.execution.commission import Commission
from app.execution.order import Order
from app.execution.slippage import SimpleSlippage


def build_cost_models(cfg):
    """(Commission, SimpleSlippage) from the run config -- the instances the finalist
    BacktestEngine runs and the source of the search's per-side cost_bps.

    buy_commission/sell_commission are percents per side; slippage_percent is a percent of price
    per side (1.0 reproduces SimpleSlippage's historical 1% default)."""
    commission = Commission(cfg.get("buy_commission", 0.0), cfg.get("sell_commission", 0.0))
    slippage = SimpleSlippage(cfg.get("slippage_percent", 1.0) / 100.0)
    return commission, slippage


def effective_cost_bps(commission, slippage) -> float:
    """Per-side cost the vectorised engine should charge, in basis points.

    sg.net_returns charges one cost as a fraction of the traded notional on every position
    change. Evaluating the commission and slippage on a unit order (price=1, quantity=1) makes
    their charge already a fraction of notional; we average the buy and sell sides and scale to
    basis points so the fast search charges what the Commission/SimpleSlippage classes model."""
    def side(s):
        order = Order(symbol="_", quantity=1.0, price=1.0, side=s)
        return float(commission.calculate(order) + slippage.calculate(order))

    return (side("buy") + side("sell")) / 2.0 * 1e4

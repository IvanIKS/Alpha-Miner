import logging
from datetime import datetime
from decimal import Decimal

from app.execution import slippage
from app.execution.slippage import Slippage, SimpleSlippage
from app.execution.commission import Commission


class ExecutionEngine:
    def __init__(self):
        pass
    
    def execute(self, order):
        raise NotImplementedError("Must be implemented by subclasses.")
    

class ExecutionResult:
    def __init__(self, success, order=None, error_message=None):
        self.success = success
        self.order = order
        self.error_message = error_message


class BacktestExecutionEngine(ExecutionEngine):
    def __init__(self, commission=None, slippage=None, log_process=False):
        super().__init__()
        self.trade_history = []  # List of closed trade PnLs: [50.5, -20.0, ...]
        self.open_positions = {}  # Tracks {symbol: {'price': Decimal, 'quantity': Decimal, 'side': str}}
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)

        if (log_process):
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger.info("Backtest execution engine initialized with trade tracking.")
        else:
            logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

        if commission is None:
            self.commission = Commission(0, 0)
        else:
            self.commission = commission
            self.logger.info(
                f"Backtest execution engine initialized with commission rate: {commission}"
            )
        if slippage is None:
            self.slippage_ = SimpleSlippage()
        else:
            self.slippage_ = slippage
            self.logger.info(f"Backtest execution engine initialized with slippage: {slippage}")

    def execute(self, order):
        """Simulate execution and record trade performance."""
        symbol = order.symbol

        if symbol in self.open_positions:
            prev_order = self.open_positions[symbol]

            if prev_order['side'] != order.side:
                # Calculate PnL: (Exit Price - Entry Price) * Quantity
                # Note: This logic assumes 'Buy Low, Sell High'
                entry_price = Decimal(prev_order['price']) + self.commission.calculate(order) + self.slippage_.calculate(order)
                exit_price = Decimal(order.price) - self.commission.calculate(order) - self.slippage_.calculate(order)

                if prev_order['side'] == 'buy':
                    pnl = Decimal((exit_price - entry_price) * Decimal(order.quantity))
                else:  # shorting logic
                    pnl = Decimal((entry_price - exit_price) * Decimal(order.quantity))

                self.trade_history.append(pnl)
                logging.info(f"TRADE CLOSED: {symbol} | PnL: {pnl:.2f}")

                # Remove from open positions
                del self.open_positions[symbol]
            else:
                # Logic for adding to an existing position could go here
                pass
        else:
            # 2. Open a new position
            self.open_positions[symbol] = {
                'price': Decimal(order.price),
                'quantity': Decimal(order.quantity),
                'side': order.side
            }
            logging.info(f"TRADE OPENED: {order.side.upper()} {symbol} at {order.price}")

        return ExecutionResult(success=True, order=order)

    def populate_equity_curve(self, prices: dict[str, Decimal], cash: float) -> None:
        """Record total equity = cash + realized PnL + unrealized PnL on open positions."""
        holdings_value = Decimal(0.0)
        for symbol, pos in self.open_positions.items():
            if symbol not in prices:
                continue
            if pos['side'] == 'buy':
                holdings_value += prices[symbol] * pos['quantity']
            else:
                holdings_value += (Decimal(2) * pos['price'] - prices[symbol] )* pos['quantity']

        equity = Decimal(cash) + holdings_value
        self.equity_curve.append(equity)

class BinanceExecutionEngine(ExecutionEngine):
    def __init__(self, api_key, api_secret):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        print("Binance execution engine initialized with API key")

    def execute(self, order):
        """Execute the given order on Binance."""
        # Here we would implement the logic to send the order to Binance's API and handle the response.
        raise NotImplementedError("Binance execution logic not implemented yet.")


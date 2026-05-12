import json
from datetime import datetime

from celery import shared_task
from sqlalchemy import select

from app.data.database import BacktestResult
from app.strategies.registry import STRATEGY_CLASSES

from app.backtest.BacktestEngine import BacktestEngine
from app.data.data_provider import MarketDataProvider, GapsFoundError, OHLCVDataProvider
from app.strategies.Strategy import Strategy
from app.utils.timeframe import get_candles_left
from app.data.database.models import Strategy as StrategyModel, BacktestResult
from app.data.database.session import db_session


MAX_RECURSION_DEPTH = 4

def approximate_wait_time(start_date, end_date, timeframe):
    timeframe = get_candles_left(start_date, end_date, timeframe)
    return timeframe / 1000

def load_historical_data(strategy : Strategy, start_date : datetime, end_date : datetime) -> MarketDataProvider:
    """Currently assumes all these things are in strategy"""
    schema = strategy.get_parameters_schema()
    schema_keys = {item['key'] for item in schema}

    required = {'symbol', 'exchange', 'timeframe'}
    has_required = required.issubset(schema_keys)

    if has_required:
        exchange = getattr(strategy, 'exchange')
        symbol = getattr(strategy, 'symbol')
        timeframe = getattr(strategy, 'timeframe')

        return OHLCVDataProvider(strategy, start_date, end_date, exchange, symbol, timeframe)
    else:
        missing = required - schema_keys
        raise Exception(f"Strategy is missing required parameters: {missing}")


@shared_task(bind=True, name="backtest_celery")
def backtest_task(self, strategy_name : str, strategy_dict : dict, start_date : datetime, end_date : datetime, money_amount : float, backtest_id : str):
    try:
        print("Starting backtest ...")
        strategy_class = STRATEGY_CLASSES[strategy_name]
        strategy = strategy_class(**strategy_dict)
        data_provider = load_historical_data(strategy, start_date, end_date)
        engine = BacktestEngine()
        result = engine.run(strategy, data_provider, money_amount)

        results_json = json.dumps({
            'overall_profit_percent': result.overall_profit_percent,
            'mean_monthly_profit_percent': result.mean_monthly_profit_percent,
            'max_drawdown_percent': result.max_drawdown_percent,
            'win_rate_percent': result.win_rate_percent,
        })
        with db_session() as db:
            backtest = db.execute(
                select(BacktestResult).where(BacktestResult.id == backtest_id)
            ).scalar_one_or_none()
            if backtest:
                backtest.results = results_json
                db.commit()

                print("Backtest results saved successfully.")
                return {"status": "success", "backtest_id": backtest_id}
            else:
                return {"status": "error", "message": f"Backtest ID not found{backtest_id}"}

    except GapsFoundError as e:
        wait_time = 0
        for gap in e.gaps:
            wait_time += approximate_wait_time(gap[0], gap[1], e.timeframe)
        print(f"Gaps found. Retrying same Task ID ({self.request.id}) in {wait_time}s.")

        raise self.retry(
            exc=e,
            countdown=wait_time,
            backtest_id=backtest_id
        )
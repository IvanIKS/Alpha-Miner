from datetime import datetime, timezone

from app.background_tasks import retry
from app.background_tasks.celery_app import celery
from app.data.service.backfill_service import BackfillService


@celery.task(bind=True, name="backfill_ohlcv", max_retries=5)
def backfill_ohlcv_task(self, exchange: str, symbol: str, start_timestamp: datetime, end_timestamp: datetime, timeframe: str):
    self.update_state(state="RUNNING", meta={"exchange": exchange, "symbol": symbol})

    service = BackfillService(exchange)
    try:
        service.run(start_timestamp, end_timestamp, symbol, timeframe)
    except retry.connection_errors() as exc:
        # Unstable connection to the exchange: wait and retry. Re-running re-finds the
        # remaining gaps and the upsert skips candles already written, so it is idempotent.
        countdown = retry.retry_countdown(self.request.retries)
        print(f"Backfill hit a connection error ({exc!r}); retrying in {countdown}s "
              f"(attempt {self.request.retries + 1}/5).")
        raise self.retry(exc=exc, countdown=countdown)

    return {
        "exchange": exchange,
        "symbol": symbol,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "status": "completed",
    }
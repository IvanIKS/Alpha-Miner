from datetime import datetime

from app.background_tasks.celery_app import celery
from app.data.service.prediction_market_service import PredictionMarketService


@celery.task(bind=True, name="sync_prediction_markets")
def sync_prediction_markets_task(self, platform: str, limit: int | None = None):
    self.update_state(state="RUNNING", meta={"platform": platform})

    service = PredictionMarketService(platform)
    count = service.sync_markets(limit=limit)

    return {"platform": platform, "markets_synced": count, "status": "completed"}


@celery.task(bind=True, name="backfill_prediction_prices")
def backfill_prediction_prices_task(
    self,
    platform: str,
    market_id: str,
    token_id: str,
    start_timestamp: datetime,
    end_timestamp: datetime,
    timeframe: str,
):
    self.update_state(state="RUNNING", meta={"platform": platform, "market_id": market_id})

    service = PredictionMarketService(platform)
    count = service.backfill_prices(
        market_id=market_id,
        token_id=token_id,
        start_date=start_timestamp,
        end_date=end_timestamp,
        timeframe=timeframe,
    )

    return {
        "platform": platform,
        "market_id": market_id,
        "token_id": token_id,
        "points_written": count,
        "status": "completed",
    }

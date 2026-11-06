
import os
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery = Celery(
    "tasks",
    broker=broker_url,
    backend=result_backend,
    include=[
        "app.background_tasks.tasks.backfill",
        "app.background_tasks.tasks.backtest",
        "app.background_tasks.tasks.prediction",
    ]
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
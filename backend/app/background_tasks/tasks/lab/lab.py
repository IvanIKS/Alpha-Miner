from celery import shared_task


@shared_task(bind=True, name="compare_strategy_mining")
def compare_strategy_mining(self):
    pass
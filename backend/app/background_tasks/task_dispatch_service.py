
import enum


class TaskExecutionProviderType(enum.Enum):
    CELERY_LOCAL = "celery-local"

class TaskExecutionProvider:
    def __init__(self, provider_type):
        self.provider_type = provider_type

    def submit_task(self, task_name, *args, **kwargs):
        raise NotImplementedError("To be implemented in subclass.")

class CeleryLocalTaskExecutionProvider(TaskExecutionProvider):
    def __init__(self):
        super().__init__(TaskExecutionProviderType.CELERY_LOCAL)

    def submit_task(self, task_name, *args, **kwargs):
        pass


class TaskDispatchService:
    def __init__(self, providers : list[TaskExecutionProvider]):
        if providers is None:
            providers = [TaskExecutionProviderType.CELERY_LOCAL]


    def submit_task(self, task_name, provider, *args, **kwargs):
        self.providers[provider].submit_task(task_name, *args, **kwargs)

task_registry =  {
    TaskExecutionProviderType.CELERY_LOCAL : [
        "app.background_tasks.tasks.backfill",
        "app.background_tasks.tasks.backtest",
        "app.background_tasks.tasks.prediction",
        "app.background_tasks.tasks.lab.lab",
        "app.background_tasks.tasks.lab.arms",
        "app.background_tasks.tasks.lab.finalize"]
}


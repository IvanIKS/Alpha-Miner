import importlib
import inspect
import pkgutil

from app.strategies.Strategy import Strategy as BaseStrategy


def _build_registry() -> dict[str, type[BaseStrategy]]:
    registry: dict[str, type[BaseStrategy]] = {}
    package_name = "app.strategies.implementation"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg:
            continue
        module = importlib.import_module(f"{package_name}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is BaseStrategy or not issubclass(cls, BaseStrategy):
                continue
            try:
                instance = cls()
                registry[instance.name] = cls
            except Exception:
                pass
    return registry


STRATEGY_CLASSES: dict[str, type[BaseStrategy]] = _build_registry()

from pathlib import Path


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_MLFLOW_URL = "http://127.0.0.1:5000"

REQUEST_TIMEOUT = 120

REPORTS_ROOT = Path(
    "reports/model_backtesting"
)

MONITORING_ROOT = Path(
    "reports/monitoring"
)

AVAILABLE_SYMBOLS = [
    "GC",
    "SI",
    "NQ",
    "ES",
]

AVAILABLE_TIMEFRAMES = [
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
]

AVAILABLE_MODELS = [
    "random_forest",
    "xgboost",
    "lstm",
]

SIGNAL_OPTIONS = [
    "BUY",
    "SELL",
    "NO_TRADE",
]
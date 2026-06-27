import os
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
THENEWSAPI_KEY = os.getenv("THENEWSAPI_KEY")

START_DATE = "2020-01-01"

MARKET_SYMBOLS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "nasdaq": "NQ=F",
    "sp500": "ES=F",
}

FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "nfp": "PAYEMS",
    "fed_rate": "FEDFUNDS",
    "unemployment_rate": "UNRATE",
}

MARKET_INTERVALS = {
    "1d": {
        "interval": "1d",
        "period": "5y"
    },
    "4h": {
        "interval": "4h",
        "period": "2y"
    },
    "1h": {
        "interval": "1h",
        "period": "2y"
    },
    "15m": {
        "interval": "15m",
        "period": "60d"
    },
    "5m": {
        "interval": "5m",
        "period": "30d"
    },
    "1m": {
        "interval": "1m",
        "period": "7d"
    },
}

FMP_API_KEY = os.getenv("FMP_API_KEY")

ECONOMIC_EVENTS_KEYWORDS = [
    "CPI",
    "Nonfarm Payrolls",
    "FOMC",
    "Interest Rate Decision",
]
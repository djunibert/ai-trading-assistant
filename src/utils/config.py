import os
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# API KEYS
# =====================================================

FRED_API_KEY = os.getenv("FRED_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
THENEWSAPI_KEY = os.getenv("THENEWSAPI_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")


# =====================================================
# GENERAL SETTINGS
# =====================================================

START_DATE = "2020-01-01"
TIMEZONE = "America/New_York"
CSV_ENCODING = "utf-8-sig"


# =====================================================
# YAHOO FINANCE
# =====================================================

YAHOO_SYMBOLS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "nasdaq": "NQ=F",
    "sp500": "ES=F",
}

YAHOO_INTERVALS = {
    "1d": {
        "interval": "1d",
        "period": "5y",
    },
    "4h": {
        "interval": "4h",
        "period": "2y",
    },
    "1h": {
        "interval": "1h",
        "period": "2y",
    },
    "15m": {
        "interval": "15m",
        "period": "60d",
    },
    "5m": {
        "interval": "5m",
        "period": "30d",
    },
    "1m": {
        "interval": "1m",
        "period": "7d",
    },
}


# =====================================================
# TWELVE DATA
# =====================================================

TWELVEDATA_SYMBOLS = {
    "gold": "XAU/USD",
    "silver": "XAG/USD",
    "nasdaq": "NAS100",
    "sp500": "SPX500",
}

TWELVEDATA_INTERVALS = {
    "1d": "1day",
    "4h": "4h",
    "1h": "1h",
    "15m": "15min",
    "5m": "5min",
    "1m": "1min",
}

TWELVEDATA_OUTPUTSIZE = 5000


# =====================================================
# FRED
# =====================================================

FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "nfp": "PAYEMS",
    "fed_rate": "FEDFUNDS",
    "unemployment_rate": "UNRATE",
}


# =====================================================
# ECONOMIC EVENTS
# =====================================================

ECONOMIC_EVENTS_KEYWORDS = [
    "CPI",
    "Core CPI",
    "Nonfarm Payrolls",
    "NFP",
    "FOMC",
    "Interest Rate Decision",
]


# =====================================================
# NEWS QUERIES
# =====================================================

NEWS_QUERY = (
    "gold OR silver OR XAU OR XAG OR "
    "Federal Reserve OR Fed OR FOMC OR "
    "inflation OR CPI OR NFP OR "
    "interest rates OR US dollar OR DXY"
)


# =====================================================
# GLOBAL DATA SOURCES CONFIG
# =====================================================

DATA_SOURCES = {
    "yahoo": {
        "enabled": True,
        "symbols": YAHOO_SYMBOLS,
        "intervals": YAHOO_INTERVALS,
    },
    "twelvedata": {
        "enabled": True,
        "symbols": TWELVEDATA_SYMBOLS,
        "intervals": TWELVEDATA_INTERVALS,
        "outputsize": TWELVEDATA_OUTPUTSIZE,
    },
    "fred": {
        "enabled": True,
        "series": FRED_SERIES,
    },
    "newsapi": {
        "enabled": False,
        "query": NEWS_QUERY,
    },
    "thenewsapi": {
        "enabled": False,
        "query": NEWS_QUERY,
    },
    "fmp": {
        "enabled": False,
        "keywords": ECONOMIC_EVENTS_KEYWORDS,
    },
}

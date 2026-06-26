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
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_MARKET_DIR = RAW_DIR / "market"
RAW_MACRO_DIR = RAW_DIR / "macro"
RAW_NEWS_DIR = RAW_DIR / "news"

PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
FINAL_DIR = DATA_DIR / "final"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
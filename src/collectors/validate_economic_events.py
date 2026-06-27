"""
Validation du fichier economic_events.csv.
"""

import pandas as pd

from src.utils.paths import RAW_NEWS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_economic_events() -> None:
    file_path = RAW_NEWS_DIR / "economic_events.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    df = pd.read_csv(file_path)

    required_columns = [
        "datetime",
        "country",
        "currency",
        "event",
        "impact",
        "actual",
        "forecast",
        "previous",
        "surprise",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    df["datetime"] = pd.to_datetime(df["datetime"])

    logger.info("economic_events.csv est valide.")
    logger.info(f"Nombre d'événements : {len(df)}")


if __name__ == "__main__":
    validate_economic_events()
"""
Nettoyage des données de marché multi-timeframe.

Entrée :
    data/raw/market/<timeframe>/<asset>.csv

Sortie :
    data/processed/market/<timeframe>/<asset>.csv
"""

import pandas as pd

from src.utils.paths import RAW_MARKET_DIR, PROCESSED_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_market_file(file_path, timeframe):
    """
    Nettoie un fichier market individuel.
    """

    asset_name = file_path.stem

    logger.info(f"Nettoyage : {asset_name} - {timeframe}")

    df = pd.read_csv(file_path)

    df.columns = [
        str(col).lower().replace(" ", "_")
        for col in df.columns
    ]

    if "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["asset"] = asset_name
    df["timeframe"] = timeframe

    df = df.dropna(subset=["datetime", "close"])
    df = df.drop_duplicates(subset=["datetime"])
    df = df.sort_values("datetime")

    return df


def clean_market_data():
    """
    Nettoie tous les fichiers de marché tout en conservant
    la structure par timeframe et par actif.
    """

    output_base_dir = PROCESSED_DIR / "market"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for timeframe_dir in RAW_MARKET_DIR.iterdir():

        if not timeframe_dir.is_dir():
            continue

        timeframe = timeframe_dir.name
        output_timeframe_dir = output_base_dir / timeframe
        output_timeframe_dir.mkdir(parents=True, exist_ok=True)

        for file_path in timeframe_dir.glob("*.csv"):

            df = clean_market_file(file_path, timeframe)

            output_file = output_timeframe_dir / file_path.name

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            logger.info(f"Fichier créé : {output_file}")


if __name__ == "__main__":
    clean_market_data()
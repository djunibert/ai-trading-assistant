"""
Module : merge_datasets.py

Description
-----------
Fusionne les données de marché et les données macroéconomiques
pour créer un jeu de données unique destiné au Machine Learning.

Auteur : Junior Hébert
Projet : AI Trading System
"""

import pandas as pd

from src.utils.paths import PROCESSED_DIR, FINAL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def merge_datasets():
    """
    Fusionne les données de marché et les données macro.
    """

    logger.info("Fusion des jeux de données...")

    market_file = PROCESSED_DIR / "market_clean.csv"
    macro_file = PROCESSED_DIR / "macro_clean.csv"

    market = pd.read_csv(
        market_file,
        parse_dates=["datetime"]
    )

    macro = pd.read_csv(
        macro_file,
        parse_dates=["datetime"]
    )

    # Tri chronologique
    market = market.sort_values("datetime")
    macro = macro.sort_values("datetime")

    # Fusion temporelle
    dataset = pd.merge_asof(
        market,
        macro,
        on="datetime",
        direction="backward"
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = FINAL_DIR / "dataset_v1.csv"

    dataset.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Dataset créé : {output_file}")

    return dataset


if __name__ == "__main__":
    merge_datasets()

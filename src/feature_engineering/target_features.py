"""
Module : target_features.py

Description
-----------
Création des variables cibles (Target)
pour les modèles de Machine Learning.

Version V1 :
    1 = prochaine clôture supérieure
    0 = prochaine clôture inférieure ou égale
"""

import pandas as pd

from src.utils.paths import FINAL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_binary_target() -> None:

    dataset_file = FINAL_DIR / "dataset_ml_v1.csv"

    logger.info("Lecture du dataset final...")

    df = pd.read_csv(
        dataset_file,
        parse_dates=["datetime"]
    )

    logger.info("Création de la Target...")

    df = df.sort_values(
        ["asset", "timeframe", "datetime"]
    )

    df["future_close"] = (
        df.groupby(
            ["asset", "timeframe"]
        )["close"]
        .shift(-1)
    )

    df["target"] = (
        df["future_close"] > df["close"]
    ).astype(int)

    df = df.drop(columns=["future_close"])

    df = df.dropna()

    output_file = FINAL_DIR / "dataset_ml_v1.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info("Target créée avec succès.")
    logger.info(f"Fichier sauvegardé : {output_file}")


if __name__ == "__main__":
    build_binary_target()
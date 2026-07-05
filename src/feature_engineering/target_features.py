"""
Module : target_features.py

Création d'une target trading à 3 classes.

0 = SELL
1 = NO_TRADE
2 = BUY
"""

import pandas as pd

from src.utils.paths import FINAL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_trading_target(
    horizon: int = 3,
    threshold: float = 0.002
) -> None:
    """
    Crée une target basée sur le rendement futur.

    Args:
        horizon:
            Nombre de bougies dans le futur.

        threshold:
            Seuil minimum de variation.
            Exemple 0.002 = 0.2 %

    Logique :
        future_return > threshold  -> BUY
        future_return < -threshold -> SELL
        sinon                      -> NO_TRADE
    """

    dataset_file = FINAL_DIR / "dataset_ml_v1.csv"

    logger.info(f"Lecture du dataset : {dataset_file}")

    df = pd.read_csv(dataset_file, parse_dates=["datetime"])

    df = df.sort_values(["asset", "timeframe", "datetime"])

    df["future_close"] = (
        df.groupby(["asset", "timeframe"])["close"]
        .shift(-horizon)
    )

    df["future_return"] = (
        df["future_close"] - df["close"]
    ) / df["close"]

    df["target"] = 1  # NO_TRADE par défaut

    df.loc[df["future_return"] > threshold, "target"] = 2  # BUY
    df.loc[df["future_return"] < -threshold, "target"] = 0  # SELL

    df = df.dropna(subset=["future_close", "future_return"])

    df = df.drop(columns=["future_close"])

    df.to_csv(
        dataset_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info("Target 3 classes créée avec succès.")
    logger.info("0 = SELL | 1 = NO_TRADE | 2 = BUY")
    logger.info(df["target"].value_counts().to_string())


if __name__ == "__main__":
    build_trading_target()

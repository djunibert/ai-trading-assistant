"""
Création des indicateurs techniques multi-timeframe.

Entrée :
    data/processed/market/<timeframe>/<asset>.csv

Sortie :
    data/features/market/<timeframe>/<asset>.csv
"""

import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

from src.utils.paths import PROCESSED_DIR, FEATURES_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les indicateurs techniques à un DataFrame market.
    """

    df = df.sort_values("datetime").copy()

    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()

    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )

    df["atr_14"] = atr.average_true_range()

    df["return_1"] = df["close"].pct_change()
    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    df = df.dropna()

    return df


def build_market_features() -> None:
    """
    Crée les features pour chaque actif et chaque timeframe.
    """

    input_base_dir = PROCESSED_DIR / "market"
    output_base_dir = FEATURES_DIR / "market"

    output_base_dir.mkdir(parents=True, exist_ok=True)

    for timeframe_dir in input_base_dir.iterdir():

        if not timeframe_dir.is_dir():
            continue

        timeframe = timeframe_dir.name
        output_timeframe_dir = output_base_dir / timeframe
        output_timeframe_dir.mkdir(parents=True, exist_ok=True)

        for file_path in timeframe_dir.glob("*.csv"):

            asset_name = file_path.stem

            logger.info(f"Création features : {asset_name} - {timeframe}")

            df = pd.read_csv(file_path, parse_dates=["datetime"])

            df_features = add_technical_indicators(df)

            output_file = output_timeframe_dir / file_path.name

            df_features.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            logger.info(f"Fichier créé : {output_file}")


if __name__ == "__main__":
    build_market_features()
"""
Module : yahoo_collector.py

Collecte multi-timeframe depuis Yahoo Finance.
"""

import pandas as pd
import yfinance as yf

from src.utils.config import MARKET_SYMBOLS, MARKET_INTERVALS
from src.utils.paths import RAW_MARKET_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def clean_yahoo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"])

    return df


def collect_market_data() -> None:
    logger.info("Début de la collecte Yahoo Finance multi-timeframe...")

    for timeframe, config in MARKET_INTERVALS.items():

        timeframe_dir = RAW_MARKET_DIR / timeframe
        timeframe_dir.mkdir(parents=True, exist_ok=True)

        for asset_name, symbol in MARKET_SYMBOLS.items():

            logger.info(
                f"Collecte de {asset_name} ({symbol}) - {timeframe}"
            )

            df = yf.download(
                tickers=symbol,
                period=config["period"],
                interval=config["interval"],
                auto_adjust=False,
                progress=False,
            )

            if df.empty:
                logger.warning(
                    f"Aucune donnée pour {asset_name} - {timeframe}"
                )
                continue

            df = clean_yahoo_dataframe(df)

            output_file = timeframe_dir / f"{asset_name}.csv"

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            logger.info(f"Fichier créé : {output_file}")

    logger.info("Collecte Yahoo Finance multi-timeframe terminée.")


if __name__ == "__main__":
    collect_market_data()
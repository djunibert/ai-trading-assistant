"""
Module : twelvedata_collector.py

Collecteur Twelve Data basé sur BaseMarketCollector.
"""

import pandas as pd
from twelvedata import TDClient

from src.collectors.base_market_collector import BaseMarketCollector
from src.utils.config import (
    TWELVEDATA_API_KEY,
    TWELVEDATA_SYMBOLS,
    TWELVEDATA_INTERVALS,
    TWELVEDATA_OUTPUTSIZE,
    TIMEZONE,
)
from src.utils.paths import RAW_DIR


class TwelveDataMarketCollector(BaseMarketCollector):
    """
    Collecteur de données de marché depuis Twelve Data.
    """

    def __init__(self):
        super().__init__(source_name="twelvedata")

        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY manquant dans le fichier .env")

        self.client = TDClient(apikey=TWELVEDATA_API_KEY)

    def download_symbol(
        self,
        symbol: str,
        interval: str,
        period: str | None = None,
    ) -> pd.DataFrame:
        ts = self.client.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=TWELVEDATA_OUTPUTSIZE,
            timezone=TIMEZONE,
        )

        return ts.as_pandas()

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.reset_index()

        df.columns = [
            str(col).lower().replace(" ", "_")
            for col in df.columns
        ]

        if "datetime" not in df.columns:
            raise ValueError("Colonne datetime introuvable dans les données Twelve Data.")

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(None)

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["datetime", "close"])
        df = df.drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")

        return df

    def save_dataframe(self, df: pd.DataFrame, output_file) -> None:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

    def collect(self) -> None:
        self.logger.info("Début collecte Twelve Data")

        output_base_dir = RAW_DIR / "market_twelvedata"
        output_base_dir.mkdir(parents=True, exist_ok=True)

        for timeframe, interval in TWELVEDATA_INTERVALS.items():
            timeframe_dir = output_base_dir / timeframe
            timeframe_dir.mkdir(parents=True, exist_ok=True)

            for asset_name, symbol in TWELVEDATA_SYMBOLS.items():
                self.logger.info(
                    f"Collecte : {asset_name} ({symbol}) [{timeframe}]"
                )

                try:
                    df = self.download_symbol(
                        symbol=symbol,
                        interval=interval,
                    )

                    if df.empty:
                        self.logger.warning(
                            f"Aucune donnée : {asset_name} [{timeframe}]"
                        )
                        continue

                    df = self.clean_dataframe(df)

                    df["asset"] = asset_name
                    df["symbol"] = symbol
                    df["timeframe"] = timeframe
                    df["source"] = self.source_name

                    output_file = timeframe_dir / f"{asset_name}.csv"

                    self.save_dataframe(df, output_file)

                    self.logger.info(f"Fichier créé : {output_file}")

                except Exception as error:
                    self.logger.error(
                        f"Erreur Twelve Data ({asset_name}, {timeframe}) : {error}"
                    )

        self.logger.info("Collecte Twelve Data terminée.")


def collect_twelvedata_market_data() -> None:
    collector = TwelveDataMarketCollector()
    collector.collect()


if __name__ == "__main__":
    collect_twelvedata_market_data()
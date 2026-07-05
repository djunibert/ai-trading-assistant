"""
Module : yahoo_collector.py

Collecteur Yahoo Finance basé sur BaseMarketCollector.
"""

import pandas as pd
import yfinance as yf

from src.collectors.base_market_collector import BaseMarketCollector
from src.utils.config import YAHOO_SYMBOLS, YAHOO_INTERVALS
from src.utils.paths import RAW_MARKET_DIR


class YahooMarketCollector(BaseMarketCollector):
    """
    Collecteur de données de marché depuis Yahoo Finance.
    """

    def __init__(self):
        super().__init__(source_name="yahoo")

    def download_symbol(
        self,
        symbol: str,
        interval: str,
        period: str | None = None,
    ) -> pd.DataFrame:
        return yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            auto_adjust=False,
            progress=False,
        )

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        df.columns = [
            str(col).lower().replace(" ", "_")
            for col in df.columns
        ]

        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})

        if "datetime" not in df.columns:
            raise ValueError("Colonne datetime introuvable dans les données Yahoo.")

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
            "adj_close",
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
        self.logger.info("Début collecte Yahoo Finance")

        for timeframe, config in YAHOO_INTERVALS.items():
            interval = config["interval"]
            period = config["period"]

            timeframe_dir = RAW_MARKET_DIR / timeframe
            timeframe_dir.mkdir(parents=True, exist_ok=True)

            for asset_name, symbol in YAHOO_SYMBOLS.items():
                self.logger.info(
                    f"Collecte : {asset_name} ({symbol}) [{timeframe}]"
                )

                try:
                    df = self.download_symbol(
                        symbol=symbol,
                        interval=interval,
                        period=period,
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
                        f"Erreur Yahoo ({asset_name}, {timeframe}) : {error}"
                    )

        self.logger.info("Collecte Yahoo Finance terminée.")


def collect_yahoo_market_data() -> None:
    collector = YahooMarketCollector()
    collector.collect()


# Alias pour garder compatibilité avec ton pipeline actuel
def collect_market_data() -> None:
    collect_yahoo_market_data()


if __name__ == "__main__":
    collect_yahoo_market_data()

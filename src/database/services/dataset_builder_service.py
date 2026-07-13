"""
Service de construction des datasets depuis PostgreSQL.

Responsabilités :
- lire les données avec MarketBarRepository
- vérifier la qualité des données OHLCV
- supprimer les doublons
- trier les bougies
- ajouter les informations du symbole et du timeframe
- sauvegarder un export CSV facultatif
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.database.repositories.market_bar_repository import (
    MarketBarRepository,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


class DatasetBuilderService:
    """
    Construit des datasets à partir des données PostgreSQL.
    """

    def __init__(
        self,
        repository: MarketBarRepository | None = None,
    ) -> None:
        self.repository = repository or MarketBarRepository()

    def build_market_dataset(
        self,
        symbol: str,
        timeframe: str,
        platform: str = "TradingView",
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Charge et nettoie un dataset OHLCV.

        Parameters
        ----------
        symbol
            Exemple : GC.

        timeframe
            Exemple : 1m, 15m ou 1h.

        platform
            Exemple : TradingView.

        start_date
            Date minimale facultative.

        end_date
            Date maximale facultative.

        limit
            Nombre maximal de lignes facultatif.

        Returns
        -------
        pd.DataFrame
            Dataset OHLCV propre et trié.
        """

        logger.info(
            "Chargement PostgreSQL : "
            f"{platform} | {symbol} | {timeframe}"
        )

        df = self.repository.load_market_data(
            symbol=symbol,
            timeframe=timeframe,
            platform=platform,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        if df.empty:
            logger.warning(
                f"Aucune donnée trouvée pour {symbol} {timeframe}."
            )
            return df

        rows_before = len(df)

        df = self._clean_market_data(df)

        rows_after = len(df)

        logger.info(f"Lignes chargées : {rows_before}")
        logger.info(f"Lignes conservées : {rows_after}")
        logger.info(f"Lignes rejetées : {rows_before - rows_after}")

        return df

    def build_all_timeframes(
        self,
        symbol: str,
        timeframes: list[str],
        platform: str = "TradingView",
    ) -> dict[str, pd.DataFrame]:
        """
        Construit un dataset pour chaque timeframe demandé.

        Returns
        -------
        dict
            Exemple :
            {
                "1m": DataFrame,
                "5m": DataFrame,
                "15m": DataFrame
            }
        """

        datasets: dict[str, pd.DataFrame] = {}

        for timeframe in timeframes:
            df = self.build_market_dataset(
                symbol=symbol,
                timeframe=timeframe,
                platform=platform,
            )

            datasets[timeframe] = df

        return datasets

    def save_dataset(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        output_dir: str | Path = "data/processed/postgresql",
    ) -> Path:
        """
        Sauvegarde un dataset extrait depuis PostgreSQL.
        """

        if df.empty:
            raise ValueError(
                "Impossible de sauvegarder un dataset vide."
            )

        output_directory = Path(output_dir)

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_directory
            / symbol.upper()
            / timeframe.lower()
            / f"{symbol.upper()}_{timeframe.lower()}_postgresql.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            f"Dataset sauvegardé : {output_path}"
        )

        return output_path

    @staticmethod
    def _clean_market_data(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Nettoie et valide les données OHLCV.
        """

        df = df.copy()

        required_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes : {missing_columns}"
            )

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            utc=True,
            errors="coerce",
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        df = df.dropna(
            subset=[
                "datetime",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        valid_ohlc = (
            (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["high"] >= df["low"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
            & (df["low"] <= df["high"])
        )

        df = df[valid_ohlc]

        if "volume" in df.columns:
            valid_volume = (
                df["volume"].isna()
                | (df["volume"] >= 0)
            )

            df = df[valid_volume]

        df = df.drop_duplicates(
            subset=["datetime"],
            keep="last",
        )

        df = df.sort_values(
            by="datetime",
        )

        return df.reset_index(drop=True)
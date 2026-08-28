"""
Composants communs aux entraînements V3.

Responsabilités :
- valider le symbole et le timeframe
- charger le dataset Analysis V3
- sélectionner les features numériques
- créer les dossiers de modèles et rapports
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


VALID_TIMEFRAMES = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}


COLUMNS_TO_EXCLUDE = {
    "datetime",
    "symbol",
    "timeframe",
    "platform",
    "quality_status",
    "file_name",
    "target",
    "future_return_1",
}


class BaseTrainer:
    """
    Classe de base utilisée par les entraîneurs V3.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.model_name = model_name.lower()

        self._validate_parameters()

        self.dataset_path = (
            Path("data/final/analysis")
            / self.symbol
            / (
                f"dataset_analysis_v3_"
                f"{self.symbol}_{self.timeframe}.csv"
            )
        )

        self.model_dir = (
            Path("models")
            / self.symbol
            / self.timeframe
            / self.model_name
        )

        self.report_dir = (
            Path("reports/training")
            / self.symbol
            / self.timeframe
            / self.model_name
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _validate_parameters(self) -> None:
        """
        Vérifie les paramètres principaux.
        """

        if not self.symbol:
            raise ValueError(
                "Le symbole ne peut pas être vide."
            )

        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Timeframe invalide : {self.timeframe}. "
                f"Valeurs acceptées : {sorted(VALID_TIMEFRAMES)}"
            )

        if not self.model_name:
            raise ValueError(
                "Le nom du modèle ne peut pas être vide."
            )

    def load_analysis_dataset(self) -> pd.DataFrame:
        """
        Charge le dataset Analysis V3.
        """

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset introuvable : {self.dataset_path}"
            )

        logger.info(
            f"Chargement du dataset : {self.dataset_path}"
        )

        df = pd.read_csv(
            self.dataset_path
        )

        if df.empty:
            raise ValueError(
                f"Le dataset est vide : {self.dataset_path}"
            )

        if "target" not in df.columns:
            raise ValueError(
                "La colonne target est absente."
            )

        logger.info(
            f"Dimensions du dataset : {df.shape}"
        )

        return df

    def prepare_tabular_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prépare X et y pour les modèles tabulaires.
        """

        feature_columns = [
            column
            for column in df.columns
            if column not in COLUMNS_TO_EXCLUDE
        ]

        X = df[feature_columns].copy()

        bool_columns = X.select_dtypes(
            include=["bool"]
        ).columns

        X[bool_columns] = X[
            bool_columns
        ].astype(int)

        X = X.select_dtypes(
            include=["number"]
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X = X.fillna(0)

        y = df["target"].astype(int)

        logger.info(
            f"Nombre de features : {X.shape[1]}"
        )

        logger.info(
            f"Distribution target :\n{y.value_counts()}"
        )

        return X, y

    def get_feature_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """
        Retourne les features numériques disponibles.
        """

        X, _ = self.prepare_tabular_features(df)

        return X.columns.tolist()
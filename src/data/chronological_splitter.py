"""
Découpage chronologique commun aux modèles V3.

Le découpage respecte l'ordre temporel :

70 % entraînement
15 % validation
15 % test
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ChronologicalSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class ChronologicalSplitter:
    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
    ) -> None:
        if train_ratio <= 0:
            raise ValueError(
                "train_ratio doit être supérieur à zéro."
            )

        if validation_ratio < 0:
            raise ValueError(
                "validation_ratio ne peut pas être négatif."
            )

        if train_ratio + validation_ratio >= 1:
            raise ValueError(
                "La somme train_ratio + validation_ratio "
                "doit être inférieure à 1."
            )

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ChronologicalSplit:
        """
        Sépare X et y sans mélange aléatoire.
        """

        if len(X) != len(y):
            raise ValueError(
                "X et y doivent avoir le même nombre de lignes."
            )

        if len(X) < 10:
            raise ValueError(
                "Le dataset est trop petit pour être séparé."
            )

        total_rows = len(X)

        train_end = int(
            total_rows * self.train_ratio
        )

        validation_end = int(
            total_rows
            * (
                self.train_ratio
                + self.validation_ratio
            )
        )

        return ChronologicalSplit(
            X_train=X.iloc[:train_end].copy(),
            X_validation=X.iloc[
                train_end:validation_end
            ].copy(),
            X_test=X.iloc[
                validation_end:
            ].copy(),
            y_train=y.iloc[:train_end].copy(),
            y_validation=y.iloc[
                train_end:validation_end
            ].copy(),
            y_test=y.iloc[
                validation_end:
            ].copy(),
        )
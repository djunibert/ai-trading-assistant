"""
Pipeline Builder.

Permet de construire un pipeline étape par étape.
Chaque étape doit avoir une méthode detect(df).
"""

from __future__ import annotations

import pandas as pd


class PipelineBuilder:
    def __init__(self):
        self.steps = []

    def add(self, step):
        """
        Ajoute une étape au pipeline.
        """
        self.steps.append(step)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute toutes les étapes dans l'ordre.
        """
        df = df.copy()

        for step in self.steps:
            df = step.detect(df)

        return df
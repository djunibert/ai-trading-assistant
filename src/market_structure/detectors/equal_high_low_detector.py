"""
Equal High / Equal Low Detector

Détection adaptative basée sur l'ATR.
"""

from __future__ import annotations

import pandas as pd


class EqualHighLowDetector:
    """
    Détecte les Equal High et Equal Low.

    La tolérance n'est pas fixe.
    Elle est calculée avec l'ATR :

        tolerance = ATR × atr_multiplier
    """

    def __init__(self, atr_multiplier: float):
        self.atr_multiplier = atr_multiplier

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["equal_high"] = False
        df["equal_low"] = False

        required_columns = [
            "is_swing_high",
            "is_swing_low",
            "high",
            "low",
            "atr_14",
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante : {col}")

        last_swing_high = None
        last_swing_low = None

        for i in range(len(df)):
            atr = float(df.loc[i, "atr_14"])
            tolerance = atr * self.atr_multiplier

            if bool(df.loc[i, "is_swing_high"]):
                current_high = float(df.loc[i, "high"])

                if last_swing_high is not None:
                    if abs(current_high - last_swing_high) <= tolerance:
                        df.loc[i, "equal_high"] = True

                last_swing_high = current_high

            if bool(df.loc[i, "is_swing_low"]):
                current_low = float(df.loc[i, "low"])

                if last_swing_low is not None:
                    if abs(current_low - last_swing_low) <= tolerance:
                        df.loc[i, "equal_low"] = True

                last_swing_low = current_low

        return df

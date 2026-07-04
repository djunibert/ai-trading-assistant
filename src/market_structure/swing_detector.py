"""
Détection des Swing High et Swing Low.

Auteur : Junior Hébert
Projet : AI Trading System
"""

import numpy as np
import pandas as pd


class SwingDetector:
    """
    Détecte les Swing High et Swing Low.
    """

    def __init__(self, window: int = 2):
        """
        window=2 signifie :

        2 bougies avant
        +
        bougie actuelle
        +
        2 bougies après
        """
        self.window = window

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df["is_swing_high"] = False
        df["is_swing_low"] = False

        highs = df["high"].values
        lows = df["low"].values

        n = len(df)

        for i in range(self.window, n - self.window):

            current_high = highs[i]

            left_high = highs[i - self.window:i]
            right_high = highs[i + 1:i + self.window + 1]

            if (
                current_high > left_high.max()
                and current_high > right_high.max()
            ):
                df.loc[i, "is_swing_high"] = True

            current_low = lows[i]

            left_low = lows[i - self.window:i]
            right_low = lows[i + 1:i + self.window + 1]

            if (
                current_low < left_low.min()
                and current_low < right_low.min()
            ):
                df.loc[i, "is_swing_low"] = True

        df["last_swing_high"] = np.where(
            df["is_swing_high"],
            df["high"],
            np.nan,
        )

        df["last_swing_low"] = np.where(
            df["is_swing_low"],
            df["low"],
            np.nan,
        )

        df["last_swing_high"] = (
            df["last_swing_high"]
            .ffill()
        )

        df["last_swing_low"] = (
            df["last_swing_low"]
            .ffill()
        )

        df["distance_to_swing_high"] = (
            df["last_swing_high"] - df["close"]
        )

        df["distance_to_swing_low"] = (
            df["close"] - df["last_swing_low"]
        )

        return df
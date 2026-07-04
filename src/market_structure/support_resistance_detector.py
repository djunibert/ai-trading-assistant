"""
Détection des supports et résistances.

Support :
    dernier swing low

Resistance :
    dernier swing high
"""

import pandas as pd


class SupportResistanceDetector:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["support_price"] = df["last_swing_low"]
        df["resistance_price"] = df["last_swing_high"]

        df["distance_to_support"] = df["close"] - df["support_price"]
        df["distance_to_resistance"] = df["resistance_price"] - df["close"]

        df["support_broken"] = df["close"] < df["support_price"].shift(1)
        df["resistance_broken"] = df["close"] > df["resistance_price"].shift(1)

        df["near_support"] = (
            df["distance_to_support"].abs() <= df["atr_14"]
        )

        df["near_resistance"] = (
            df["distance_to_resistance"].abs() <= df["atr_14"]
        )

        return df
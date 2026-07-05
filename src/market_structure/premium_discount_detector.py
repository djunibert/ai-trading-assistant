"""
Premium / Discount Detector
"""

import pandas as pd


class PremiumDiscountDetector:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["range_high"] = df["last_swing_high"]
        df["range_low"] = df["last_swing_low"]

        df["equilibrium_price"] = (
            df["range_high"] + df["range_low"]
        ) / 2

        df["premium_zone"] = df["close"] > df["equilibrium_price"]
        df["discount_zone"] = df["close"] < df["equilibrium_price"]

        df["distance_to_equilibrium"] = (
            df["close"] - df["equilibrium_price"]
        )

        df["equilibrium_zone"] = (
            df["distance_to_equilibrium"].abs()
            <= df["atr_14"] * 0.10
        )

        return df
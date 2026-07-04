"""
Détection Premium / Discount.

Basé sur le dernier swing high et le dernier swing low.

Premium :
    close > midpoint

Discount :
    close < midpoint

Equilibrium :
    close proche du midpoint
"""

import pandas as pd


class PremiumDiscountDetector:
    def __init__(self, equilibrium_tolerance: float = 0.001):
        self.equilibrium_tolerance = equilibrium_tolerance

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
            / df["equilibrium_price"]
        ) <= self.equilibrium_tolerance

        return df
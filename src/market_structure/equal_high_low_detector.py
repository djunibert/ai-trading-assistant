"""
Détection des Equal High et Equal Low.

Equal High :
    deux swing highs proches selon une tolérance

Equal Low :
    deux swing lows proches selon une tolérance
"""

import pandas as pd


class EqualHighLowDetector:
    def __init__(self, tolerance: float = 0.001):
        """
        tolerance = 0.001 signifie 0.1%
        """
        self.tolerance = tolerance

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["equal_high"] = False
        df["equal_low"] = False

        previous_swing_high = (
            df["last_swing_high"]
            .where(df["is_swing_high"])
            .shift(1)
            .ffill()
        )

        previous_swing_low = (
            df["last_swing_low"]
            .where(df["is_swing_low"])
            .shift(1)
            .ffill()
        )

        high_diff_pct = (
            (df["last_swing_high"] - previous_swing_high).abs()
            / previous_swing_high
        )

        low_diff_pct = (
            (df["last_swing_low"] - previous_swing_low).abs()
            / previous_swing_low
        )

        df.loc[
            df["is_swing_high"] & (high_diff_pct <= self.tolerance),
            "equal_high",
        ] = True

        df.loc[
            df["is_swing_low"] & (low_diff_pct <= self.tolerance),
            "equal_low",
        ] = True

        df["liquidity_above"] = df["equal_high"]
        df["liquidity_below"] = df["equal_low"]

        return df
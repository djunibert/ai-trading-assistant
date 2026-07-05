"""
Détection de tendance basée sur les Swing High / Swing Low.

Trend :
  1  = Bullish
  0  = Neutral
 -1  = Bearish
"""

import pandas as pd


class TrendDetector:
    """
    Détecte la tendance à partir des derniers swings.
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["previous_swing_high"] = (
            df["last_swing_high"]
            .where(df["is_swing_high"])
            .shift(1)
            .ffill()
        )

        df["previous_swing_low"] = (
            df["last_swing_low"]
            .where(df["is_swing_low"])
            .shift(1)
            .ffill()
        )

        df["higher_high"] = df["last_swing_high"] > df["previous_swing_high"]
        df["higher_low"] = df["last_swing_low"] > df["previous_swing_low"]

        df["lower_high"] = df["last_swing_high"] < df["previous_swing_high"]
        df["lower_low"] = df["last_swing_low"] < df["previous_swing_low"]

        df["trend"] = 0

        df.loc[
            df["higher_high"] & df["higher_low"],
            "trend"
        ] = 1

        df.loc[
            df["lower_high"] & df["lower_low"],
            "trend"
        ] = -1

        #df["trend"] = df["trend"].replace(0, pd.NA).ffill().fillna(0)
        df["trend"] = (df["trend"]
        .replace(0, pd.NA)
        .ffill()
        .infer_objects(copy=False)
        .fillna(0))

        return df

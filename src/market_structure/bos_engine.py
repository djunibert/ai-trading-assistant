"""
BOS Engine V2.

Ajoute des informations avancées sur les Break Of Structure :
- force du BOS
- force en ATR
- volume ratio
- confirmation
- distance au BOS
- score de qualité
"""

import pandas as pd


class BOSEngine:
    def __init__(self, volume_window: int = 20):
        self.volume_window = volume_window

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["bos_v2"] = 0
        df["bos_v2_direction"] = "NONE"
        df["bos_price"] = None

        df["bos_strength_points"] = 0.0
        df["bos_strength_atr"] = 0.0
        df["bos_volume_ratio"] = 0.0
        df["bos_confirmed"] = False
        df["distance_to_bos"] = 0.0
        df["bos_score"] = 0.0

        previous_swing_high = df["last_swing_high"].shift(1)
        previous_swing_low = df["last_swing_low"].shift(1)

        volume_average = df["volume"].rolling(
            window=self.volume_window
        ).mean()

        bullish_bos = df["close"] > previous_swing_high
        bearish_bos = df["close"] < previous_swing_low

        df.loc[bullish_bos, "bos_v2"] = 1
        df.loc[bullish_bos, "bos_v2_direction"] = "BULLISH"
        df.loc[bullish_bos, "bos_price"] = previous_swing_high
        df.loc[bullish_bos, "bos_strength_points"] = (
            df["close"] - previous_swing_high
        )

        df.loc[bearish_bos, "bos_v2"] = -1
        df.loc[bearish_bos, "bos_v2_direction"] = "BEARISH"
        df.loc[bearish_bos, "bos_price"] = previous_swing_low
        df.loc[bearish_bos, "bos_strength_points"] = (
            previous_swing_low - df["close"]
        )

        df["bos_strength_atr"] = (
            df["bos_strength_points"] / df["atr_14"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        df["bos_volume_ratio"] = (
            df["volume"] / volume_average
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        df["bos_confirmed"] = df["bos_strength_atr"] >= 0.25

        df["last_bos_price"] = (
            df["bos_price"]
            .where(df["bos_v2"] != 0)
            .ffill()
        )

        df["distance_to_bos"] = df["close"] - df["last_bos_price"]

        df["bos_score"] = 0.0

        df.loc[df["bos_strength_atr"] >= 0.25, "bos_score"] += 20
        df.loc[df["bos_strength_atr"] >= 0.50, "bos_score"] += 20
        df.loc[df["bos_strength_atr"] >= 1.00, "bos_score"] += 20

        df.loc[df["bos_volume_ratio"] >= 1.0, "bos_score"] += 15
        df.loc[df["bos_volume_ratio"] >= 1.5, "bos_score"] += 15

        df.loc[df["bos_confirmed"], "bos_score"] += 10

        df.loc[df["bos_v2"] == 0, "bos_score"] = 0

        return df
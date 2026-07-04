"""
Détection du BOS : Break Of Structure.

BOS bullish :
    close > dernier swing high précédent

BOS bearish :
    close < dernier swing low précédent
"""

import pandas as pd


class BOSDetector:
    """
    Détecte les BOS bullish et bearish.
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["bos"] = 0
        df["bos_bullish"] = False
        df["bos_bearish"] = False
        df["bos_direction"] = "NONE"
        df["bos_strength"] = 0.0

        previous_swing_high = df["last_swing_high"].shift(1)
        previous_swing_low = df["last_swing_low"].shift(1)

        bullish_bos = df["close"] > previous_swing_high
        bearish_bos = df["close"] < previous_swing_low

        df.loc[bullish_bos, "bos"] = 1
        df.loc[bullish_bos, "bos_bullish"] = True
        df.loc[bullish_bos, "bos_direction"] = "BULLISH"
        df.loc[bullish_bos, "bos_strength"] = (
            df["close"] - previous_swing_high
        )

        df.loc[bearish_bos, "bos"] = -1
        df.loc[bearish_bos, "bos_bearish"] = True
        df.loc[bearish_bos, "bos_direction"] = "BEARISH"
        df.loc[bearish_bos, "bos_strength"] = (
            previous_swing_low - df["close"]
        )

        df["bos_event"] = df["bos"] != 0

        df["last_bos_direction"] = (
            df["bos_direction"]
            .where(df["bos_event"])
            .replace("NONE", pd.NA)
            .ffill()
            .fillna("NONE")
        )

        return df
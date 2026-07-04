"""
Détection du BOS : Break Of Structure.

BOS bullish :
    close > dernier swing high

BOS bearish :
    close < dernier swing low
"""

import pandas as pd


class BOSDetector:
    """
    Détecte les cassures de structure.
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["bos"] = 0
        df["bos_bullish"] = False
        df["bos_bearish"] = False

        df["bos_strength"] = 0.0

        bullish_condition = df["close"] > df["last_swing_high"].shift(1)
        bearish_condition = df["close"] < df["last_swing_low"].shift(1)

        df.loc[bullish_condition, "bos"] = 1
        df.loc[bullish_condition, "bos_bullish"] = True
        df.loc[bullish_condition, "bos_strength"] = (
            df["close"] - df["last_swing_high"].shift(1)
        )

        df.loc[bearish_condition, "bos"] = -1
        df.loc[bearish_condition, "bos_bearish"] = True
        df.loc[bearish_condition, "bos_strength"] = (
            df["last_swing_low"].shift(1) - df["close"]
        )

        df["bars_since_bos"] = (
            df["bos"]
            .ne(0)
            .cumsum()
        )

        return df
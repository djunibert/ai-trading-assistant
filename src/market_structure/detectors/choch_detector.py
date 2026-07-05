"""
Détection du CHOCH : Change Of Character.

CHOCH bullish :
    tendance précédente bearish
    +
    BOS bullish

CHOCH bearish :
    tendance précédente bullish
    +
    BOS bearish
"""

import pandas as pd


class CHOCHDetector:
    """
    Détecte les changements de caractère du marché.
    """

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["choch"] = 0
        df["choch_bullish"] = False
        df["choch_bearish"] = False
        df["choch_direction"] = "NONE"

        previous_trend = df["trend"].shift(1)

        choch_bullish = (previous_trend == -1) & (df["bos_bullish"])
        choch_bearish = (previous_trend == 1) & (df["bos_bearish"])

        df.loc[choch_bullish, "choch"] = 1
        df.loc[choch_bullish, "choch_bullish"] = True
        df.loc[choch_bullish, "choch_direction"] = "BULLISH"

        df.loc[choch_bearish, "choch"] = -1
        df.loc[choch_bearish, "choch_bearish"] = True
        df.loc[choch_bearish, "choch_direction"] = "BEARISH"

        df["choch_event"] = df["choch"] != 0

        df["last_choch_direction"] = (
            df["choch_direction"]
            .where(df["choch_event"])
            .replace("NONE", pd.NA)
            .ffill()
            .fillna("NONE")
        )

        return df

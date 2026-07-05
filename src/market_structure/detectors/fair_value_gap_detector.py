"""
Détection des Fair Value Gaps.

Bullish FVG :
    low[i] > high[i-2]

Bearish FVG :
    high[i] < low[i-2]
"""

import pandas as pd


class FairValueGapDetector:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["fvg"] = 0
        df["fvg_bullish"] = False
        df["fvg_bearish"] = False
        df["fvg_direction"] = "NONE"
        df["fvg_top"] = None
        df["fvg_bottom"] = None
        df["fvg_size"] = 0.0

        high_two_bars_ago = df["high"].shift(2)
        low_two_bars_ago = df["low"].shift(2)

        bullish_fvg = df["low"] > high_two_bars_ago
        bearish_fvg = df["high"] < low_two_bars_ago

        df.loc[bullish_fvg, "fvg"] = 1
        df.loc[bullish_fvg, "fvg_bullish"] = True
        df.loc[bullish_fvg, "fvg_direction"] = "BULLISH"
        df.loc[bullish_fvg, "fvg_top"] = df["low"]
        df.loc[bullish_fvg, "fvg_bottom"] = high_two_bars_ago
        df.loc[bullish_fvg, "fvg_size"] = df["low"] - high_two_bars_ago

        df.loc[bearish_fvg, "fvg"] = -1
        df.loc[bearish_fvg, "fvg_bearish"] = True
        df.loc[bearish_fvg, "fvg_direction"] = "BEARISH"
        df.loc[bearish_fvg, "fvg_top"] = low_two_bars_ago
        df.loc[bearish_fvg, "fvg_bottom"] = df["high"]
        df.loc[bearish_fvg, "fvg_size"] = low_two_bars_ago - df["high"]

        return df

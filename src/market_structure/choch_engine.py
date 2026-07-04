"""
CHOCH Engine V2.

Ajoute :
- force du CHOCH
- confirmation
- confluence avec liquidity sweep
- score de qualité
"""

import pandas as pd


class CHOCHEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["choch_v2"] = 0
        df["choch_v2_direction"] = "NONE"
        df["choch_strength_atr"] = 0.0
        df["choch_confirmed"] = False
        df["choch_after_liquidity"] = False
        df["choch_score"] = 0.0

        previous_trend = df["trend"].shift(1)

        bullish_choch = (previous_trend == -1) & (df["bos_v2"] == 1)
        bearish_choch = (previous_trend == 1) & (df["bos_v2"] == -1)

        df.loc[bullish_choch, "choch_v2"] = 1
        df.loc[bullish_choch, "choch_v2_direction"] = "BULLISH"

        df.loc[bearish_choch, "choch_v2"] = -1
        df.loc[bearish_choch, "choch_v2_direction"] = "BEARISH"

        df.loc[df["choch_v2"] != 0, "choch_strength_atr"] = df["bos_strength_atr"]

        df["choch_confirmed"] = df["choch_strength_atr"] >= 0.25

        if "liquidity_sweep" in df.columns:
            df["recent_liquidity_sweep"] = (
                df["liquidity_sweep"]
                .ne(0)
                .rolling(window=5)
                .max()
                .fillna(False)
                .astype(bool)
            )

            df["choch_after_liquidity"] = (
                (df["choch_v2"] != 0)
                & df["recent_liquidity_sweep"]
            )

        df.loc[df["choch_strength_atr"] >= 0.25, "choch_score"] += 25
        df.loc[df["choch_strength_atr"] >= 0.50, "choch_score"] += 25
        df.loc[df["choch_confirmed"], "choch_score"] += 25
        df.loc[df["choch_after_liquidity"], "choch_score"] += 25

        df.loc[df["choch_v2"] == 0, "choch_score"] = 0

        return df
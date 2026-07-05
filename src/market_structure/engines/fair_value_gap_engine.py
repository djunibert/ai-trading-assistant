"""
Fair Value Gap Engine V2.

Suit la vie du dernier FVG détecté :
- actif
- âge
- fill percent
- partial fill
- complete fill
- distance
- score
"""

import pandas as pd


class FairValueGapEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["fvg_v2_active"] = False
        df["fvg_v2_age"] = 0
        df["fvg_v2_fill_percent"] = 0.0
        df["fvg_v2_partial_fill"] = False
        df["fvg_v2_complete_fill"] = False
        df["fvg_v2_distance"] = 0.0
        df["fvg_v2_score"] = 0.0

        active_direction = "NONE"
        active_top = None
        active_bottom = None
        age = 0

        for i in range(len(df)):

            if df.loc[i, "fvg"] != 0:
                active_direction = df.loc[i, "fvg_direction"]
                active_top = df.loc[i, "fvg_top"]
                active_bottom = df.loc[i, "fvg_bottom"]
                age = 0

            if active_direction != "NONE" and active_top is not None and active_bottom is not None:

                age += 1

                current_high = df.loc[i, "high"]
                current_low = df.loc[i, "low"]
                current_close = df.loc[i, "close"]

                fvg_size = active_top - active_bottom

                if fvg_size <= 0:
                    continue

                if active_direction == "BULLISH":
                    fill_amount = max(0, active_top - current_low)
                    distance = current_close - active_bottom

                else:
                    fill_amount = max(0, current_high - active_bottom)
                    distance = active_top - current_close

                fill_percent = min(100, (fill_amount / fvg_size) * 100)

                partial_fill = 0 < fill_percent < 100
                complete_fill = fill_percent >= 100
                active = not complete_fill

                score = 0

                if active:
                    score += 25

                if age <= 20:
                    score += 20

                if fill_percent < 50:
                    score += 25

                if fill_percent < 100:
                    score += 20

                if distance >= 0:
                    score += 10

                df.loc[i, "fvg_v2_active"] = active
                df.loc[i, "fvg_v2_age"] = age
                df.loc[i, "fvg_v2_fill_percent"] = round(fill_percent, 2)
                df.loc[i, "fvg_v2_partial_fill"] = partial_fill
                df.loc[i, "fvg_v2_complete_fill"] = complete_fill
                df.loc[i, "fvg_v2_distance"] = round(distance, 4)
                df.loc[i, "fvg_v2_score"] = score

        return df

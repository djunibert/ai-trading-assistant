"""
Order Block Lifecycle Engine.

Suit la vie du dernier Order Block détecté :
- actif
- âge
- nombre de touches
- retesté
- mitigé
- invalidé
- score
"""

import pandas as pd


class OrderBlockLifecycleEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ob_lifecycle_active"] = False
        df["ob_lifecycle_age"] = 0
        df["ob_lifecycle_touch_count"] = 0
        df["ob_lifecycle_retested"] = False
        df["ob_lifecycle_mitigated"] = False
        df["ob_lifecycle_invalidated"] = False
        df["ob_lifecycle_score"] = 0.0

        active_direction = "NONE"
        active_top = None
        active_bottom = None
        age = 0
        touch_count = 0
        retested = False
        mitigated = False
        invalidated = False

        for i in range(len(df)):

            # Nouveau Order Block détecté
            if df.loc[i, "ob_v2"] != 0:
                active_direction = df.loc[i, "ob_v2_direction"]
                active_top = df.loc[i, "ob_v2_top"]
                active_bottom = df.loc[i, "ob_v2_bottom"]

                age = 0
                touch_count = 0
                retested = False
                mitigated = False
                invalidated = False

            # Si un Order Block actif existe
            if active_direction != "NONE" and active_top is not None and active_bottom is not None:

                age += 1

                current_high = df.loc[i, "high"]
                current_low = df.loc[i, "low"]
                current_close = df.loc[i, "close"]

                touches_zone = (
                    current_low <= active_top
                    and current_high >= active_bottom
                )

                if touches_zone:
                    touch_count += 1

                    if age > 1:
                        retested = True

                # Mitigation simple : le prix traverse au moins le milieu de la zone
                midpoint = (active_top + active_bottom) / 2

                if active_direction == "BULLISH":
                    if current_low <= midpoint:
                        mitigated = True

                    if current_close < active_bottom:
                        invalidated = True

                if active_direction == "BEARISH":
                    if current_high >= midpoint:
                        mitigated = True

                    if current_close > active_top:
                        invalidated = True

                active = not invalidated

                score = 0

                if active:
                    score += 25

                if retested:
                    score += 15

                if not mitigated:
                    score += 20

                if touch_count <= 2:
                    score += 20

                if age <= 20:
                    score += 20

                df.loc[i, "ob_lifecycle_active"] = active
                df.loc[i, "ob_lifecycle_age"] = age
                df.loc[i, "ob_lifecycle_touch_count"] = touch_count
                df.loc[i, "ob_lifecycle_retested"] = retested
                df.loc[i, "ob_lifecycle_mitigated"] = mitigated
                df.loc[i, "ob_lifecycle_invalidated"] = invalidated
                df.loc[i, "ob_lifecycle_score"] = score

        return df
"""
Trade Setup Engine.

Calcule les scores BUY et SELL à partir des éléments
de structure de marché.
"""

import pandas as pd


class TradeSetupEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["buy_setup_score"] = 0.0
        df["sell_setup_score"] = 0.0

        # BUY setup
        df.loc[df["bos_v2"] == 1, "buy_setup_score"] += 20
        df.loc[df["choch_v2"] == 1, "buy_setup_score"] += 15
        df.loc[df["ob_v2"] == 1, "buy_setup_score"] += 20
        df.loc[df["fvg"] == 1, "buy_setup_score"] += 15
        df.loc[df["sell_side_sweep"], "buy_setup_score"] += 15
        df.loc[df["discount_zone"], "buy_setup_score"] += 10
        df.loc[df["london_session"] | df["new_york_session"], "buy_setup_score"] += 5

        # SELL setup
        df.loc[df["bos_v2"] == -1, "sell_setup_score"] += 20
        df.loc[df["choch_v2"] == -1, "sell_setup_score"] += 15
        df.loc[df["ob_v2"] == -1, "sell_setup_score"] += 20
        df.loc[df["fvg"] == -1, "sell_setup_score"] += 15
        df.loc[df["buy_side_sweep"], "sell_setup_score"] += 15
        df.loc[df["premium_zone"], "sell_setup_score"] += 10
        df.loc[df["london_session"] | df["new_york_session"], "sell_setup_score"] += 5

        df["buy_setup"] = df["buy_setup_score"] >= 70
        df["sell_setup"] = df["sell_setup_score"] >= 70

        df["trade_setup"] = "NO_TRADE"
        df.loc[df["buy_setup"], "trade_setup"] = "BUY"
        df.loc[df["sell_setup"], "trade_setup"] = "SELL"

        return df
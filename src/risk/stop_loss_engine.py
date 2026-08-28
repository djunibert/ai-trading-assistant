"""
Stop Loss Engine.

Calcule un stop loss intelligent selon :
- le signal BUY / SELL
- l'ATR
- les swings
- les Order Blocks
"""

import pandas as pd


class StopLossEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["sl_method"] = "NONE"
        df["sl_price"] = 0.0
        df["sl_distance"] = 0.0

        buy_condition = df["risk_signal"] == "BUY"
        sell_condition = df["risk_signal"] == "SELL"

        # BUY : stop sous le dernier swing low
        df.loc[buy_condition, "sl_price"] = df["last_swing_low"]

        # SELL : stop au-dessus du dernier swing high
        df.loc[sell_condition, "sl_price"] = df["last_swing_high"]

        df.loc[buy_condition, "sl_method"] = "SWING_LOW"
        df.loc[sell_condition, "sl_method"] = "SWING_HIGH"

        df["sl_distance"] = (df["close"] - df["sl_price"]).abs()

        return df
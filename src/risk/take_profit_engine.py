"""
Take Profit Engine.

Calcule un take profit intelligent selon :
- le signal BUY / SELL
- le stop loss
- le ratio risk/reward
"""

import pandas as pd

from src.config.market_structure_config import DEFAULT_RISK_REWARD


class TakeProfitEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["tp_method"] = "NONE"
        df["tp_price"] = 0.0
        df["tp_distance"] = 0.0

        buy_condition = df["risk_signal"] == "BUY"
        sell_condition = df["risk_signal"] == "SELL"

        df.loc[buy_condition, "tp_price"] = (
            df["close"] + (df["sl_distance"] * DEFAULT_RISK_REWARD)
        )

        df.loc[sell_condition, "tp_price"] = (
            df["close"] - (df["sl_distance"] * DEFAULT_RISK_REWARD)
        )

        df.loc[buy_condition | sell_condition, "tp_method"] = "RISK_REWARD"

        df["tp_distance"] = (df["tp_price"] - df["close"]).abs()

        return df
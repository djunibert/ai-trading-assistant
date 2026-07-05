"""
Risk Engine V1.

Calcule :
- signal final
- autorisation du trade
- entry price
- stop loss
- take profit
- distance du stop
- risk/reward
"""

import pandas as pd

from src.config.market_structure_config import (
    DEFAULT_RISK_REWARD,
    BUY_SETUP_THRESHOLD,
    SELL_SETUP_THRESHOLD,
)


class RiskEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["risk_signal"] = "NO_TRADE"
        df["risk_trade_allowed"] = False

        df["risk_entry_price"] = df["close"]
        df["risk_stop_loss"] = 0.0
        df["risk_take_profit"] = 0.0
        df["risk_stop_distance"] = 0.0
        df["risk_reward_ratio"] = DEFAULT_RISK_REWARD

        buy_condition = df["buy_setup_score"] >= BUY_SETUP_THRESHOLD
        sell_condition = df["sell_setup_score"] >= SELL_SETUP_THRESHOLD

        # BUY
        df.loc[buy_condition, "risk_signal"] = "BUY"
        df.loc[buy_condition, "risk_trade_allowed"] = True
        df.loc[buy_condition, "risk_stop_loss"] = (
            df["risk_entry_price"] - df["atr_14"]
        )
        df.loc[buy_condition, "risk_take_profit"] = (
            df["risk_entry_price"] + (df["atr_14"] * DEFAULT_RISK_REWARD)
        )
        df.loc[buy_condition, "risk_stop_distance"] = df["atr_14"]

        # SELL
        df.loc[sell_condition, "risk_signal"] = "SELL"
        df.loc[sell_condition, "risk_trade_allowed"] = True
        df.loc[sell_condition, "risk_stop_loss"] = (
            df["risk_entry_price"] + df["atr_14"]
        )
        df.loc[sell_condition, "risk_take_profit"] = (
            df["risk_entry_price"] - (df["atr_14"] * DEFAULT_RISK_REWARD)
        )
        df.loc[sell_condition, "risk_stop_distance"] = df["atr_14"]

        # Si BUY et SELL arrivent en même temps, on annule le trade
        conflict = buy_condition & sell_condition

        df.loc[conflict, "risk_signal"] = "NO_TRADE"
        df.loc[conflict, "risk_trade_allowed"] = False
        df.loc[conflict, "risk_stop_loss"] = 0.0
        df.loc[conflict, "risk_take_profit"] = 0.0
        df.loc[conflict, "risk_stop_distance"] = 0.0

        return df
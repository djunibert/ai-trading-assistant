"""
Détection des Liquidity Sweeps.

Buy-side liquidity sweep :
    le prix dépasse un Equal High / dernier Swing High
    puis clôture en dessous.

Sell-side liquidity sweep :
    le prix passe sous un Equal Low / dernier Swing Low
    puis clôture au-dessus.
"""

import pandas as pd


class LiquidityDetector:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["liquidity_sweep"] = 0
        df["buy_side_sweep"] = False
        df["sell_side_sweep"] = False
        df["liquidity_sweep_direction"] = "NONE"
        df["liquidity_sweep_strength"] = 0.0

        previous_swing_high = df["last_swing_high"].shift(1)
        previous_swing_low = df["last_swing_low"].shift(1)

        buy_side_sweep = (
            (df["high"] > previous_swing_high)
            & (df["close"] < previous_swing_high)
        )

        sell_side_sweep = (
            (df["low"] < previous_swing_low)
            & (df["close"] > previous_swing_low)
        )

        df.loc[buy_side_sweep, "liquidity_sweep"] = -1
        df.loc[buy_side_sweep, "buy_side_sweep"] = True
        df.loc[buy_side_sweep, "liquidity_sweep_direction"] = "BUY_SIDE_SWEEP"
        df.loc[buy_side_sweep, "liquidity_sweep_strength"] = (
            df["high"] - previous_swing_high
        )

        df.loc[sell_side_sweep, "liquidity_sweep"] = 1
        df.loc[sell_side_sweep, "sell_side_sweep"] = True
        df.loc[sell_side_sweep, "liquidity_sweep_direction"] = "SELL_SIDE_SWEEP"
        df.loc[sell_side_sweep, "liquidity_sweep_strength"] = (
            previous_swing_low - df["low"]
        )

        return df

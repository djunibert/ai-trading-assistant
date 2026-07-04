"""
Détection simple des Order Blocks.

Bullish Order Block :
    dernière bougie bearish avant un BOS bullish

Bearish Order Block :
    dernière bougie bullish avant un BOS bearish
"""

import pandas as pd


class OrderBlockDetector:
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["order_block"] = 0
        df["order_block_direction"] = "NONE"
        df["order_block_top"] = None
        df["order_block_bottom"] = None
        df["order_block_size"] = 0.0

        for i in range(self.lookback, len(df)):

            if df.loc[i, "bos_bullish"]:
                recent = df.iloc[i - self.lookback:i]
                bearish_candles = recent[recent["close"] < recent["open"]]

                if not bearish_candles.empty:
                    ob = bearish_candles.iloc[-1]

                    df.loc[i, "order_block"] = 1
                    df.loc[i, "order_block_direction"] = "BULLISH"
                    df.loc[i, "order_block_top"] = ob["high"]
                    df.loc[i, "order_block_bottom"] = ob["low"]
                    df.loc[i, "order_block_size"] = ob["high"] - ob["low"]

            if df.loc[i, "bos_bearish"]:
                recent = df.iloc[i - self.lookback:i]
                bullish_candles = recent[recent["close"] > recent["open"]]

                if not bullish_candles.empty:
                    ob = bullish_candles.iloc[-1]

                    df.loc[i, "order_block"] = -1
                    df.loc[i, "order_block_direction"] = "BEARISH"
                    df.loc[i, "order_block_top"] = ob["high"]
                    df.loc[i, "order_block_bottom"] = ob["low"]
                    df.loc[i, "order_block_size"] = ob["high"] - ob["low"]

        return df
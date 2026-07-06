"""
Equity Curve.

Construit la courbe de capital à partir des trades.
"""

import pandas as pd


class EquityCurve:
    def build(
        self,
        trades_df: pd.DataFrame,
        initial_capital: float = 100000.0,
    ) -> pd.DataFrame:
        if trades_df.empty:
            return pd.DataFrame(
                columns=[
                    "trade_number",
                    "equity",
                    "pnl",
                    "drawdown",
                ]
            )

        equity_df = trades_df.copy()
        equity_df["trade_number"] = range(1, len(equity_df) + 1)
        equity_df["equity"] = initial_capital + equity_df["pnl"].cumsum()
        equity_df["running_max"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] - equity_df["running_max"]

        return equity_df[
            [
                "trade_number",
                "entry_datetime",
                "signal",
                "pnl",
                "equity",
                "drawdown",
            ]
        ]
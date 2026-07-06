"""
Statistiques de backtesting.

Calcule les métriques principales :
- nombre de trades
- win rate
- profit factor
- expectancy
- drawdown
"""

import pandas as pd


class BacktestStatistics:
    def calculate(self, trades_df: pd.DataFrame) -> dict:
        if trades_df.empty:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "max_drawdown": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
            }

        wins_df = trades_df[trades_df["result"] == "WIN"]
        losses_df = trades_df[trades_df["result"] == "LOSS"]

        total_trades = len(trades_df)
        wins = len(wins_df)
        losses = len(losses_df)

        gross_profit = wins_df["pnl"].sum()
        gross_loss = abs(losses_df["pnl"].sum())

        total_pnl = trades_df["pnl"].sum()

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0.0

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        average_win = wins_df["pnl"].mean() if wins > 0 else 0.0
        average_loss = losses_df["pnl"].mean() if losses > 0 else 0.0

        expectancy = (
            (win_rate / 100) * average_win
            + ((1 - win_rate / 100) * average_loss)
        )

        equity = trades_df["pnl"].cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = drawdown.min()

        return {
            "total_trades": int(total_trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": round(float(win_rate), 2),
            "total_pnl": round(float(total_pnl), 2),
            "profit_factor": round(float(profit_factor), 2),
            "expectancy": round(float(expectancy), 2),
            "max_drawdown": round(float(max_drawdown), 2),
            "average_win": round(float(average_win), 2),
            "average_loss": round(float(average_loss), 2),
        }
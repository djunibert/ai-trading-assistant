"""
Performance Metrics.

Calcule des métriques avancées de backtesting.
"""

import pandas as pd


class PerformanceMetrics:
    def calculate(self, trades_df: pd.DataFrame) -> dict:
        if trades_df.empty:
            return self._empty_metrics()

        gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
        gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()

        returns = trades_df["pnl"]

        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        max_drawdown = self._max_drawdown(returns)
        calmar_ratio = self._calmar_ratio(returns, max_drawdown)

        return {
            "gross_profit": round(float(gross_profit), 2),
            "gross_loss": round(float(gross_loss), 2),
            "average_trade": round(float(trades_df["pnl"].mean()), 2),
            "best_trade": round(float(trades_df["pnl"].max()), 2),
            "worst_trade": round(float(trades_df["pnl"].min()), 2),
            "longest_win_streak": self._longest_streak(trades_df, "WIN"),
            "longest_loss_streak": self._longest_streak(trades_df, "LOSS"),
            "sharpe_ratio": round(float(sharpe_ratio), 2),
            "sortino_ratio": round(float(sortino_ratio), 2),
            "calmar_ratio": round(float(calmar_ratio), 2),
        }

    def _empty_metrics(self) -> dict:
        return {
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "average_trade": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
        }

    def _longest_streak(self, trades_df: pd.DataFrame, result: str) -> int:
        longest = 0
        current = 0

        for value in trades_df["result"]:
            if value == result:
                current += 1
                longest = max(longest, current)
            else:
                current = 0

        return longest

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std()

    def _sortino_ratio(self, returns: pd.Series) -> float:
        downside_returns = returns[returns < 0]

        if downside_returns.empty or downside_returns.std() == 0:
            return 0.0

        return returns.mean() / downside_returns.std()

    def _max_drawdown(self, returns: pd.Series) -> float:
        equity = returns.cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max
        return float(drawdown.min())

    def _calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        total_return = returns.sum()

        if max_drawdown == 0:
            return 0.0

        return total_return / abs(max_drawdown)
"""
Backtesting Engine V2.

Simule les trades et calcule :
- statistiques
- historique des trades
- courbe de capital
"""

from dataclasses import asdict

import pandas as pd

from src.backtesting.trade import Trade
from src.backtesting.statistics import BacktestStatistics
from src.backtesting.equity_curve import EquityCurve


class BacktestingEngine:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame) -> dict:
        trades = []

        for entry_index in range(len(df) - 1):
            row = df.iloc[entry_index]

            if not bool(row["risk_trade_allowed"]):
                continue

            trade = Trade(
                entry_datetime=str(row["datetime"]),
                signal=str(row["risk_signal"]),
                entry_price=float(row["risk_entry_price"]),
                stop_loss=float(row["sl_price"]),
                take_profit=float(row["tp_price"]),
            )

            future_df = df.iloc[entry_index + 1:]

            for duration, (_, future_row) in enumerate(future_df.iterrows(), start=1):
                high = float(future_row["high"])
                low = float(future_row["low"])

                if trade.signal == "BUY":
                    if low <= trade.stop_loss:
                        trade.close_trade(
                            str(future_row["datetime"]),
                            trade.stop_loss,
                            "LOSS",
                            duration,
                        )
                        break

                    if high >= trade.take_profit:
                        trade.close_trade(
                            str(future_row["datetime"]),
                            trade.take_profit,
                            "WIN",
                            duration,
                        )
                        break

                elif trade.signal == "SELL":
                    if high >= trade.stop_loss:
                        trade.close_trade(
                            str(future_row["datetime"]),
                            trade.stop_loss,
                            "LOSS",
                            duration,
                        )
                        break

                    if low <= trade.take_profit:
                        trade.close_trade(
                            str(future_row["datetime"]),
                            trade.take_profit,
                            "WIN",
                            duration,
                        )
                        break

            trades.append(asdict(trade))

        trades_df = pd.DataFrame(trades)

        statistics = BacktestStatistics().calculate(trades_df)

        equity_curve = EquityCurve().build(
            trades_df=trades_df,
            initial_capital=self.initial_capital,
        )

        return {
            "statistics": statistics,
            "trades": trades_df,
            "equity_curve": equity_curve,
        }
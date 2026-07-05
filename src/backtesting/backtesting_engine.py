"""
Backtesting Engine V1.

Simule les trades à partir des colonnes :
- risk_trade_allowed
- risk_signal
- risk_entry_price
- sl_price
- tp_price
"""

import pandas as pd


class BacktestingEngine:
    def run(self, df: pd.DataFrame) -> dict:
        trades = []

        for i in range(len(df) - 1):
            row = df.iloc[i]

            if not row["risk_trade_allowed"]:
                continue

            signal = row["risk_signal"]
            entry = row["risk_entry_price"]
            sl = row["sl_price"]
            tp = row["tp_price"]

            future = df.iloc[i + 1:]

            result = "OPEN"
            pnl = 0.0

            for _, future_row in future.iterrows():
                high = future_row["high"]
                low = future_row["low"]

                if signal == "BUY":
                    if low <= sl:
                        result = "LOSS"
                        pnl = sl - entry
                        break

                    if high >= tp:
                        result = "WIN"
                        pnl = tp - entry
                        break

                if signal == "SELL":
                    if high >= sl:
                        result = "LOSS"
                        pnl = entry - sl
                        break

                    if low <= tp:
                        result = "WIN"
                        pnl = entry - tp
                        break

            trades.append({
                "datetime": row["datetime"],
                "signal": signal,
                "entry": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "result": result,
                "pnl": pnl,
            })

        trades_df = pd.DataFrame(trades)

        if trades_df.empty:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "trades": trades_df,
            }

        wins = (trades_df["result"] == "WIN").sum()
        losses = (trades_df["result"] == "LOSS").sum()
        total_trades = len(trades_df)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / total_trades) * 100, 2),
            "total_pnl": round(trades_df["pnl"].sum(), 2),
            "trades": trades_df,
        }
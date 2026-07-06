"""
Report Generator.

Génère les rapports de backtesting.
"""

import json
from pathlib import Path

import pandas as pd


class BacktestReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        statistics: dict,
        trades_df: pd.DataFrame,
        equity_curve: pd.DataFrame,
    ) -> None:
        trades_df.to_csv(
            self.output_dir / "backtesting_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )

        equity_curve.to_csv(
            self.output_dir / "backtesting_equity_curve.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame([statistics]).to_csv(
            self.output_dir / "backtesting_statistics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            self.output_dir / "backtesting_statistics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(statistics, file, indent=4, ensure_ascii=False)
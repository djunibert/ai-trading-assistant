"""
Modèle Trade.

Représente un trade simulé pendant le backtesting.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Trade:
    entry_datetime: str
    signal: str

    entry_price: float
    stop_loss: float
    take_profit: float

    exit_datetime: Optional[str] = None
    exit_price: Optional[float] = None

    result: str = "OPEN"
    pnl: float = 0.0
    duration_bars: int = 0

    risk_reward_ratio: float = 0.0

    def close_trade(
        self,
        exit_datetime: str,
        exit_price: float,
        result: str,
        duration_bars: int,
    ) -> None:
        self.exit_datetime = exit_datetime
        self.exit_price = exit_price
        self.result = result
        self.duration_bars = duration_bars

        if self.signal == "BUY":
            self.pnl = exit_price - self.entry_price

        elif self.signal == "SELL":
            self.pnl = self.entry_price - exit_price

        risk = abs(self.entry_price - self.stop_loss)

        if risk > 0:
            self.risk_reward_ratio = abs(self.pnl) / risk
        else:
            self.risk_reward_ratio = 0.0
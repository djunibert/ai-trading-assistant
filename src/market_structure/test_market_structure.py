import pandas as pd

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector
from src.market_structure.bos_detector import BOSDetector


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

df = SwingDetector(window=2).detect(df)
df = TrendDetector().detect(df)
df = BOSDetector().detect(df)

print(
    df[
        [
            "datetime",
            "close",
            "last_swing_high",
            "last_swing_low",
            "trend",
            "bos",
            "bos_bullish",
            "bos_bearish",
            "bos_strength",
        ]
    ].tail(40)
)
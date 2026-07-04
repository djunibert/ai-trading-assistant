import pandas as pd

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

swing_detector = SwingDetector(window=2)
trend_detector = TrendDetector()

df = swing_detector.detect(df)
df = trend_detector.detect(df)

print(
    df[
        [
            "datetime",
            "close",
            "is_swing_high",
            "is_swing_low",
            "last_swing_high",
            "last_swing_low",
            "higher_high",
            "higher_low",
            "lower_high",
            "lower_low",
            "trend",
        ]
    ].tail(40)
)
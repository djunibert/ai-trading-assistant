import pandas as pd

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector
from src.market_structure.bos_detector import BOSDetector
from src.market_structure.choch_detector import CHOCHDetector


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

df = SwingDetector(window=2).detect(df)
df = TrendDetector().detect(df)
df = BOSDetector().detect(df)
df = CHOCHDetector().detect(df)

print(
    df[
        [
            "datetime",
            "close",
            "trend",
            "bos",
            "bos_direction",
            "choch",
            "choch_direction",
            "last_choch_direction",
        ]
    ].tail(60)
)
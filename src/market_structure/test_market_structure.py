import pandas as pd

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector
from src.market_structure.bos_detector import BOSDetector
from src.market_structure.choch_detector import CHOCHDetector
from src.market_structure.equal_high_low_detector import EqualHighLowDetector
from src.market_structure.liquidity_detector import LiquidityDetector


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

df = SwingDetector(window=2).detect(df)
df = TrendDetector().detect(df)
df = BOSDetector().detect(df)
df = CHOCHDetector().detect(df)
df = EqualHighLowDetector(tolerance=0.001).detect(df)
df = LiquidityDetector().detect(df)

columns = [
    "datetime",
    "close",
    "high",
    "low",
    "is_swing_high",
    "is_swing_low",
    "last_swing_high",
    "last_swing_low",
    "trend",
    "bos",
    "bos_direction",
    "bos_strength",
    "choch",
    "choch_direction",
    "equal_high",
    "equal_low",
    "liquidity_above",
    "liquidity_below",
    "liquidity_sweep",
    "buy_side_sweep",
    "sell_side_sweep",
    "liquidity_sweep_direction",
    "liquidity_sweep_strength",
]

print(df[columns].tail(60))

print("\nRésumé")
print("=" * 50)
print(f"Swing High : {df['is_swing_high'].sum()}")
print(f"Swing Low  : {df['is_swing_low'].sum()}")
print(f"BOS Bullish : {df['bos_bullish'].sum()}")
print(f"BOS Bearish : {df['bos_bearish'].sum()}")
print(f"CHOCH Bullish : {df['choch_bullish'].sum()}")
print(f"CHOCH Bearish : {df['choch_bearish'].sum()}")
print(f"Equal High : {df['equal_high'].sum()}")
print(f"Equal Low  : {df['equal_low'].sum()}")
print(f"Buy-side Sweep  : {df['buy_side_sweep'].sum()}")
print(f"Sell-side Sweep : {df['sell_side_sweep'].sum()}")

print("\nTest terminé avec succès.")
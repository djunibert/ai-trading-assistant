import pandas as pd

from src.market_structure.market_structure_pipeline import MarketStructurePipeline


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

pipeline = MarketStructurePipeline()
df = pipeline.run(df)

columns = [
    "datetime",
    "close",
    "trend",
    "bos",
    "bos_direction",
    "choch",
    "choch_direction",
    "equal_high",
    "equal_low",
    "liquidity_sweep",
    "fvg",
    "fvg_direction",
    "order_block",
    "order_block_direction",
    "support_price",
    "resistance_price",
    "premium_zone",
    "discount_zone",
    "session",
    "asian_session",
    "london_session",
    "new_york_session",
    "london_new_york_overlap",
]

print(df[columns].tail(80))

print("\nSUMMARY")
print("=" * 70)
print(f"Swing High          : {df['is_swing_high'].sum()}")
print(f"Swing Low           : {df['is_swing_low'].sum()}")
print(f"BOS Bullish         : {df['bos_bullish'].sum()}")
print(f"BOS Bearish         : {df['bos_bearish'].sum()}")
print(f"CHOCH Bullish       : {df['choch_bullish'].sum()}")
print(f"CHOCH Bearish       : {df['choch_bearish'].sum()}")
print(f"Equal High          : {df['equal_high'].sum()}")
print(f"Equal Low           : {df['equal_low'].sum()}")
print(f"Buy Side Sweep      : {df['buy_side_sweep'].sum()}")
print(f"Sell Side Sweep     : {df['sell_side_sweep'].sum()}")
print(f"Bullish FVG         : {df['fvg_bullish'].sum()}")
print(f"Bearish FVG         : {df['fvg_bearish'].sum()}")
print(f"Bullish Order Block : {(df['order_block'] == 1).sum()}")
print(f"Bearish Order Block : {(df['order_block'] == -1).sum()}")
print(f"Asia Session        : {df['asian_session'].sum()}")
print(f"London Session      : {df['london_session'].sum()}")
print(f"New York Session    : {df['new_york_session'].sum()}")
print("=" * 70)
print("\nTest terminé avec succès.")
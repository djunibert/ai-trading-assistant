"""
Test du Market Structure Pipeline complet.
"""

import pandas as pd

from src.market_structure.market_structure_pipeline import MarketStructurePipeline


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

pipeline = MarketStructurePipeline()
df = pipeline.run(df)

columns = [
    "datetime",
    "close",
    "trend",
    "bos_v2",
    "bos_v2_direction",
    "bos_score",
    "choch_v2",
    "choch_v2_direction",
    "choch_score",
    "liquidity_pool_direction",
    "liquidity_pool_score",
    "liquidity_sweep",
    "fvg",
    "fvg_v2_score",
    "ob_v2",
    "ob_v2_direction",
    "ob_v2_score",
    "ob_lifecycle_score",
    "support_price",
    "resistance_price",
    "premium_zone",
    "discount_zone",
    "session",
]

print(df[columns].tail(100))

print("\nSUMMARY")
print("=" * 80)
print(f"Nombre de lignes             : {len(df)}")
print(f"Swing High                   : {df['is_swing_high'].sum()}")
print(f"Swing Low                    : {df['is_swing_low'].sum()}")
print(f"BOS V2 total                 : {(df['bos_v2'] != 0).sum()}")
print(f"BOS score max                : {df['bos_score'].max():.2f}")
print(f"CHOCH V2 total               : {(df['choch_v2'] != 0).sum()}")
print(f"CHOCH score max              : {df['choch_score'].max():.2f}")
print(f"Liquidity Pool active        : {df['liquidity_pool_active'].sum()}")
print(f"Liquidity Pool swept         : {df['liquidity_pool_swept'].sum()}")
print(f"FVG total                    : {(df['fvg'] != 0).sum()}")
print(f"FVG score max                : {df['fvg_v2_score'].max():.2f}")
print(f"Order Block V2 total         : {(df['ob_v2'] != 0).sum()}")
print(f"Order Block lifecycle max    : {df['ob_lifecycle_score'].max():.2f}")
print("=" * 80)
print("Test terminé avec succès.")
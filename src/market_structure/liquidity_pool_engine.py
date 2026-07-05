"""
Test complet du Market Structure Engine.
"""

import pandas as pd

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector
from src.market_structure.bos_detector import BOSDetector
from src.market_structure.bos_engine import BOSEngine
from src.market_structure.choch_detector import CHOCHDetector
from src.market_structure.choch_engine import CHOCHEngine
from src.market_structure.equal_high_low_detector import EqualHighLowDetector
from src.market_structure.liquidity_pool_engine import LiquidityPoolEngine
from src.market_structure.liquidity_detector import LiquidityDetector
from src.market_structure.fair_value_gap_detector import FairValueGapDetector
from src.market_structure.fair_value_gap_engine import FairValueGapEngine
from src.market_structure.order_block_detector import OrderBlockDetector
from src.market_structure.order_block_engine import OrderBlockEngine
from src.market_structure.order_block_lifecycle_engine import OrderBlockLifecycleEngine
from src.market_structure.support_resistance_detector import SupportResistanceDetector
from src.market_structure.premium_discount_detector import PremiumDiscountDetector
from src.market_structure.session_detector import SessionDetector


# Charger les données Gold 1H
df = pd.read_csv("data/features/market_macro/1h/gold.csv")

# Exécuter les détecteurs dans le bon ordre
df = SwingDetector(window=2).detect(df)
df = TrendDetector().detect(df)

df = BOSDetector().detect(df)
df = BOSEngine().detect(df)

df = CHOCHDetector().detect(df)

df = EqualHighLowDetector(tolerance=0.001).detect(df)
df = LiquidityPoolEngine().detect(df)
df = LiquidityDetector().detect(df)

df = CHOCHEngine().detect(df)

df = FairValueGapDetector().detect(df)
df = FairValueGapEngine().detect(df)

df = OrderBlockDetector(lookback=5).detect(df)
df = OrderBlockEngine(lookback=5).detect(df)
df = OrderBlockLifecycleEngine().detect(df)

df = SupportResistanceDetector().detect(df)
df = PremiumDiscountDetector(equilibrium_tolerance=0.001).detect(df)
df = SessionDetector().detect(df)


columns = [
    "datetime",
    "close",

    # Trend / BOS / CHOCH
    "trend",
    "bos_v2",
    "bos_v2_direction",
    "bos_strength_atr",
    "bos_score",
    "choch_v2",
    "choch_v2_direction",
    "choch_score",

    # Equal High / Low
    "equal_high",
    "equal_low",

    # Liquidity Pool
    "liquidity_pool_direction",
    "liquidity_pool_price",
    "liquidity_pool_strength",
    "liquidity_pool_age",
    "liquidity_pool_distance",
    "liquidity_pool_swept",
    "liquidity_pool_active",
    "liquidity_pool_score",

    # Liquidity Sweep
    "liquidity_sweep",
    "buy_side_sweep",
    "sell_side_sweep",

    # FVG
    "fvg",
    "fvg_direction",
    "fvg_v2_active",
    "fvg_v2_fill_percent",
    "fvg_v2_score",

    # Order Block
    "ob_v2",
    "ob_v2_direction",
    "ob_v2_score",
    "ob_lifecycle_active",
    "ob_lifecycle_age",
    "ob_lifecycle_touch_count",
    "ob_lifecycle_retested",
    "ob_lifecycle_mitigated",
    "ob_lifecycle_invalidated",
    "ob_lifecycle_score",

    # Support / Resistance
    "support_price",
    "resistance_price",

    # Premium / Discount
    "premium_zone",
    "discount_zone",

    # Session
    "session",
]

print(df[columns].tail(100))

print("\nSUMMARY")
print("=" * 80)
print(f"Swing High                 : {df['is_swing_high'].sum()}")
print(f"Swing Low                  : {df['is_swing_low'].sum()}")

print(f"BOS V2 total               : {(df['bos_v2'] != 0).sum()}")
print(f"BOS V2 score max           : {df['bos_score'].max():.2f}")

print(f"CHOCH V2 total             : {(df['choch_v2'] != 0).sum()}")
print(f"CHOCH V2 score max         : {df['choch_score'].max():.2f}")

print(f"Liquidity Pool active      : {df['liquidity_pool_active'].sum()}")
print(f"Liquidity Pool swept       : {df['liquidity_pool_swept'].sum()}")
print(f"Liquidity Pool score max   : {df['liquidity_pool_score'].max():.2f}")

print(f"Buy Side Sweep             : {df['buy_side_sweep'].sum()}")
print(f"Sell Side Sweep            : {df['sell_side_sweep'].sum()}")

print(f"FVG total                  : {(df['fvg'] != 0).sum()}")
print(f"FVG V2 active              : {df['fvg_v2_active'].sum()}")
print(f"FVG V2 score max           : {df['fvg_v2_score'].max():.2f}")

print(f"Order Block V2 total       : {(df['ob_v2'] != 0).sum()}")
print(f"OB Lifecycle active        : {df['ob_lifecycle_active'].sum()}")
print(f"OB Lifecycle retested      : {df['ob_lifecycle_retested'].sum()}")
print(f"OB Lifecycle mitigated     : {df['ob_lifecycle_mitigated'].sum()}")
print(f"OB Lifecycle invalidated   : {df['ob_lifecycle_invalidated'].sum()}")
print(f"OB Lifecycle score max     : {df['ob_lifecycle_score'].max():.2f}")
print("=" * 80)

print("\nTest terminé avec succès.")
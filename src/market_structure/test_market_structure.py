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
from src.market_structure.liquidity_detector import LiquidityDetector
from src.market_structure.fair_value_gap_detector import FairValueGapDetector
from src.market_structure.order_block_detector import OrderBlockDetector
from src.market_structure.order_block_engine import OrderBlockEngine
from src.market_structure.order_block_lifecycle_engine import OrderBlockLifecycleEngine
from src.market_structure.support_resistance_detector import SupportResistanceDetector
from src.market_structure.premium_discount_detector import PremiumDiscountDetector
from src.market_structure.session_detector import SessionDetector


df = pd.read_csv("data/features/market_macro/1h/gold.csv")

df = SwingDetector(window=2).detect(df)
df = TrendDetector().detect(df)
df = BOSDetector().detect(df)
df = BOSEngine().detect(df)
df = CHOCHDetector().detect(df)
df = EqualHighLowDetector(tolerance=0.001).detect(df)
df = LiquidityDetector().detect(df)
df = CHOCHEngine().detect(df)
df = FairValueGapDetector().detect(df)
df = OrderBlockDetector(lookback=5).detect(df)
df = OrderBlockEngine(lookback=5).detect(df)
df = OrderBlockLifecycleEngine().detect(df)
df = SupportResistanceDetector().detect(df)
df = PremiumDiscountDetector(equilibrium_tolerance=0.001).detect(df)
df = SessionDetector().detect(df)

columns = [
    "datetime",
    "close",

    "trend",

    "bos",
    "bos_direction",
    "bos_v2",
    "bos_v2_direction",
    "bos_strength_atr",
    "bos_score",

    "choch",
    "choch_direction",
    "choch_v2",
    "choch_v2_direction",
    "choch_score",

    "equal_high",
    "equal_low",

    "liquidity_sweep",
    "buy_side_sweep",
    "sell_side_sweep",

    "fvg",
    "fvg_direction",
    "fvg_size",

    "order_block",
    "order_block_direction",

    "ob_v2",
    "ob_v2_direction",
    "ob_v2_top",
    "ob_v2_bottom",
    "ob_v2_score",

    "ob_lifecycle_active",
    "ob_lifecycle_age",
    "ob_lifecycle_touch_count",
    "ob_lifecycle_retested",
    "ob_lifecycle_mitigated",
    "ob_lifecycle_invalidated",
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
print(f"Swing High                 : {df['is_swing_high'].sum()}")
print(f"Swing Low                  : {df['is_swing_low'].sum()}")
print(f"BOS V2 total               : {(df['bos_v2'] != 0).sum()}")
print(f"BOS V2 score max           : {df['bos_score'].max():.2f}")
print(f"CHOCH V2 total             : {(df['choch_v2'] != 0).sum()}")
print(f"CHOCH V2 score max         : {df['choch_score'].max():.2f}")
print(f"FVG total                  : {(df['fvg'] != 0).sum()}")
print(f"Order Block V2 total       : {(df['ob_v2'] != 0).sum()}")
print(f"OB Lifecycle active        : {df['ob_lifecycle_active'].sum()}")
print(f"OB Lifecycle retested      : {df['ob_lifecycle_retested'].sum()}")
print(f"OB Lifecycle mitigated     : {df['ob_lifecycle_mitigated'].sum()}")
print(f"OB Lifecycle invalidated   : {df['ob_lifecycle_invalidated'].sum()}")
print(f"OB Lifecycle score max     : {df['ob_lifecycle_score'].max():.2f}")
print("=" * 80)

print("\nTest terminé avec succès.")
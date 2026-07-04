"""
Test complet du Market Structure Engine.

Projet : AI Trading System
Auteur : Junior Hébert
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
from src.market_structure.support_resistance_detector import SupportResistanceDetector
from src.market_structure.premium_discount_detector import PremiumDiscountDetector
from src.market_structure.session_detector import SessionDetector


print("=" * 90)
print("MARKET STRUCTURE ENGINE V2")
print("=" * 90)

# ============================================================
# Chargement des données
# ============================================================

df = pd.read_csv("data/features/market_macro/1h/gold.csv")

print(f"\nNombre de bougies : {len(df)}")

# ============================================================
# Détecteurs V1
# ============================================================

print("\n[1] Swing Detector")
df = SwingDetector(window=2).detect(df)

print("[2] Trend Detector")
df = TrendDetector().detect(df)

print("[3] BOS Detector V1")
df = BOSDetector().detect(df)

print("[4] CHOCH Detector V1")
df = CHOCHDetector().detect(df)

print("[5] Equal High / Low")
df = EqualHighLowDetector(tolerance=0.001).detect(df)

print("[6] Liquidity")
df = LiquidityDetector().detect(df)

print("[7] Fair Value Gap")
df = FairValueGapDetector().detect(df)

print("[8] Order Block")
df = OrderBlockDetector(lookback=5).detect(df)

print("[9] Support / Resistance")
df = SupportResistanceDetector().detect(df)

print("[10] Premium / Discount")
df = PremiumDiscountDetector(
    equilibrium_tolerance=0.001
).detect(df)

print("[11] Sessions")
df = SessionDetector().detect(df)

# ============================================================
# Engines V2
# ============================================================

print("[12] BOS Engine V2")
df = BOSEngine().detect(df)

print("[13] CHOCH Engine V2")
df = CHOCHEngine().detect(df)

# ============================================================
# Colonnes
# ============================================================

columns = [

    "datetime",

    "close",

    "trend",

    "bos",
    "bos_direction",

    "bos_v2",
    "bos_v2_direction",

    "bos_strength_points",
    "bos_strength_atr",
    "bos_volume_ratio",
    "bos_confirmed",
    "bos_score",

    "choch",
    "choch_direction",

    "choch_v2",
    "choch_v2_direction",
    "choch_strength_atr",
    "choch_confirmed",
    "choch_after_liquidity",
    "choch_score",

    "equal_high",
    "equal_low",

    "buy_side_sweep",
    "sell_side_sweep",

    "fvg",
    "fvg_direction",
    "fvg_size",

    "order_block",
    "order_block_direction",

    "support_price",
    "resistance_price",

    "premium_zone",
    "discount_zone",

    "session"

]

print("\n")
print(df[columns].tail(100))

# ============================================================
# Statistiques
# ============================================================

print("\n")
print("=" * 90)
print("STATISTIQUES")
print("=" * 90)

print(f"Swing High                 : {df['is_swing_high'].sum()}")
print(f"Swing Low                  : {df['is_swing_low'].sum()}")

print()

print(f"BOS Bullish                : {df['bos_bullish'].sum()}")
print(f"BOS Bearish                : {df['bos_bearish'].sum()}")

print(f"BOS V2                     : {(df['bos_v2'] != 0).sum()}")
print(f"Average BOS Score          : {df['bos_score'].mean():.2f}")
print(f"Maximum BOS Score          : {df['bos_score'].max():.2f}")

print()

print(f"CHOCH Bullish              : {df['choch_bullish'].sum()}")
print(f"CHOCH Bearish              : {df['choch_bearish'].sum()}")

print(f"CHOCH V2                   : {(df['choch_v2'] != 0).sum()}")
print(f"Average CHOCH Score        : {df['choch_score'].mean():.2f}")

print()

print(f"Equal High                 : {df['equal_high'].sum()}")
print(f"Equal Low                  : {df['equal_low'].sum()}")

print()

print(f"Buy Side Sweep             : {df['buy_side_sweep'].sum()}")
print(f"Sell Side Sweep            : {df['sell_side_sweep'].sum()}")

print()

print(f"Bullish FVG                : {df['fvg_bullish'].sum()}")
print(f"Bearish FVG                : {df['fvg_bearish'].sum()}")

print()

print(f"Bullish Order Block        : {(df['order_block']==1).sum()}")
print(f"Bearish Order Block        : {(df['order_block']==-1).sum()}")

print()

print(f"Near Support               : {df['near_support'].sum()}")
print(f"Near Resistance            : {df['near_resistance'].sum()}")

print()

print(f"Premium Zone               : {df['premium_zone'].sum()}")
print(f"Discount Zone              : {df['discount_zone'].sum()}")
print(f"Equilibrium Zone           : {df['equilibrium_zone'].sum()}")

print()

print(f"Asian Session              : {df['asian_session'].sum()}")
print(f"London Session             : {df['london_session'].sum()}")
print(f"New York Session           : {df['new_york_session'].sum()}")
print(f"London/NewYork Overlap     : {df['london_new_york_overlap'].sum()}")

print("=" * 90)

print("\nTest terminé avec succès.")
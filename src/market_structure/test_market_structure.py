"""
Test complet du Market Structure Engine.

Ce fichier sert à vérifier que tous les détecteurs de structure
de marché fonctionnent correctement sur les données Gold 1H.

Modules testés :
- Swing High / Swing Low
- Trend
- BOS V1
- BOS V2
- CHOCH V1
- CHOCH V2
- Equal High / Equal Low
- Liquidity Sweep
- Fair Value Gap
- Order Block V1
- Order Block V2
- Support / Resistance
- Premium / Discount
- Sessions
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
from src.market_structure.support_resistance_detector import SupportResistanceDetector
from src.market_structure.premium_discount_detector import PremiumDiscountDetector
from src.market_structure.session_detector import SessionDetector


# ============================================================
# 1. Charger les données
# ============================================================

df = pd.read_csv("data/features/market_macro/1h/gold.csv")

print("=" * 90)
print("MARKET STRUCTURE ENGINE TEST")
print("=" * 90)
print(f"Nombre de bougies chargées : {len(df)}")


# ============================================================
# 2. Exécuter les détecteurs dans le bon ordre
# ============================================================

# Les swings sont la base de toute la structure du marché.
df = SwingDetector(window=2).detect(df)

# La tendance utilise les Swing High et Swing Low.
df = TrendDetector().detect(df)

# BOS V1 : première version simple du Break Of Structure.
df = BOSDetector().detect(df)

# BOS V2 : version avancée avec force, score, ATR, volume.
df = BOSEngine().detect(df)

# CHOCH V1 : première version simple du Change Of Character.
df = CHOCHDetector().detect(df)

# Equal High / Equal Low : zones de liquidité potentielles.
df = EqualHighLowDetector(tolerance=0.001).detect(df)

# Liquidity Sweep : détection des pièges au-dessus ou sous les swings.
df = LiquidityDetector().detect(df)

# CHOCH V2 : utilise notamment la liquidité récente.
df = CHOCHEngine().detect(df)

# Fair Value Gap : déséquilibre entre trois bougies.
df = FairValueGapDetector().detect(df)

# Order Block V1 : dernière bougie opposée avant un BOS.
df = OrderBlockDetector(lookback=5).detect(df)

# Order Block V2 : zones, mitigation, confluences, score.
df = OrderBlockEngine(lookback=5).detect(df)

# Support / Resistance : basé sur les derniers swings.
df = SupportResistanceDetector().detect(df)

# Premium / Discount : basé sur la zone entre swing high et swing low.
df = PremiumDiscountDetector(equilibrium_tolerance=0.001).detect(df)

# Sessions : Asie, Londres, New York, overlap.
df = SessionDetector().detect(df)


# ============================================================
# 3. Colonnes principales à afficher
# ============================================================

columns = [
    "datetime",
    "close",

    # Swing
    "is_swing_high",
    "is_swing_low",
    "last_swing_high",
    "last_swing_low",

    # Trend
    "trend",

    # BOS V1
    "bos",
    "bos_direction",

    # BOS V2
    "bos_v2",
    "bos_v2_direction",
    "bos_price",
    "bos_strength_points",
    "bos_strength_atr",
    "bos_volume_ratio",
    "bos_confirmed",
    "bos_score",

    # CHOCH V1
    "choch",
    "choch_direction",

    # CHOCH V2
    "choch_v2",
    "choch_v2_direction",
    "choch_strength_atr",
    "choch_confirmed",
    "choch_after_liquidity",
    "choch_score",

    # Equal High / Equal Low
    "equal_high",
    "equal_low",

    # Liquidity
    "liquidity_sweep",
    "buy_side_sweep",
    "sell_side_sweep",
    "liquidity_sweep_direction",
    "liquidity_sweep_strength",

    # FVG
    "fvg",
    "fvg_direction",
    "fvg_top",
    "fvg_bottom",
    "fvg_size",

    # Order Block V1
    "order_block",
    "order_block_direction",

    # Order Block V2
    "ob_v2",
    "ob_v2_direction",
    "ob_v2_top",
    "ob_v2_bottom",
    "ob_v2_size",
    "ob_v2_size_atr",
    "ob_v2_mitigated",
    "ob_v2_confluence_bos",
    "ob_v2_confluence_fvg",
    "ob_v2_confluence_liquidity",
    "ob_v2_score",

    # Support / Resistance
    "support_price",
    "resistance_price",
    "distance_to_support",
    "distance_to_resistance",
    "near_support",
    "near_resistance",

    # Premium / Discount
    "equilibrium_price",
    "premium_zone",
    "discount_zone",
    "equilibrium_zone",
    "distance_to_equilibrium",

    # Sessions
    "session",
    "asian_session",
    "london_session",
    "new_york_session",
    "london_new_york_overlap",
]


# ============================================================
# 4. Afficher les dernières lignes
# ============================================================

print("\nDernières lignes avec Market Structure :")
print("=" * 90)

print(df[columns].tail(100))


# ============================================================
# 5. Résumé statistique
# ============================================================

print("\nSUMMARY")
print("=" * 90)

print(f"Swing High                 : {df['is_swing_high'].sum()}")
print(f"Swing Low                  : {df['is_swing_low'].sum()}")

print(f"BOS Bullish V1             : {df['bos_bullish'].sum()}")
print(f"BOS Bearish V1             : {df['bos_bearish'].sum()}")
print(f"BOS V2 total               : {(df['bos_v2'] != 0).sum()}")
print(f"BOS V2 score moyen         : {df['bos_score'].mean():.2f}")
print(f"BOS V2 score max           : {df['bos_score'].max():.2f}")

print(f"CHOCH Bullish V1           : {df['choch_bullish'].sum()}")
print(f"CHOCH Bearish V1           : {df['choch_bearish'].sum()}")
print(f"CHOCH V2 total             : {(df['choch_v2'] != 0).sum()}")
print(f"CHOCH V2 score moyen       : {df['choch_score'].mean():.2f}")
print(f"CHOCH V2 score max         : {df['choch_score'].max():.2f}")

print(f"Equal High                 : {df['equal_high'].sum()}")
print(f"Equal Low                  : {df['equal_low'].sum()}")

print(f"Buy Side Sweep             : {df['buy_side_sweep'].sum()}")
print(f"Sell Side Sweep            : {df['sell_side_sweep'].sum()}")

print(f"Bullish FVG                : {df['fvg_bullish'].sum()}")
print(f"Bearish FVG                : {df['fvg_bearish'].sum()}")

print(f"Order Block Bullish V1     : {(df['order_block'] == 1).sum()}")
print(f"Order Block Bearish V1     : {(df['order_block'] == -1).sum()}")
print(f"Order Block V2 total       : {(df['ob_v2'] != 0).sum()}")
print(f"Order Block V2 score moyen : {df['ob_v2_score'].mean():.2f}")
print(f"Order Block V2 score max   : {df['ob_v2_score'].max():.2f}")

print(f"Near Support               : {df['near_support'].sum()}")
print(f"Near Resistance            : {df['near_resistance'].sum()}")

print(f"Premium Zone               : {df['premium_zone'].sum()}")
print(f"Discount Zone              : {df['discount_zone'].sum()}")
print(f"Equilibrium Zone           : {df['equilibrium_zone'].sum()}")

print(f"Asian Session              : {df['asian_session'].sum()}")
print(f"London Session             : {df['london_session'].sum()}")
print(f"New York Session           : {df['new_york_session'].sum()}")
print(f"London/New York Overlap    : {df['london_new_york_overlap'].sum()}")

print("=" * 90)
print("Test terminé avec succès.")
"""
Configuration du moteur Market Structure.

Toutes les constantes du projet sont regroupées ici.

Ainsi, si on veut modifier un paramètre,
il suffit de changer cette valeur.
"""

# ==========================================================
# Swing
# ==========================================================

# Nombre de bougies à gauche et à droite
# pour détecter un Swing High / Swing Low.
SWING_WINDOW = 2


# ==========================================================
# Equal High / Equal Low
# ==========================================================

# Tolérance basée sur l'ATR.
# 0.10 signifie 10 % de l'ATR.
EQUAL_HIGH_LOW_ATR_MULTIPLIER = 0.10


# ==========================================================
# BOS
# ==========================================================

# BOS confirmé à partir de cette valeur.
BOS_CONFIRMATION_ATR = 0.25

# BOS fort.
BOS_STRONG_ATR = 0.50

# BOS très fort.
BOS_VERY_STRONG_ATR = 1.00


# ==========================================================
# CHOCH
# ==========================================================

CHOCH_CONFIRMATION_ATR = 0.25


# ==========================================================
# Order Block
# ==========================================================

# Nombre de bougies regardées
# avant le BOS.
ORDER_BLOCK_LOOKBACK = 5


# ==========================================================
# Fair Value Gap
# ==========================================================

# Taille minimale du gap
# exprimée en ATR.
FVG_MIN_ATR = 0.20


# ==========================================================
# Liquidity Pool
# ==========================================================

# Nombre minimum de Swing identiques
# pour créer une zone de liquidité.
LIQUIDITY_MIN_STRENGTH = 2


# ==========================================================
# Sessions
# ==========================================================

LONDON_START = 3
LONDON_END = 11

NEW_YORK_START = 8
NEW_YORK_END = 16

ASIA_START = 18
ASIA_END = 2


# ==========================================================
# Market Structure Score
# ==========================================================

BOS_WEIGHT = 0.20

CHOCH_WEIGHT = 0.15

LIQUIDITY_WEIGHT = 0.15

FVG_WEIGHT = 0.15

ORDER_BLOCK_WEIGHT = 0.15

ORDER_BLOCK_LIFECYCLE_WEIGHT = 0.10


# ==========================================================
# Trade Setup
# ==========================================================

BUY_SETUP_THRESHOLD = 80

SELL_SETUP_THRESHOLD = 80


# ==========================================================
# Risk Management
# ==========================================================

DEFAULT_RISK_PERCENT = 1.0

DEFAULT_RISK_REWARD = 2.0
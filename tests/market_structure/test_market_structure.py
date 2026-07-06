"""
====================================================================
TEST - MARKET STRUCTURE PIPELINE
====================================================================

Auteur :
    Junior Hébert

Description :
    Teste le pipeline complet de Structure de Marché.

Le pipeline exécute :

    - Swing
    - Trend
    - BOS
    - CHOCH
    - Equal High / Equal Low
    - Liquidity
    - Fair Value Gap
    - Order Block
    - Support / Resistance
    - Premium / Discount
    - Sessions
    - Market Structure Score
    - Trade Setup
    - Risk Engine
    - Stop Loss Engine
    - Take Profit Engine

====================================================================
"""

import pandas as pd

from src.pipelines.market_structure_pipeline import (
    MarketStructurePipeline,
)


# ==========================================================
# Chargement des données
# ==========================================================

DATA_PATH = "data/features/market_macro/1h/gold.csv"

df = pd.read_csv(DATA_PATH)

print("\n")
print("=" * 90)
print("CHARGEMENT DES DONNÉES")
print("=" * 90)
print(df.head())

# ==========================================================
# Exécution du pipeline
# ==========================================================

pipeline = MarketStructurePipeline()

df = pipeline.run(df)

# ==========================================================
# Colonnes importantes à afficher
# ==========================================================

columns = [

    # -----------------------------
    # Prix
    # -----------------------------

    "datetime",
    "close",

    # -----------------------------
    # Trend
    # -----------------------------

    "trend",

    # -----------------------------
    # BOS
    # -----------------------------

    "bos_v2",
    "bos_v2_direction",
    "bos_score",

    # -----------------------------
    # CHOCH
    # -----------------------------

    "choch_v2",
    "choch_v2_direction",
    "choch_score",

    # -----------------------------
    # Liquidité
    # -----------------------------

    "equal_high",
    "equal_low",

    "liquidity_pool_direction",
    "liquidity_pool_score",

    "liquidity_sweep",

    # -----------------------------
    # Fair Value Gap
    # -----------------------------

    "fvg",
    "fvg_direction",
    "fvg_v2_score",

    # -----------------------------
    # Order Block
    # -----------------------------

    "ob_v2",
    "ob_v2_direction",
    "ob_v2_score",

    "ob_lifecycle_score",

    # -----------------------------
    # Premium / Discount
    # -----------------------------

    "premium_zone",
    "discount_zone",

    # -----------------------------
    # Sessions
    # -----------------------------

    "session",

    # -----------------------------
    # Score global
    # -----------------------------

    "market_structure_score",

    # -----------------------------
    # Trade Setup
    # -----------------------------

    "buy_setup",

    "sell_setup",

    "buy_setup_score",

    "sell_setup_score",

    "trade_setup",

    # -----------------------------
    # Risk
    # -----------------------------

    "risk_signal",

    "risk_trade_allowed",

    "risk_entry_price",

    "sl_price",

    "tp_price",

    "sl_distance",

    "tp_distance",

    "risk_reward_ratio",

]

print("\n")
print("=" * 90)
print("DERNIÈRES BOUGIES")
print("=" * 90)

print(df[columns].tail(100))

# ==========================================================
# Résumé
# ==========================================================

print("\n")
print("=" * 90)
print("RÉSUMÉ")
print("=" * 90)

print(f"Nombre de lignes                : {len(df)}")

print(f"Swing High                      : {df['is_swing_high'].sum()}")

print(f"Swing Low                       : {df['is_swing_low'].sum()}")

print(f"BOS détectés                    : {(df['bos_v2'] != 0).sum()}")

print(f"CHOCH détectés                  : {(df['choch_v2'] != 0).sum()}")

print(f"Liquidity Pool                  : {df['liquidity_pool_active'].sum()}")

print(f"Liquidity Sweep                 : {df['liquidity_sweep'].sum()}")

print(f"Fair Value Gap                  : {(df['fvg'] != 0).sum()}")

print(f"Order Block                     : {(df['ob_v2'] != 0).sum()}")

print(f"BUY Setup                       : {df['buy_setup'].sum()}")

print(f"SELL Setup                      : {df['sell_setup'].sum()}")

print(f"Trades autorisés                : {df['risk_trade_allowed'].sum()}")

print(f"Score maximum                   : {df['market_structure_score'].max():.2f}")

print(f"Stop Loss calculés              : {(df['sl_price'] > 0).sum()}")

print(f"Take Profit calculés            : {(df['tp_price'] > 0).sum()}")

print("=" * 90)

# ==========================================================
# Sauvegarde
# ==========================================================

OUTPUT_PATH = "reports/market_structure_result.csv"

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n")
print("=" * 90)
print("RÉSULTAT SAUVEGARDÉ")
print("=" * 90)

print(f"Fichier : {OUTPUT_PATH}")

print("\n")
print("=" * 90)
print("TEST TERMINÉ AVEC SUCCÈS")
print("=" * 90)
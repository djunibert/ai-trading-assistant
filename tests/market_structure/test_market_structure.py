"""
Test du pipeline complet de structure de marché.
"""

import pandas as pd

from src.pipelines.market_structure_pipeline import MarketStructurePipeline


def main() -> None:
    # Charger les données de test
    df = pd.read_csv("data/features/market_macro/1h/gold.csv")

    # Exécuter le pipeline complet
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
        "fvg_direction",
        "fvg_v2_score",

        "ob_v2",
        "ob_v2_direction",
        "ob_v2_score",
        "ob_lifecycle_score",

        "market_structure_score",

        "buy_setup_score",
        "sell_setup_score",
        "trade_setup",

        "risk_signal",
        "risk_trade_allowed",
        "risk_entry_price",
        "risk_stop_loss",
        "risk_take_profit",
        "risk_stop_distance",
        "risk_reward_ratio",
    ]

    print(df[columns].tail(100))

    print("\nSUMMARY")
    print("=" * 80)
    print(f"Nombre de lignes              : {len(df)}")
    print(f"Swing High                    : {df['is_swing_high'].sum()}")
    print(f"Swing Low                     : {df['is_swing_low'].sum()}")
    print(f"BOS V2 total                  : {(df['bos_v2'] != 0).sum()}")
    print(f"BOS score max                 : {df['bos_score'].max():.2f}")
    print(f"CHOCH V2 total                : {(df['choch_v2'] != 0).sum()}")
    print(f"CHOCH score max               : {df['choch_score'].max():.2f}")
    print(f"Liquidity Pool active         : {df['liquidity_pool_active'].sum()}")
    print(f"Liquidity Pool swept          : {df['liquidity_pool_swept'].sum()}")
    print(f"FVG total                     : {(df['fvg'] != 0).sum()}")
    print(f"FVG score max                 : {df['fvg_v2_score'].max():.2f}")
    print(f"Order Block V2 total          : {(df['ob_v2'] != 0).sum()}")
    print(f"Order Block lifecycle max     : {df['ob_lifecycle_score'].max():.2f}")
    print(f"Market Structure score max    : {df['market_structure_score'].max():.2f}")
    print(f"BUY setup                     : {df['buy_setup'].sum()}")
    print(f"SELL setup                    : {df['sell_setup'].sum()}")
    print(f"Trades autorisés              : {df['risk_trade_allowed'].sum()}")
    print("=" * 80)

    print("\nTest terminé avec succès.")


if __name__ == "__main__":
    main()
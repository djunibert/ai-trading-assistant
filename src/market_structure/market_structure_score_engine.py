"""
Market Structure Score Engine.

Calcule un score global de qualité de la structure de marché
en utilisant les poids définis dans le fichier de configuration.
"""

import pandas as pd

from src.config.market_structure_config import (
    BOS_WEIGHT,
    CHOCH_WEIGHT,
    LIQUIDITY_WEIGHT,
    FVG_WEIGHT,
    ORDER_BLOCK_WEIGHT,
    ORDER_BLOCK_LIFECYCLE_WEIGHT,
)


class MarketStructureScoreEngine:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["market_structure_score"] = 0.0

        if "bos_score" in df.columns:
            df["market_structure_score"] += df["bos_score"] * BOS_WEIGHT

        if "choch_score" in df.columns:
            df["market_structure_score"] += df["choch_score"] * CHOCH_WEIGHT

        if "liquidity_pool_score" in df.columns:
            df["market_structure_score"] += (
                df["liquidity_pool_score"] * LIQUIDITY_WEIGHT
            )

        if "fvg_v2_score" in df.columns:
            df["market_structure_score"] += df["fvg_v2_score"] * FVG_WEIGHT

        if "ob_v2_score" in df.columns:
            df["market_structure_score"] += df["ob_v2_score"] * ORDER_BLOCK_WEIGHT

        if "ob_lifecycle_score" in df.columns:
            df["market_structure_score"] += (
                df["ob_lifecycle_score"] * ORDER_BLOCK_LIFECYCLE_WEIGHT
            )

        if "london_session" in df.columns:
            df["market_structure_score"] += df["london_session"].astype(int) * 5

        if "new_york_session" in df.columns:
            df["market_structure_score"] += df["new_york_session"].astype(int) * 5

        if "london_new_york_overlap" in df.columns:
            df["market_structure_score"] += (
                df["london_new_york_overlap"].astype(int) * 10
            )

        df["market_structure_score"] = (
            df["market_structure_score"]
            .clip(lower=0, upper=100)
            .round(2)
        )

        return df
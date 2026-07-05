"""
=====================================================================
MARKET STRUCTURE PIPELINE
=====================================================================

Ce pipeline exécute l'ensemble des détecteurs et moteurs
de Smart Money Concepts dans le bon ordre.

Projet :
    AI Trading System

Auteur :
    Junior Hébert

=====================================================================
"""

from __future__ import annotations

import pandas as pd

# ==========================================================
# Configuration du projet
# ==========================================================

from src.config.market_structure_config import (
    SWING_WINDOW,
    EQUAL_HIGH_LOW_ATR_MULTIPLIER,
    ORDER_BLOCK_LOOKBACK,
)

# ==========================================================
# Structure du marché
# ==========================================================

from src.market_structure.swing_detector import SwingDetector
from src.market_structure.trend_detector import TrendDetector

# ==========================================================
# Break Of Structure
# ==========================================================

from src.market_structure.bos_detector import BOSDetector
from src.market_structure.bos_engine import BOSEngine

# ==========================================================
# Change Of Character
# ==========================================================

from src.market_structure.choch_detector import CHOCHDetector
from src.market_structure.choch_engine import CHOCHEngine

# ==========================================================
# Equal High / Equal Low
# ==========================================================

from src.market_structure.equal_high_low_detector import (
    EqualHighLowDetector,
)

# ==========================================================
# Liquidité
# ==========================================================

from src.market_structure.liquidity_pool_engine import (
    LiquidityPoolEngine,
)

from src.market_structure.liquidity_detector import (
    LiquidityDetector,
)

# ==========================================================
# Fair Value Gap
# ==========================================================

from src.market_structure.fair_value_gap_detector import (
    FairValueGapDetector,
)

from src.market_structure.fair_value_gap_engine import (
    FairValueGapEngine,
)

# ==========================================================
# Order Block
# ==========================================================

from src.market_structure.order_block_detector import (
    OrderBlockDetector,
)

from src.market_structure.order_block_engine import (
    OrderBlockEngine,
)

from src.market_structure.order_block_lifecycle_engine import (
    OrderBlockLifecycleEngine,
)

# ==========================================================
# Support / Resistance
# ==========================================================

from src.market_structure.support_resistance_detector import (
    SupportResistanceDetector,
)

from src.market_structure.premium_discount_detector import (
    PremiumDiscountDetector,
)

# ==========================================================
# Sessions de marché
# ==========================================================

from src.market_structure.session_detector import (
    SessionDetector,
)

# ==========================================================
# Score global
# ==========================================================

from src.market_structure.market_structure_score_engine import (
    MarketStructureScoreEngine,
)

# ==========================================================
# Trade Setup
# ==========================================================

from src.market_structure.trade_setup_engine import (
    TradeSetupEngine,
)


class MarketStructurePipeline:
    """
    Pipeline principal du moteur Smart Money Concepts.

    Chaque moteur ajoute de nouvelles variables
    dans le DataFrame.

    L'ordre est IMPORTANT car plusieurs moteurs
    dépendent des résultats des précédents.
    """

    def __init__(self):

        self.engines = [

            # ==================================================
            # 1. Swing High / Swing Low
            # ==================================================

            SwingDetector(
                window=SWING_WINDOW
            ),

            # ==================================================
            # 2. Tendance
            # ==================================================

            TrendDetector(),

            # ==================================================
            # 3. Break Of Structure
            # ==================================================

            BOSDetector(),

            BOSEngine(),

            # ==================================================
            # 4. Change Of Character
            # ==================================================

            CHOCHDetector(),

            CHOCHEngine(),

            # ==================================================
            # 5. Equal High / Equal Low
            # ==================================================

            EqualHighLowDetector(
                atr_multiplier=EQUAL_HIGH_LOW_ATR_MULTIPLIER
            ),

            # ==================================================
            # 6. Liquidité
            # ==================================================

            LiquidityPoolEngine(),

            LiquidityDetector(),

            # ==================================================
            # 7. Fair Value Gap
            # ==================================================

            FairValueGapDetector(),

            FairValueGapEngine(),

            # ==================================================
            # 8. Order Block
            # ==================================================

            OrderBlockDetector(
                lookback=ORDER_BLOCK_LOOKBACK
            ),

            OrderBlockEngine(
                lookback=ORDER_BLOCK_LOOKBACK
            ),

            OrderBlockLifecycleEngine(),

            # ==================================================
            # 9. Support / Resistance
            # ==================================================

            SupportResistanceDetector(),

            PremiumDiscountDetector(),

            # ==================================================
            # 10. Sessions
            # ==================================================

            SessionDetector(),

            # ==================================================
            # 11. Score global
            # ==================================================

            MarketStructureScoreEngine(),

            # ==================================================
            # 12. Trade Setup
            # ==================================================

            TradeSetupEngine(),

        ]

    # ==========================================================
    # Exécution complète
    # ==========================================================

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute tout le pipeline.

        Parameters
        ----------
        df : DataFrame

        Returns
        -------
        DataFrame enrichi
        """

        df = df.copy()

        for engine in self.engines:
            df = engine.detect(df)

        return df
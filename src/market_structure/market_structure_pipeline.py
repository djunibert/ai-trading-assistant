"""
Market Structure Pipeline
"""

from __future__ import annotations

import pandas as pd

from src.config.market_structure_config import (
    SWING_WINDOW,
    EQUAL_HIGH_LOW_ATR_MULTIPLIER,
    ORDER_BLOCK_LOOKBACK,
)

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
from src.market_structure.market_structure_score_engine import MarketStructureScoreEngine


class MarketStructurePipeline:
    """
    Exécute tous les détecteurs dans le bon ordre.
    """

    def __init__(self):
        self.engines = [
            SwingDetector(window=SWING_WINDOW),
            TrendDetector(),

            BOSDetector(),
            BOSEngine(),

            CHOCHDetector(),
            CHOCHEngine(),

            EqualHighLowDetector(
                atr_multiplier=EQUAL_HIGH_LOW_ATR_MULTIPLIER
            ),

            LiquidityPoolEngine(),
            LiquidityDetector(),

            FairValueGapDetector(),
            FairValueGapEngine(),

            OrderBlockDetector(
                lookback=ORDER_BLOCK_LOOKBACK
            ),

            OrderBlockEngine(
                lookback=ORDER_BLOCK_LOOKBACK
            ),

            OrderBlockLifecycleEngine(),

            SupportResistanceDetector(),
            PremiumDiscountDetector(),

            SessionDetector(),
            MarketStructureScoreEngine(),
        ]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for engine in self.engines:
            df = engine.detect(df)

        return df
"""
Pipeline complet du Market Structure Engine.

Ce pipeline orchestre tous les détecteurs et engines
dans le bon ordre.
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


class MarketStructurePipeline:
    """
    Exécute tout le moteur de Market Structure.
    """

    def __init__(self):
        self.detectors = [
            SwingDetector(window=2),
            TrendDetector(),

            BOSDetector(),
            BOSEngine(),

            CHOCHDetector(),

            EqualHighLowDetector(tolerance=0.001),
            LiquidityPoolEngine(),
            LiquidityDetector(),

            CHOCHEngine(),

            FairValueGapDetector(),
            FairValueGapEngine(),

            OrderBlockDetector(lookback=5),
            OrderBlockEngine(lookback=5),
            OrderBlockLifecycleEngine(),

            SupportResistanceDetector(),
            PremiumDiscountDetector(equilibrium_tolerance=0.001),
            SessionDetector(),
        ]

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for detector in self.detectors:
            df = detector.detect(df)

        return df
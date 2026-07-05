"""
Pipeline Market Structure.

Construit et exécute le pipeline complet du moteur SMC.
"""

from __future__ import annotations

import pandas as pd

from src.config.market_structure_config import (
    SWING_WINDOW,
    EQUAL_HIGH_LOW_ATR_MULTIPLIER,
    ORDER_BLOCK_LOOKBACK,
)

from src.pipelines.pipeline_builder import PipelineBuilder

from src.market_structure.detectors.swing_detector import SwingDetector
from src.market_structure.detectors.trend_detector import TrendDetector
from src.market_structure.detectors.bos_detector import BOSDetector
from src.market_structure.detectors.choch_detector import CHOCHDetector
from src.market_structure.detectors.equal_high_low_detector import EqualHighLowDetector
from src.market_structure.detectors.liquidity_detector import LiquidityDetector
from src.market_structure.detectors.fair_value_gap_detector import FairValueGapDetector
from src.market_structure.detectors.order_block_detector import OrderBlockDetector
from src.market_structure.detectors.support_resistance_detector import SupportResistanceDetector
from src.market_structure.detectors.premium_discount_detector import PremiumDiscountDetector
from src.market_structure.detectors.session_detector import SessionDetector

from src.market_structure.engines.bos_engine import BOSEngine
from src.market_structure.engines.choch_engine import CHOCHEngine
from src.market_structure.engines.liquidity_pool_engine import LiquidityPoolEngine
from src.market_structure.engines.fair_value_gap_engine import FairValueGapEngine
from src.market_structure.engines.order_block_engine import OrderBlockEngine
from src.market_structure.engines.order_block_lifecycle_engine import OrderBlockLifecycleEngine
from src.market_structure.engines.market_structure_score_engine import MarketStructureScoreEngine
from src.market_structure.engines.trade_setup_engine import TradeSetupEngine

from src.risk.risk_engine import RiskEngine


class MarketStructurePipeline:
    """
    Pipeline complet de structure de marché.

    Il ajoute :
    - swings
    - tendance
    - BOS / CHOCH
    - liquidité
    - FVG
    - order blocks
    - support / résistance
    - premium / discount
    - sessions
    - score global
    - setup de trading
    - risque
    """

    def __init__(self):
        self.pipeline = (
            PipelineBuilder()
            .add(SwingDetector(window=SWING_WINDOW))
            .add(TrendDetector())
            .add(BOSDetector())
            .add(BOSEngine())
            .add(CHOCHDetector())
            .add(CHOCHEngine())
            .add(EqualHighLowDetector(atr_multiplier=EQUAL_HIGH_LOW_ATR_MULTIPLIER))
            .add(LiquidityPoolEngine())
            .add(LiquidityDetector())
            .add(FairValueGapDetector())
            .add(FairValueGapEngine())
            .add(OrderBlockDetector(lookback=ORDER_BLOCK_LOOKBACK))
            .add(OrderBlockEngine(lookback=ORDER_BLOCK_LOOKBACK))
            .add(OrderBlockLifecycleEngine())
            .add(SupportResistanceDetector())
            .add(PremiumDiscountDetector())
            .add(SessionDetector())
            .add(MarketStructureScoreEngine())
            .add(TradeSetupEngine())
            .add(RiskEngine())
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pipeline.run(df)
"""
Construction d'un dataset V3 pour un seul timeframe.

Exemples :

python -m src.pipelines.build_dataset_v3_single_timeframe --timeframe 5m
python -m src.pipelines.build_dataset_v3_single_timeframe --timeframe 15m
"""

from __future__ import annotations

import argparse
from pathlib import Path

#import pandas as pd

from src.database.services.dataset_builder_service import DatasetBuilderService
from src.pipelines.build_dataset_v3_from_postgresql import (
    build_ml_dataset,
    build_timeframe_dataset,
)
from src.pipelines.market_structure_pipeline import MarketStructurePipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)

VALID_TIMEFRAMES = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}


def build_single_timeframe_dataset(
    timeframe: str,
    symbol: str = "GC",
) -> None:
    """
    Construit les datasets Analysis et ML d'un seul timeframe.
    """

    timeframe = timeframe.lower()
    symbol = symbol.upper()

    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Timeframe invalide : {timeframe}. "
            f"Valeurs acceptées : {sorted(VALID_TIMEFRAMES)}"
        )

    analysis_dir = Path("data/final/analysis") / symbol
    ml_dir = Path("data/final/ml") / symbol

    analysis_dir.mkdir(parents=True, exist_ok=True)
    ml_dir.mkdir(parents=True, exist_ok=True)

    analysis_output = (
        analysis_dir
        / f"dataset_analysis_v3_{symbol}_{timeframe}.csv"
    )

    ml_output = (
        ml_dir
        / f"dataset_ml_v3_{symbol}_{timeframe}.csv"
    )

    logger.info(
        f"Construction du dataset V3 : {symbol} | {timeframe}"
    )

    service = DatasetBuilderService()
    market_pipeline = MarketStructurePipeline()

    # La fonction utilise actuellement GC et TradingView
    # définis dans le pipeline V3 principal.
    analysis_df = build_timeframe_dataset(
        service=service,
        market_pipeline=market_pipeline,
        timeframe=timeframe,
    )

    if analysis_df.empty:
        raise ValueError(
            f"Aucune donnée générée pour {symbol} {timeframe}."
        )

    ml_df = build_ml_dataset(analysis_df)

    analysis_df.to_csv(
        analysis_output,
        index=False,
        encoding="utf-8-sig",
    )

    ml_df.to_csv(
        ml_output,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        f"Dataset Analysis : {analysis_output}"
    )
    logger.info(
        f"Dimensions Analysis : {analysis_df.shape}"
    )

    logger.info(
        f"Dataset ML : {ml_output}"
    )
    logger.info(
        f"Dimensions ML : {ml_df.shape}"
    )

    logger.info(
        "Construction terminée avec succès."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construire un dataset V3 par timeframe."
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=sorted(VALID_TIMEFRAMES),
        help="Timeframe à traiter, par exemple 5m ou 15m.",
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à traiter. Valeur par défaut : GC.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    build_single_timeframe_dataset(
        timeframe=arguments.timeframe,
        symbol=arguments.symbol,
    )
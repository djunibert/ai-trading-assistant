"""
Construction du dataset ML V3 depuis PostgreSQL Neon.

Flux :

PostgreSQL
    ↓
DatasetBuilderService
    ↓
Indicateurs techniques
    ↓
Market Structure Pipeline
    ↓
Encodage
    ↓
Target
    ↓
dataset_analysis_v3.csv
dataset_ml_v3.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.database.services.dataset_builder_service import (
    DatasetBuilderService,
)
from src.pipelines.market_structure_pipeline import (
    MarketStructurePipeline,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# Configuration
# ==========================================================

SYMBOL = "GC"
PLATFORM = "TradingView"

TIMEFRAMES = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
]

OUTPUT_DIR = Path("data/final")

ANALYSIS_OUTPUT = OUTPUT_DIR / "dataset_analysis_v3.csv"
ML_OUTPUT = OUTPUT_DIR / "dataset_ml_v3.csv"

TARGET_THRESHOLD = 0.001


# ==========================================================
# Colonnes texte à encoder
# ==========================================================

TEXT_MAPPINGS = {
    "bos_v2_direction": {
        "NONE": 0,
        "BULLISH": 1,
        "BEARISH": -1,
    },
    "choch_v2_direction": {
        "NONE": 0,
        "BULLISH": 1,
        "BEARISH": -1,
    },
    "liquidity_pool_direction": {
        "NONE": 0,
        "BUY_SIDE": 1,
        "SELL_SIDE": -1,
    },
    "fvg_direction": {
        "NONE": 0,
        "BULLISH": 1,
        "BEARISH": -1,
    },
    "ob_v2_direction": {
        "NONE": 0,
        "BULLISH": 1,
        "BEARISH": -1,
    },
    "trade_setup": {
        "NO_TRADE": 0,
        "BUY": 1,
        "SELL": -1,
    },
    "risk_signal": {
        "NO_TRADE": 0,
        "BUY": 1,
        "SELL": -1,
    },
    "session": {
        "OTHER": 0,
        "ASIA": 1,
        "LONDON": 2,
        "NEW_YORK": 3,
        "LONDON_NY_OVERLAP": 4,
    },
}


# Ces colonnes restent dans le dataset d’analyse,
# mais elles ne seront pas utilisées comme entrées ML.
ML_COLUMNS_TO_EXCLUDE = [
    "datetime",
    "symbol",
    "timeframe",
    "platform",
    "quality_status",
    "file_name",
    "future_return_1",
]


# ==========================================================
# Indicateurs techniques
# ==========================================================

def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calcule le RSI.
    """

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    return 100 - (100 / (1 + relative_strength))


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Calcule l'Average True Range.
    """

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_technical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ajoute les indicateurs nécessaires au Market Structure Pipeline.
    """

    df = df.copy()

    df["adj_close"] = df["close"]

    df["ema_20"] = df["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    df["ema_50"] = df["close"].ewm(
        span=50,
        adjust=False,
    ).mean()

    df["ema_200"] = df["close"].ewm(
        span=200,
        adjust=False,
    ).mean()

    df["rsi_14"] = calculate_rsi(
        df["close"],
        period=14,
    )

    df["atr_14"] = calculate_atr(
        df,
        period=14,
    )

    df["return_1"] = df["close"].pct_change()

    df["volatility_20"] = (
        df["return_1"]
        .rolling(window=20)
        .std()
    )

    return df


# ==========================================================
# Variables temporelles
# ==========================================================

def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extrait les variables de temps à partir de datetime.
    """

    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce",
    )

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["week"] = (
        df["datetime"]
        .dt.isocalendar()
        .week
        .astype("Int64")
    )
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute

    return df


# ==========================================================
# Encodage
# ==========================================================

def encode_text_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode les sorties texte du Market Structure Pipeline.
    """

    df = df.copy()

    for column, mapping in TEXT_MAPPINGS.items():
        if column in df.columns:
            df[column] = (
                df[column]
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    return df


def encode_identifiers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode le symbole et le timeframe.
    """

    df = df.copy()

    if "symbol" in df.columns:
        df["asset_code"] = (
            df["symbol"]
            .astype("category")
            .cat.codes
        )

    timeframe_mapping = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    if "timeframe" in df.columns:
        df["timeframe_code"] = (
            df["timeframe"]
            .map(timeframe_mapping)
            .fillna(0)
            .astype(int)
        )

    return df


# ==========================================================
# Target
# ==========================================================

def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crée une classification à trois classes.

    SELL     = -1
    NO_TRADE = 0
    BUY      = 1

    Le calcul est effectué séparément pour chaque timeframe.
    """

    df = df.copy()

    df["future_return_1"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1
    )

    df["target"] = 0

    df.loc[
        df["future_return_1"] > TARGET_THRESHOLD,
        "target",
    ] = 1

    df.loc[
        df["future_return_1"] < -TARGET_THRESHOLD,
        "target",
    ] = -1

    # La dernière ligne n'a pas de rendement futur.
    df = df.dropna(
        subset=["future_return_1"]
    )

    return df


# ==========================================================
# Construction d'un timeframe
# ==========================================================

def build_timeframe_dataset(
    service: DatasetBuilderService,
    market_pipeline: MarketStructurePipeline,
    timeframe: str,
) -> pd.DataFrame:
    """
    Construit les features pour un timeframe.
    """

    logger.info(
        f"Construction du dataset : {SYMBOL} | {timeframe}"
    )

    df = service.build_market_dataset(
        symbol=SYMBOL,
        timeframe=timeframe,
        platform=PLATFORM,
    )

    if df.empty:
        logger.warning(
            f"Aucune donnée disponible pour {timeframe}."
        )
        return df

    df = add_time_features(df)
    df = add_technical_features(df)

    # Les premières lignes ne disposent pas encore de toutes
    # les valeurs EMA, RSI, ATR et volatilité.
    df = df.dropna(
        subset=[
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi_14",
            "atr_14",
            "return_1",
            "volatility_20",
        ]
    ).reset_index(drop=True)

    df = market_pipeline.run(df)

    df = encode_text_columns(df)
    df = create_target(df)

    logger.info(
        f"Dimensions {timeframe} : {df.shape}"
    )

    return df


# ==========================================================
# Dataset ML
# ==========================================================

def build_ml_dataset(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transforme le dataset d'analyse en dataset numérique ML.
    """

    df = analysis_df.copy()

    df = encode_identifiers(df)

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    df[bool_columns] = (
        df[bool_columns]
        .astype(int)
    )

    columns_to_drop = [
        column
        for column in ML_COLUMNS_TO_EXCLUDE
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    df = df.select_dtypes(
        include=["number"]
    ).copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.fillna(0)

    if "target" not in df.columns:
        raise ValueError(
            "La colonne target est absente du dataset ML V3."
        )

    return df


# ==========================================================
# Pipeline principal
# ==========================================================

def build_dataset_v3() -> None:
    """
    Construit les deux datasets V3 depuis PostgreSQL.
    """

    logger.info(
        "Début de la construction du dataset V3 depuis PostgreSQL."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    service = DatasetBuilderService()
    market_pipeline = MarketStructurePipeline()

    timeframe_datasets = []

    for timeframe in TIMEFRAMES:
        try:
            timeframe_df = build_timeframe_dataset(
                service=service,
                market_pipeline=market_pipeline,
                timeframe=timeframe,
            )

            if not timeframe_df.empty:
                timeframe_datasets.append(
                    timeframe_df
                )

        except Exception:
            logger.exception(
                f"Erreur pendant le traitement du timeframe {timeframe}."
            )
            raise

    if not timeframe_datasets:
        raise ValueError(
            "Aucun dataset n'a été construit depuis PostgreSQL."
        )

    analysis_df = pd.concat(
        timeframe_datasets,
        ignore_index=True,
    )

    analysis_df = analysis_df.sort_values(
        by=[
            "timeframe",
            "datetime",
        ]
    ).reset_index(drop=True)

    ml_df = build_ml_dataset(
        analysis_df
    )

    analysis_df.to_csv(
        ANALYSIS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    ml_df.to_csv(
        ML_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(
        f"Dataset d'analyse sauvegardé : {ANALYSIS_OUTPUT}"
    )
    logger.info(
        f"Dimensions analysis V3 : {analysis_df.shape}"
    )

    logger.info(
        f"Dataset ML sauvegardé : {ML_OUTPUT}"
    )
    logger.info(
        f"Dimensions ML V3 : {ml_df.shape}"
    )

    logger.info(
        "Construction du dataset V3 terminée avec succès."
    )


if __name__ == "__main__":
    build_dataset_v3()
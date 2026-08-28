"""
Pipeline de construction du dataset ML V2.

Ce pipeline génère deux fichiers :

1. dataset_analysis_v2.csv
   - conserve datetime, asset, timeframe
   - conserve toutes les features
   - utile pour analyse, Streamlit, backtesting, vérification

2. dataset_ml_v2.csv
   - conserve uniquement les colonnes numériques utiles au modèle
   - exclut les colonnes de décision/risk trop directes
   - prêt pour l'entraînement ML
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipelines.market_structure_pipeline import MarketStructurePipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)

INPUT_DIR = Path("data/features/market_macro")
OUTPUT_DIR = Path("data/final")

ANALYSIS_OUTPUT_FILE = OUTPUT_DIR / "dataset_analysis_v2.csv"
ML_OUTPUT_FILE = OUTPUT_DIR / "dataset_ml_v2.csv"


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
    "session": {
        "OTHER": 0,
        "ASIA": 1,
        "LONDON": 2,
        "NEW_YORK": 3,
        "LONDON_NY_OVERLAP": 4,
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
}


RISK_COLUMNS_TO_EXCLUDE_FROM_ML = [
    "risk_signal",
    "risk_trade_allowed",
    "risk_entry_price",
    "risk_stop_loss",
    "risk_take_profit",
    "risk_stop_distance",
    "risk_reward_ratio",
    "sl_price",
    "sl_distance",
    "tp_price",
    "tp_distance",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les variables temporelles utiles au modèle.
    """

    df = df.copy()

    if "datetime" not in df.columns:
        raise ValueError("La colonne datetime est manquante.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["week"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute

    return df


def encode_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes texte en valeurs numériques.
    """

    df = df.copy()

    for column, mapping in TEXT_MAPPINGS.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(0)

    return df


def encode_asset_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode asset et timeframe pour le dataset ML.
    """

    df = df.copy()

    if "asset" in df.columns:
        df["asset_code"] = df["asset"].astype("category").cat.codes

    if "timeframe" in df.columns:
        df["timeframe_code"] = df["timeframe"].astype("category").cat.codes

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une target simple à 3 classes :

    BUY = 1
    SELL = -1
    NO_TRADE = 0

    Ici on utilise le rendement futur.
    """

    df = df.copy()

    if "close" not in df.columns:
        raise ValueError("La colonne close est manquante.")

    df["future_return_1"] = df["close"].shift(-1) / df["close"] - 1

    threshold = 0.001

    df["target"] = 0
    df.loc[df["future_return_1"] > threshold, "target"] = 1
    df.loc[df["future_return_1"] < -threshold, "target"] = -1

    df = df.dropna(subset=["future_return_1"])

    return df


def build_analysis_dataset() -> pd.DataFrame:
    """
    Construit le dataset complet d'analyse.
    """

    logger.info("Construction du dataset analysis V2")

    pipeline = MarketStructurePipeline()
    all_dataframes = []

    for timeframe_dir in INPUT_DIR.iterdir():
        if not timeframe_dir.is_dir():
            continue

        timeframe = timeframe_dir.name
        logger.info(f"Traitement timeframe : {timeframe}")

        for file_path in timeframe_dir.glob("*.csv"):
            asset = file_path.stem
            logger.info(f"Lecture : {asset} - {timeframe}")

            df = pd.read_csv(file_path)

            df["asset"] = asset
            df["timeframe"] = timeframe

            df = add_time_features(df)
            df = pipeline.run(df)
            df = encode_text_columns(df)
            df = create_target(df)

            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("Aucun fichier trouvé dans data/features/market_macro.")

    final_df = pd.concat(all_dataframes, ignore_index=True)

    final_df = final_df.replace([float("inf"), float("-inf")], 0)
    final_df = final_df.fillna(0)

    return final_df


def build_ml_dataset(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le dataset utilisé pour l'entraînement ML.
    """

    df = analysis_df.copy()

    df = encode_asset_timeframe(df)

    # On retire les colonnes texte non utilisables directement par les modèles.
    columns_to_drop = [
        "datetime",
        "asset",
        "timeframe",
    ]

    columns_to_drop.extend(RISK_COLUMNS_TO_EXCLUDE_FROM_ML)

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors="ignore",
    )

    # Convertir booléens en 0/1.
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Garder uniquement les colonnes numériques.
    df = df.select_dtypes(include=["number"]).copy()

    df = df.replace([float("inf"), float("-inf")], 0)
    df = df.fillna(0)

    if "target" not in df.columns:
        raise ValueError("La colonne target est manquante dans dataset_ml_v2.")

    return df


def build_dataset_v2() -> None:
    """
    Fonction principale.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis_df = build_analysis_dataset()
    ml_df = build_ml_dataset(analysis_df)

    analysis_df.to_csv(
        ANALYSIS_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    ml_df.to_csv(
        ML_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(f"Dataset analysis sauvegardé : {ANALYSIS_OUTPUT_FILE}")
    logger.info(f"Dimensions analysis : {analysis_df.shape}")

    logger.info(f"Dataset ML sauvegardé : {ML_OUTPUT_FILE}")
    logger.info(f"Dimensions ML : {ml_df.shape}")

    logger.info("Construction du dataset V2 terminée.")


if __name__ == "__main__":
    build_dataset_v2()
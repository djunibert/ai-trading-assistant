"""
Pipeline de construction du dataset ML V2.

Objectif :
- lire les features market_macro
- appliquer le Market Structure Pipeline
- encoder les colonnes texte
- garder les colonnes numériques
- sauvegarder data/final/dataset_ml_v2.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipelines.market_structure_pipeline import MarketStructurePipeline
from src.utils.logger import get_logger


logger = get_logger(__name__)

INPUT_DIR = Path("data/features/market_macro")
OUTPUT_DIR = Path("data/final")
OUTPUT_FILE = OUTPUT_DIR / "dataset_ml_v2.csv"


def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mappings = {
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

    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping).fillna(0)

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir les booléens en 0/1
    bool_columns = df.select_dtypes(include=["bool"]).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Supprimer datetime pour l'entraînement, mais le garder si tu veux analyser.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Garder uniquement les colonnes numériques + datetime temporairement
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # Nettoyage
    numeric_df = numeric_df.replace([float("inf"), float("-inf")], 0)
    numeric_df = numeric_df.fillna(0)

    return numeric_df


def build_dataset() -> pd.DataFrame:
    logger.info("Construction du dataset ML V2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

            df = pipeline.run(df)

            df = encode_columns(df)
            df = clean_dataset(df)

            all_dataframes.append(df)

    if not all_dataframes:
        raise ValueError("Aucun fichier trouvé pour construire le dataset.")

    final_df = pd.concat(all_dataframes, ignore_index=True)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(f"Dataset sauvegardé : {OUTPUT_FILE}")
    logger.info(f"Dimensions : {final_df.shape}")

    return final_df


if __name__ == "__main__":
    build_dataset()
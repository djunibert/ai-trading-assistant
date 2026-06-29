"""
Module : clean_market_data.py

Description
-----------
Nettoie les données de marché multi-timeframe.

Entrée :
    data/raw/market/<timeframe>/<asset>.csv

Sortie :
    data/processed/market/<timeframe>/<asset>.csv

Auteur : Junior Hébert
Projet : AI Trading System
"""

import pandas as pd

from src.utils.paths import RAW_MARKET_DIR, PROCESSED_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def clean_market_file(file_path, timeframe):
    """
    Nettoie un fichier de données de marché.
    """

    asset_name = file_path.stem

    logger.info(f"Nettoyage : {asset_name} - {timeframe}")

    # Lecture du fichier CSV
    df = pd.read_csv(file_path)

    # Normalisation des noms de colonnes
    df.columns = [
        str(col).lower().replace(" ", "_")
        for col in df.columns
    ]

    # Renommer la colonne Date si nécessaire
    if "date" in df.columns:
        df = df.rename(columns={"date": "datetime"})

    # Conversion des dates
    # utc=True permet d'éviter les problèmes de fuseau horaire
    df["datetime"] = (
        pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
    )

    # Colonnes numériques
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]

    for col in numeric_columns:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Ajout des informations
    df["asset"] = asset_name
    df["timeframe"] = timeframe

    # Suppression des lignes invalides
    df = df.dropna(subset=["datetime", "close"])

    # Suppression des doublons
    df = df.drop_duplicates(subset=["datetime"])

    # Tri chronologique
    df = df.sort_values("datetime")

    return df


def clean_market_data():
    """
    Nettoie tous les fichiers market en conservant
    la structure par timeframe.
    """

    input_base_dir = RAW_MARKET_DIR
    output_base_dir = PROCESSED_DIR / "market"

    output_base_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for timeframe_dir in input_base_dir.iterdir():

        if not timeframe_dir.is_dir():
            continue

        timeframe = timeframe_dir.name

        output_timeframe_dir = output_base_dir / timeframe

        output_timeframe_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for file_path in timeframe_dir.glob("*.csv"):

            df = clean_market_file(
                file_path=file_path,
                timeframe=timeframe
            )

            output_file = output_timeframe_dir / file_path.name

            df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            logger.info(f"Fichier créé : {output_file}")


if __name__ == "__main__":
    clean_market_data()
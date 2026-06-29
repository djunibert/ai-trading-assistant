"""
Module : macro_features.py

Description
-----------
Ajoute les variables macroéconomiques FRED aux features de marché.

Entrées :
    data/features/market/<timeframe>/<asset>.csv
    data/processed/macro_clean.csv

Sortie :
    data/features/market_macro/<timeframe>/<asset>.csv

Auteur : Junior Hébert
Projet : AI Trading System
"""

import pandas as pd

from src.utils.paths import FEATURES_DIR, PROCESSED_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise la colonne datetime.

    Objectif :
    - convertir en datetime
    - gérer les timezones
    - supprimer la timezone pour permettre merge_asof()
    """

    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    df = df.dropna(subset=["datetime"])

    return df


def add_macro_features_to_file(
    market_file,
    macro_df: pd.DataFrame,
    timeframe: str
) -> pd.DataFrame:
    """
    Ajoute les variables macroéconomiques à un fichier market features.
    """

    asset_name = market_file.stem

    logger.info(f"Ajout macro : {asset_name} - {timeframe}")

    market_df = pd.read_csv(market_file)

    market_df = normalize_datetime(market_df)
    macro_df = normalize_datetime(macro_df)

    market_df = market_df.sort_values("datetime")
    macro_df = macro_df.sort_values("datetime")

    merged_df = pd.merge_asof(
        market_df,
        macro_df,
        on="datetime",
        direction="backward"
    )

    return merged_df


def build_macro_features() -> None:
    """
    Fusionne les features marché avec les données macro FRED.
    """

    logger.info("Début de la création des features macroéconomiques...")

    macro_file = PROCESSED_DIR / "macro_clean.csv"

    if not macro_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {macro_file}")

    macro_df = pd.read_csv(macro_file)

    input_base_dir = FEATURES_DIR / "market"
    output_base_dir = FEATURES_DIR / "market_macro"

    if not input_base_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {input_base_dir}")

    output_base_dir.mkdir(parents=True, exist_ok=True)

    for timeframe_dir in input_base_dir.iterdir():

        if not timeframe_dir.is_dir():
            continue

        timeframe = timeframe_dir.name

        output_timeframe_dir = output_base_dir / timeframe
        output_timeframe_dir.mkdir(parents=True, exist_ok=True)

        for market_file in timeframe_dir.glob("*.csv"):

            merged_df = add_macro_features_to_file(
                market_file=market_file,
                macro_df=macro_df,
                timeframe=timeframe
            )

            output_file = output_timeframe_dir / market_file.name

            merged_df.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            logger.info(f"Fichier créé : {output_file}")

    logger.info("Création des features macroéconomiques terminée.")


if __name__ == "__main__":
    build_macro_features()
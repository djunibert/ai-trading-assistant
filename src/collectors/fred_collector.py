"""
Module : collect_fred_data.py

Description
-----------
Ce module permet de télécharger les principaux indicateurs
macroéconomiques depuis l'API FRED.

Les données sont sauvegardées dans :
data/raw/macro/macro_fred.csv

Indicateurs collectés :
    - CPI
    - NFP
    - Fed Funds Rate
    - Unemployment Rate

Auteur : Junior Hébert
Projet : AI Trading System
Version : 1.0
"""

import pandas as pd
import requests

from src.utils.config import (
    FRED_API_KEY,
    FRED_SERIES,
    START_DATE,
)

from src.utils.paths import RAW_MACRO_DIR
from src.utils.logger import get_logger


# Initialisation du logger
logger = get_logger(__name__)


def collect_fred_series(name: str, series_id: str) -> pd.DataFrame:
    """
    Télécharge une série économique depuis FRED.

    Args:
        name (str):
            Nom de l'indicateur.

        series_id (str):
            Identifiant officiel de la série FRED.

    Returns:
        pd.DataFrame:
            DataFrame contenant :
                datetime
                valeur
    """

    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY est absent du fichier .env")

    url = "https://api.stlouisfed.org/fred/series/observations"

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
    }

    logger.info(f"Téléchargement : {name}")

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()["observations"]

    df = pd.DataFrame(data)

    df = df[["date", "value"]]

    df["date"] = pd.to_datetime(df["date"])

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.rename(
        columns={
            "date": "datetime",
            "value": name,
        }
    )

    return df


def collect_fred_data():
    """
    Télécharge toutes les séries macroéconomiques
    définies dans config.py puis les fusionne
    dans un seul fichier CSV.
    """

    logger.info("Début de la collecte FRED...")

    RAW_MACRO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    final_df = None

    for name, series_id in FRED_SERIES.items():

        df = collect_fred_series(
            name,
            series_id
        )

        if final_df is None:
            final_df = df

        else:
            final_df = final_df.merge(
                df,
                on="datetime",
                how="outer"
            )

    final_df = final_df.sort_values(
        by="datetime"
    )

    output_file = RAW_MACRO_DIR / "macro_fred.csv"

    final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Fichier créé : {output_file}")

    logger.info("Collecte FRED terminée.")


if __name__ == "__main__":

    collect_fred_data()

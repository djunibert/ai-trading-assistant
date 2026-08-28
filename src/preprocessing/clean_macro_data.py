"""
Module : clean_macro_data.py

Description
-----------
Nettoie les données macroéconomiques collectées depuis FRED.

Entrée :
    data/raw/macro/macro_fred.csv

Sortie :
    data/processed/macro_clean.csv
"""

import pandas as pd

from src.utils.paths import RAW_MACRO_DIR, PROCESSED_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def clean_macro_data():
    """
    Nettoie les données macroéconomiques FRED.
    """

    input_file = RAW_MACRO_DIR / "macro_fred.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_file}")

    logger.info(f"Lecture du fichier : {input_file}")

    df = pd.read_csv(input_file)

    df.columns = [
        str(col).lower().replace(" ", "_")
        for col in df.columns
    ]

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime"])

    df = df.drop_duplicates(subset=["datetime"])

    df = df.sort_values("datetime")

    # Remplir les valeurs manquantes avec la dernière valeur connue
    df = df.ffill()

    output_file = PROCESSED_DIR / "macro_clean.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Fichier créé : {output_file}")


if __name__ == "__main__":
    clean_macro_data()

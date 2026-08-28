"""
Génération d'un rapport Evidently pour détecter
la dérive des données de marché.

Exemples :

python -m src.monitoring.evidently_report `
    --symbol GC `
    --timeframe 5m

python -m src.monitoring.evidently_report `
    --symbol GC `
    --timeframe 15m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import (
    DataDriftPreset,
    DataSummaryPreset,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

VALID_TIMEFRAMES = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}


def clean_numeric_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nettoie les données numériques avant Evidently.

    Les valeurs infinies sont remplacées par NaN,
    puis les valeurs manquantes sont remplacées par zéro.
    """

    cleaned_df = dataframe.copy()

    cleaned_df = cleaned_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    cleaned_df = cleaned_df.fillna(0)

    return cleaned_df


def remove_constant_features(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Supprime les colonnes constantes dans les données
    de référence ou les données courantes.

    Une colonne constante possède une seule valeur unique.
    Elle ne peut pas produire une corrélation valide.
    """

    reference_unique_values = reference_data.nunique(
        dropna=False
    )

    current_unique_values = current_data.nunique(
        dropna=False
    )

    valid_columns = [
        column
        for column in reference_data.columns
        if reference_unique_values.get(
            column,
            0,
        ) > 1
        and current_unique_values.get(
            column,
            0,
        ) > 1
    ]

    removed_columns = [
        column
        for column in reference_data.columns
        if column not in valid_columns
    ]

    if removed_columns:
        logger.warning(
            "Colonnes constantes supprimées : "
            f"{removed_columns}"
        )

    if not valid_columns:
        raise ValueError(
            "Aucune feature variable disponible "
            "pour générer le rapport Evidently."
        )

    return (
        reference_data[
            valid_columns
        ].copy(),
        current_data[
            valid_columns
        ].copy(),
    )


def align_columns(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conserve uniquement les colonnes communes
    dans le même ordre.
    """

    common_columns = reference_data.columns.intersection(
        current_data.columns
    )

    if common_columns.empty:
        raise ValueError(
            "Aucune colonne commune entre les données "
            "de référence et les données courantes."
        )

    return (
        reference_data[
            common_columns
        ].copy(),
        current_data[
            common_columns
        ].copy(),
    )


def create_reference_and_current_data(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée les périodes utilisées par Evidently.

    Référence :
    les 70 % premières observations.

    Données courantes :
    les 15 % dernières observations.
    """

    total_rows = len(
        features_df
    )

    if total_rows < 100:
        raise ValueError(
            "Le dataset est trop petit pour produire "
            "un rapport de dérive fiable."
        )

    reference_end = int(
        total_rows
        * TRAIN_RATIO
    )

    current_start = int(
        total_rows
        * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    reference_data = features_df.iloc[
        :reference_end
    ].copy()

    current_data = features_df.iloc[
        current_start:
    ].copy()

    if reference_data.empty:
        raise ValueError(
            "Les données de référence sont vides."
        )

    if current_data.empty:
        raise ValueError(
            "Les données courantes sont vides."
        )

    return (
        reference_data,
        current_data,
    )


def prepare_evidently_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prépare les données de référence et courantes.
    """

    reference_data, current_data = (
        create_reference_and_current_data(
            features_df=dataframe
        )
    )

    reference_data = clean_numeric_data(
        reference_data
    )

    current_data = clean_numeric_data(
        current_data
    )

    reference_data, current_data = align_columns(
        reference_data=reference_data,
        current_data=current_data,
    )

    reference_data, current_data = remove_constant_features(
        reference_data=reference_data,
        current_data=current_data,
    )

    reference_data, current_data = align_columns(
        reference_data=reference_data,
        current_data=current_data,
    )

    return (
        reference_data,
        current_data,
    )


def generate_evidently_report(
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Génère le rapport Evidently HTML et JSON.
    """

    symbol = symbol.upper()
    timeframe = timeframe.lower()

    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Timeframe invalide : {timeframe}"
        )

    logger.info(
        f"Génération du rapport Evidently : "
        f"{symbol} | {timeframe}"
    )

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="monitoring",
    )

    analysis_df = trainer.load_analysis_dataset()

    features_df, target = (
        trainer.prepare_tabular_features(
            analysis_df
        )
    )

    logger.info(
        f"Dataset Analysis : {analysis_df.shape}"
    )

    logger.info(
        f"Features numériques : {features_df.shape}"
    )

    logger.info(
        "Distribution target :\n"
        f"{target.value_counts()}"
    )

    reference_data, current_data = prepare_evidently_data(
        dataframe=features_df
    )

    logger.info(
        f"Données de référence : "
        f"{reference_data.shape}"
    )

    logger.info(
        f"Données courantes : "
        f"{current_data.shape}"
    )

    report = Report(
        [
            DataDriftPreset(),
            DataSummaryPreset(),
        ]
    )

    result = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    output_directory = (
        Path("reports/monitoring")
        / symbol
        / timeframe
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_path = (
        output_directory
        / "evidently_report.html"
    )

    json_path = (
        output_directory
        / "evidently_report.json"
    )

    result.save_html(
        str(html_path)
    )

    json_path.write_text(
        result.json(),
        encoding="utf-8",
    )

    logger.info(
        f"Rapport HTML créé : {html_path}"
    )

    logger.info(
        f"Rapport JSON créé : {json_path}"
    )

    logger.info(
        "Rapport Evidently terminé avec succès."
    )

    return html_path


def parse_arguments() -> argparse.Namespace:
    """
    Lit les arguments PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Générer un rapport Evidently "
            "pour un symbole et un timeframe."
        )
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à analyser.",
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=sorted(
            VALID_TIMEFRAMES
        ),
        help="Timeframe à analyser.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    generate_evidently_report(
        symbol=arguments.symbol,
        timeframe=arguments.timeframe,
    )
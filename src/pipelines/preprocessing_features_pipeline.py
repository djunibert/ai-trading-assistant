"""
Module : preprocessing_features_pipeline.py

Description
-----------
Pipeline complet de préparation des données.

Étapes :
1. Nettoyage des données de marché
2. Nettoyage des données macroéconomiques
3. Création des indicateurs techniques

Auteur : Junior Hébert
Projet : AI Trading System
"""

from src.preprocessing.clean_market_data import clean_market_data
from src.preprocessing.clean_macro_data import clean_macro_data
from src.feature_engineering.technical_indicators import (
    build_market_features,
)

from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_preprocessing_pipeline():
    """
    Exécute le pipeline complet de préparation des données.
    """

    logger.info("=" * 60)
    logger.info("Début du pipeline de prétraitement")
    logger.info("=" * 60)

    logger.info("Étape 1 : Nettoyage des données de marché")
    clean_market_data()

    logger.info("Étape 2 : Nettoyage des données macro")
    clean_macro_data()

    logger.info("Étape 3 : Création des indicateurs techniques")
    build_market_features()

    logger.info("=" * 60)
    logger.info("Pipeline terminé avec succès")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing_pipeline()
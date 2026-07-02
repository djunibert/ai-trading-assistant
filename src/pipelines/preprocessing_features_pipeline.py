"""
Module : preprocessing_features_pipeline.py

Description
-----------
Pipeline complet de préparation des données.

Étapes :
1. Nettoyage des données de marché
2. Nettoyage des données macroéconomiques
3. Création des indicateurs techniques
4. Ajout des variables macroéconomiques
5. Création du dataset final ML V1

Auteur : Junior Hébert
Projet : AI Trading System
"""

from src.preprocessing.clean_market_data import clean_market_data
from src.preprocessing.clean_macro_data import clean_macro_data
from src.feature_engineering.target_features import build_binary_target
from src.feature_engineering.technical_indicators import build_market_features
from src.feature_engineering.macro_features import build_macro_features
from src.feature_engineering.build_dataset import build_dataset_v1

from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_preprocessing_pipeline() -> None:
    """
    Exécute le pipeline complet de prétraitement
    et de création des features.
    """

    logger.info("=" * 60)
    logger.info("Début du pipeline de prétraitement et de features")
    logger.info("=" * 60)

    logger.info("Étape 1 : Nettoyage des données de marché")
    clean_market_data()

    logger.info("Étape 2 : Nettoyage des données macroéconomiques")
    clean_macro_data()

    logger.info("Étape 3 : Création des indicateurs techniques")
    build_market_features()

    logger.info("Étape 4 : Ajout des features macroéconomiques")
    build_macro_features()

    logger.info("Étape 5 : Création du dataset final ML V1")
    build_dataset_v1()

    logger.info("=" * 60)
    logger.info("Pipeline terminé avec succès")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_preprocessing_pipeline()
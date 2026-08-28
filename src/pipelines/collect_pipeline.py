"""
Module : run_data_pipeline.py

Description
-----------
Point d'entrée principal du pipeline de données.

Ce script exécute :
1. Collecte des données de marché
2. Collecte des données macroéconomiques FRED

Auteur : Junior Hébert
Projet : AI Trading System
"""

from src.collectors.yahoo_collector import collect_market_data
from src.collectors.fred_collector import collect_fred_data
from src.utils.logger import get_logger


logger = get_logger(__name__)


def run_data_pipeline():
    """
    Exécute le pipeline complet de collecte de données.
    """

    logger.info("Démarrage du pipeline de données...")

    logger.info("Étape 1 : Collecte Market Data")
    collect_market_data()

    logger.info("Étape 2 : Collecte FRED Data")
    collect_fred_data()

    logger.info("Pipeline de données terminé.")


if __name__ == "__main__":
    run_data_pipeline()

"""
Module : base_market_collector.py

Description
-----------
Classe de base pour tous les collecteurs de données de marché.

Elle définit une structure commune pour :
- télécharger les données
- nettoyer les données
- sauvegarder les données

Auteur : Junior Hébert
Projet : AI Trading System
"""

from abc import ABC, abstractmethod
import pandas as pd

from src.utils.logger import get_logger


class BaseMarketCollector(ABC):
    """
    Classe abstraite pour les collecteurs de marché.
    """

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def collect(self) -> None:
        """
        Lance la collecte complète.
        """
        pass

    @abstractmethod
    def download_symbol(
        self,
        symbol: str,
        interval: str,
        period: str | None = None,
    ) -> pd.DataFrame:
        """
        Télécharge les données d'un symbole.
        """
        pass

    @abstractmethod
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données téléchargées.
        """
        pass

    @abstractmethod
    def save_dataframe(
        self,
        df: pd.DataFrame,
        output_file,
    ) -> None:
        """
        Sauvegarde les données nettoyées.
        """
        pass
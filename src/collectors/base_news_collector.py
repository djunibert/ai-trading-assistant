"""
Base commune pour les collecteurs de nouvelles économiques.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseNewsCollector(ABC):
    """
    Classe abstraite pour tous les collecteurs de news économiques.
    """

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """
        Collecte les événements économiques.
        """
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        """
        Sauvegarde les événements économiques.
        """
        pass

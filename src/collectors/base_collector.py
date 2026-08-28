from abc import ABC, abstractmethod
import pandas as pd

from src.utils.logger import get_logger


class BaseCollector(ABC):

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """
        Télécharge les données.
        """
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame):
        """
        Sauvegarde les données.
        """
        pass

    def run(self):
        self.logger.info("Début de la collecte...")

        df = self.collect()

        self.save(df)

        self.logger.info("Collecte terminée.")

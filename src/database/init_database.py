"""
Initialisation de la base PostgreSQL Neon.

Ce script :
- lit le fichier SQL du projet
- ouvre une connexion SQLAlchemy
- exécute le schéma dans une transaction
"""

from pathlib import Path

from src.config.database import engine
from src.utils.logger import get_logger


logger = get_logger(__name__)

SQL_FILE = Path(
    "database/schema/create_trading_database_tables.sql"
)


def initialize_database() -> None:
    """
    Exécute le fichier SQL complet dans PostgreSQL Neon.
    """

    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier SQL introuvable : {SQL_FILE}"
        )

    sql_script = SQL_FILE.read_text(
        encoding="utf-8"
    )

    if not sql_script.strip():
        raise ValueError(
            f"Le fichier SQL est vide : {SQL_FILE}"
        )

    logger.info(
        f"Lecture du schéma SQL : {SQL_FILE}"
    )

    # raw_connection donne accès directement au pilote psycopg.
    # Cette approche accepte le script PostgreSQL complet,
    # y compris les fonctions, triggers et blocs contenant plusieurs
    # instructions SQL.
    raw_connection = engine.raw_connection()

    try:
        with raw_connection.cursor() as cursor:
            cursor.execute(sql_script)

        raw_connection.commit()

        logger.info(
            "Schéma PostgreSQL créé avec succès dans Neon."
        )

    except Exception:
        raw_connection.rollback()
        logger.exception(
            "Erreur pendant l'initialisation de la base."
        )
        raise

    finally:
        raw_connection.close()


if __name__ == "__main__":
    initialize_database()
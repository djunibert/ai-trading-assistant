"""
Configuration centralisée de la connexion PostgreSQL Neon.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


# Charge les variables définies dans le fichier .env.
load_dotenv()

# Récupère l'URL de connexion PostgreSQL.
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL est absente du fichier .env."
    )

# Crée un moteur SQLAlchemy réutilisable dans tout le projet.
engine: Engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

# Fabrique les sessions utilisées pour les opérations en base.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def test_database_connection() -> bool:
    """
    Vérifie que la connexion PostgreSQL fonctionne.
    """

    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.scalar() == 1
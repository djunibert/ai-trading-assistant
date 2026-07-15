"""
Repository Market Bars.

Centralise toutes les lectures de données OHLCV
depuis PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.config.database import engine


class MarketBarRepository:
    """
    Accès aux données de la table market_bars.
    """

    def load_market_data(
        self,
        symbol: str,
        timeframe: str,
        platform: str = "TradingView",
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Charge les données OHLCV.
        """

        query = """
            SELECT
                mb.timestamp AS datetime,
                mb.open,
                mb.high,
                mb.low,
                mb.close,
                mb.volume,
                s.symbol,
                tf.name AS timeframe,
                p.name AS platform,
                mb.quality_status,
                mb.file_name
            FROM market_bars mb

            INNER JOIN symbols s
                ON s.id = mb.symbol_id

            INNER JOIN timeframes tf
                ON tf.id = mb.timeframe_id

            INNER JOIN platforms p
                ON p.id = mb.platform_id

            WHERE
                UPPER(s.symbol)=UPPER(:symbol)
                AND LOWER(tf.name)=LOWER(:timeframe)
                AND LOWER(p.name)=LOWER(:platform)
        """

        parameters = {
            "symbol": symbol,
            "timeframe": timeframe,
            "platform": platform,
        }

        if start_date is not None:

            query += """
                AND mb.timestamp >= :start_date
            """

            parameters["start_date"] = start_date

        if end_date is not None:

            query += """
                AND mb.timestamp <= :end_date
            """

            parameters["end_date"] = end_date

        query += """
            ORDER BY
                mb.timestamp
        """

        if limit is not None:

            query += """
                LIMIT :limit
            """

            parameters["limit"] = limit

        with engine.connect() as connection:

            df = pd.read_sql_query(
                sql=text(query),
                con=connection,
                params=parameters,
            )

        if not df.empty:

            df["datetime"] = pd.to_datetime(
                df["datetime"],
                utc=True,
            )

            numeric_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            for column in numeric_columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        return df

    def count_market_bars(
        self,
        symbol: str,
        timeframe: str,
        platform: str = "TradingView",
    ) -> int:
        """
        Nombre de bougies.
        """

        query = text(
            """
            SELECT
                COUNT(*)

            FROM market_bars mb

            INNER JOIN symbols s
                ON s.id = mb.symbol_id

            INNER JOIN timeframes tf
                ON tf.id = mb.timeframe_id

            INNER JOIN platforms p
                ON p.id = mb.platform_id

            WHERE
                UPPER(s.symbol)=UPPER(:symbol)
                AND LOWER(tf.name)=LOWER(:timeframe)
                AND LOWER(p.name)=LOWER(:platform)
            """
        )

        with engine.connect() as connection:

            result = connection.execute(
                query,
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "platform": platform,
                },
            ).scalar_one()

        return int(result)

    def list_available_datasets(
        self,
    ) -> pd.DataFrame:
        """
        Retourne tous les datasets disponibles.
        """

        query = text(
            """
            SELECT

                p.name AS platform,

                s.symbol,

                tf.name AS timeframe,

                tf.minutes,

                COUNT(*) AS row_count,

                MIN(mb.timestamp) AS start_datetime,

                MAX(mb.timestamp) AS end_datetime

            FROM market_bars mb

            INNER JOIN symbols s
                ON s.id = mb.symbol_id

            INNER JOIN timeframes tf
                ON tf.id = mb.timeframe_id

            INNER JOIN platforms p
                ON p.id = mb.platform_id

            GROUP BY

                p.name,

                s.symbol,

                tf.name,

                tf.minutes

            ORDER BY

                p.name,

                s.symbol,

                tf.minutes
            """
        )

        with engine.connect() as connection:

            df = pd.read_sql_query(
                sql=query,
                con=connection,
            )

        return df.drop(columns=["minutes"])
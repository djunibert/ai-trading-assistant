"""
Importateur CSV TradingView vers PostgreSQL Neon.

Le script :

1. Parcourt data/raw/tradingview
2. Détecte le symbole et le timeframe avec les dossiers
3. Normalise les colonnes TradingView
4. Valide les données OHLCV
5. Insère les nouvelles bougies dans market_bars
6. Ignore les doublons
7. Enregistre le résultat dans data_import_log
8. Déplace les fichiers importés vers data/imported/tradingview
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config.database import engine
from src.utils.logger import get_logger


logger = get_logger(__name__)

RAW_ROOT = Path("data/raw/tradingview")
IMPORTED_ROOT = Path("data/imported/tradingview")


COLUMN_ALIASES = {
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les noms de colonnes TradingView.
    """

    df = df.copy()

    normalized_names = {}

    for column in df.columns:
        clean_name = str(column).strip().lower()

        if clean_name in COLUMN_ALIASES:
            normalized_names[column] = COLUMN_ALIASES[clean_name]

    df = df.rename(columns=normalized_names)

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    if "volume" not in df.columns:
        df["volume"] = None

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les types, enlève les lignes invalides
    et supprime les doublons du fichier.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
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

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    valid_ohlc = (
        (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["high"] >= df["low"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["low"] <= df["high"])
    )

    valid_volume = (
        df["volume"].isna()
        | (df["volume"] >= 0)
    )

    df = df[valid_ohlc & valid_volume]

    df = df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    )

    df = df.sort_values("timestamp")

    return df.reset_index(drop=True)


def get_platform_id(connection, platform_name: str) -> int:
    """
    Retourne l'identifiant de la plateforme.
    """

    result = connection.execute(
        text(
            """
            SELECT id
            FROM platforms
            WHERE LOWER(name) = LOWER(:name)
            """
        ),
        {"name": platform_name},
    ).scalar_one_or_none()

    if result is None:
        result = connection.execute(
            text(
                """
                INSERT INTO platforms (name)
                VALUES (:name)
                RETURNING id
                """
            ),
            {"name": platform_name},
        ).scalar_one()

    return int(result)


def get_symbol_id(connection, symbol: str) -> int:
    """
    Retourne ou crée le symbole.
    """

    result = connection.execute(
        text(
            """
            SELECT id
            FROM symbols
            WHERE symbol = :symbol
            """
        ),
        {"symbol": symbol},
    ).scalar_one_or_none()

    if result is None:
        result = connection.execute(
            text(
                """
                INSERT INTO symbols (
                    symbol,
                    description,
                    asset_class,
                    exchange
                )
                VALUES (
                    :symbol,
                    :description,
                    :asset_class,
                    :exchange
                )
                RETURNING id
                """
            ),
            {
                "symbol": symbol,
                "description": f"{symbol} Futures",
                "asset_class": "FUTURES",
                "exchange": "COMEX/CME",
            },
        ).scalar_one()

    return int(result)


def get_timeframe_id(connection, timeframe: str) -> int:
    """
    Retourne l'identifiant du timeframe.
    """

    result = connection.execute(
        text(
            """
            SELECT id
            FROM timeframes
            WHERE name = :name
            """
        ),
        {"name": timeframe},
    ).scalar_one_or_none()

    if result is None:
        raise ValueError(
            f"Timeframe absent de la base : {timeframe}"
        )

    return int(result)


def create_import_log(
    connection,
    platform_id: int,
    file_path: Path,
    symbol: str,
    timeframe: str,
    rows_read: int,
) -> int:
    """
    Crée le journal initial de l'import.
    """

    return int(
        connection.execute(
            text(
                """
                INSERT INTO data_import_log (
                    platform_id,
                    file_name,
                    symbol,
                    timeframe,
                    rows_read,
                    status
                )
                VALUES (
                    :platform_id,
                    :file_name,
                    :symbol,
                    :timeframe,
                    :rows_read,
                    'started'
                )
                RETURNING id
                """
            ),
            {
                "platform_id": platform_id,
                "file_name": file_path.name,
                "symbol": symbol,
                "timeframe": timeframe,
                "rows_read": rows_read,
            },
        ).scalar_one()
    )


def complete_import_log(
    connection,
    import_log_id: int,
    rows_inserted: int,
    rows_rejected: int,
) -> None:
    """
    Marque l'import comme terminé.
    """

    connection.execute(
        text(
            """
            UPDATE data_import_log
            SET
                rows_inserted = :rows_inserted,
                rows_rejected = :rows_rejected,
                status = 'completed',
                completed_at = NOW()
            WHERE id = :import_log_id
            """
        ),
        {
            "rows_inserted": rows_inserted,
            "rows_rejected": rows_rejected,
            "import_log_id": import_log_id,
        },
    )


def fail_import_log(
    import_log_id: int,
    error_message: str,
) -> None:
    """
    Enregistre l'échec d'un import.
    """

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE data_import_log
                SET
                    status = 'failed',
                    error_message = :error_message,
                    completed_at = NOW()
                WHERE id = :import_log_id
                """
            ),
            {
                "error_message": error_message[:5000],
                "import_log_id": import_log_id,
            },
        )


def move_to_imported(file_path: Path) -> Path:
    """
    Déplace le CSV vers data/imported/tradingview
    en conservant symbole/timeframe.
    """

    relative_path = file_path.relative_to(RAW_ROOT)
    destination = IMPORTED_ROOT / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    shutil.move(
        str(file_path),
        str(destination),
    )

    return destination


def import_csv_file(file_path: Path) -> None:
    """
    Importe un seul fichier CSV.
    """

    relative_path = file_path.relative_to(RAW_ROOT)

    if len(relative_path.parts) < 3:
        raise ValueError(
            "Chemin invalide. Format attendu : "
            "SYMBOL/TIMEFRAME/fichier.csv"
        )

    symbol = relative_path.parts[0].upper()
    timeframe = relative_path.parts[1].lower()

    logger.info(
        f"Import : {symbol} | {timeframe} | {file_path.name}"
    )

    original_df = pd.read_csv(file_path)

    rows_read = len(original_df)

    normalized_df = normalize_columns(original_df)
    clean_df = clean_dataframe(normalized_df)

    rows_rejected = rows_read - len(clean_df)
    import_log_id = 0

    try:
        with engine.begin() as connection:
            platform_id = get_platform_id(
                connection,
                "TradingView",
            )

            symbol_id = get_symbol_id(
                connection,
                symbol,
            )

            timeframe_id = get_timeframe_id(
                connection,
                timeframe,
            )

            import_log_id = create_import_log(
                connection=connection,
                platform_id=platform_id,
                file_path=file_path,
                symbol=symbol,
                timeframe=timeframe,
                rows_read=rows_read,
            )

            records = []

            for row in clean_df.itertuples(index=False):
                records.append(
                    {
                        "platform_id": platform_id,
                        "symbol_id": symbol_id,
                        "timeframe_id": timeframe_id,
                        "timestamp": row.timestamp.to_pydatetime(),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": (
                            None
                            if pd.isna(row.volume)
                            else float(row.volume)
                        ),
                        "contract_symbol": None,
                        "file_name": file_path.name,
                    }
                )

            insert_statement = text(
                """
                INSERT INTO market_bars (
                    platform_id,
                    symbol_id,
                    timeframe_id,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    contract_symbol,
                    file_name,
                    quality_status
                )
                VALUES (
                    :platform_id,
                    :symbol_id,
                    :timeframe_id,
                    :timestamp,
                    :open,
                    :high,
                    :low,
                    :close,
                    :volume,
                    :contract_symbol,
                    :file_name,
                    'validated'
                )
                ON CONFLICT (
                    platform_id,
                    symbol_id,
                    timeframe_id,
                    timestamp
                )
                DO NOTHING
                """
            )

            inserted_rows = 0

            if records:
                result = connection.execute(
                    insert_statement,
                    records,
                )

                inserted_rows = max(
                    result.rowcount or 0,
                    0,
                )

            complete_import_log(
                connection=connection,
                import_log_id=import_log_id,
                rows_inserted=inserted_rows,
                rows_rejected=rows_rejected,
            )

        destination = move_to_imported(file_path)

        logger.info(
            f"Import terminé : {inserted_rows} nouvelles lignes"
        )
        logger.info(
            f"Fichier déplacé vers : {destination}"
        )

    except Exception as error:
        if import_log_id:
            fail_import_log(
                import_log_id=import_log_id,
                error_message=str(error),
            )

        logger.exception(
            f"Échec de l'import : {file_path}"
        )

        raise


def import_all_tradingview_csv() -> None:
    """
    Importe tous les CSV présents dans raw/tradingview.
    """

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {RAW_ROOT}"
        )

    csv_files = sorted(
        RAW_ROOT.glob("*/*/*.csv")
    )

    if not csv_files:
        logger.warning(
            "Aucun fichier CSV TradingView à importer."
        )
        return

    logger.info(
        f"Nombre de fichiers détectés : {len(csv_files)}"
    )

    success_count = 0
    failure_count = 0

    for file_path in csv_files:
        try:
            import_csv_file(file_path)
            success_count += 1

        except Exception:
            failure_count += 1

    logger.info(
        f"Imports réussis : {success_count}"
    )
    logger.info(
        f"Imports échoués : {failure_count}"
    )


if __name__ == "__main__":
    import_all_tradingview_csv()
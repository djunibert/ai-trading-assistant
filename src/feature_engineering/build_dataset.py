"""
Création du dataset final ML V1.

Entrée :
    data/features/market_macro/<timeframe>/<asset>.csv

Sortie :
    data/final/dataset_ml_v1.csv
"""

import pandas as pd

from src.utils.paths import FEATURES_DIR, FINAL_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dataset_v1(
    selected_assets=None,
    selected_timeframes=None
) -> None:
    if selected_assets is None:
        selected_assets = ["gold", "silver"]

    if selected_timeframes is None:
        selected_timeframes = ["1d", "4h", "1h"]

    input_base_dir = FEATURES_DIR / "market_macro"

    if not input_base_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {input_base_dir}")

    all_data = []

    for timeframe in selected_timeframes:
        timeframe_dir = input_base_dir / timeframe

        if not timeframe_dir.exists():
            logger.warning(f"Timeframe absent : {timeframe}")
            continue

        for asset in selected_assets:
            file_path = timeframe_dir / f"{asset}.csv"

            if not file_path.exists():
                logger.warning(f"Fichier absent : {file_path}")
                continue

            logger.info(f"Ajout dataset : {asset} - {timeframe}")

            df = pd.read_csv(file_path, parse_dates=["datetime"])

            df["asset"] = asset
            df["timeframe"] = timeframe

            all_data.append(df)

    if not all_data:
        raise ValueError("Aucune donnée trouvée pour créer le dataset final.")

    dataset = pd.concat(all_data, ignore_index=True)

    dataset = dataset.sort_values(["asset", "timeframe", "datetime"])

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    output_file = FINAL_DIR / "dataset_ml_v1.csv"

    dataset.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Dataset final créé : {output_file}")
    logger.info(f"Nombre de lignes : {len(dataset)}")


if __name__ == "__main__":
    build_dataset_v1()
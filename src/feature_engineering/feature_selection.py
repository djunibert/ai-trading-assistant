"""
Feature Selection.

Sélectionne les meilleures features pour les modèles ML/DL.
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_ml_v2.csv")
OUTPUT_PATH = Path("data/final/dataset_lstm_v2.csv")
REPORT_PATH = Path("reports/selected_features_lstm_v2.csv")


def select_features(
    max_features: int = 80,
) -> None:
    logger.info("Chargement du dataset ML V2")

    df = pd.read_csv(DATASET_PATH)

    columns_to_drop = [
        "target",
        "future_return_1",
    ]

    X = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors="ignore",
    )

    y = df["target"]

    logger.info(f"Features initiales : {X.shape[1]}")

    selector = VarianceThreshold(
        threshold=0.0
    )

    X_selected_array = selector.fit_transform(X)

    selected_columns = X.columns[
        selector.get_support()
    ]

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_columns,
    )

    logger.info(f"Après variance threshold : {X_selected.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    logger.info("Calcul de l'importance des features")

    model.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "feature": X_selected.columns,
        "importance": model.feature_importances_,
    }).sort_values(
        by="importance",
        ascending=False,
    )

    top_features = importance_df.head(max_features)["feature"].tolist()

    final_df = df[top_features + ["target"]].copy()

    if "future_return_1" in df.columns:
        final_df["future_return_1"] = df["future_return_1"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    importance_df.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info(f"Dataset LSTM sauvegardé : {OUTPUT_PATH}")
    logger.info(f"Rapport sauvegardé : {REPORT_PATH}")
    logger.info(f"Nombre final de features : {len(top_features)}")


if __name__ == "__main__":
    select_features(
        max_features=80
    )
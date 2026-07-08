"""
Training Random Forest V2.

Ce fichier entraîne le modèle ML V2 à partir de :
- dataset_ml_v2.csv
- features Market Structure
- features Trade Setup
- target BUY / SELL / NO_TRADE

La gestion MLflow est séparée dans :
src/mlops/mlflow_manager.py
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.mlops.mlflow_manager import MLflowManager
from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_ml_v2.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest_v2.pkl"


def train_model_v2() -> None:
    logger.info("Chargement du dataset ML V2")

    df = pd.read_csv(DATASET_PATH)

    logger.info(f"Dataset chargé : {df.shape}")

    if "target" not in df.columns:
        raise ValueError("La colonne target est manquante.")

    columns_to_drop = [
        "target",
        "future_return_1",
    ]

    X = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        errors="ignore",
    )

    y = df["target"]

    logger.info(f"Nombre de features : {X.shape[1]}")
    logger.info(f"Distribution target :\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    mlflow_manager = MLflowManager(
        experiment_name="AI Trading System V2"
    )

    with mlflow_manager.start_run(
        run_name="random_forest_v2_market_structure"
    ):
        logger.info("Entraînement du modèle Random Forest V2")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"F1 macro : {f1_macro:.4f}")
        logger.info(f"F1 weighted : {f1_weighted:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))

        mlflow_manager.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": "None",
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "dataset": str(DATASET_PATH),
            "features_count": X.shape[1],
        })

        mlflow_manager.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

        mlflow_manager.log_sklearn_model(
            model=model,
            artifact_name="random_forest_v2",
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": X.columns.tolist(),
            },
            MODEL_PATH,
        )

        logger.info(f"Modèle sauvegardé : {MODEL_PATH}")
        logger.info("Training V2 terminé avec succès.")


if __name__ == "__main__":
    train_model_v2()
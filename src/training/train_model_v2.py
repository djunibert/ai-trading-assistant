"""
Training Random Forest V2.

Ce fichier entraîne le modèle Random Forest V2 avec :
- dataset_ml_v2.csv
- DataSplitter centralisé
- MLflow séparé dans mlops/mlflow_manager.py
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data.data_splitter import DataSplitter
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

    # ==========================================================
    # MISE À JOUR IMPORTANTE
    # ==========================================================
    # Avant, le split était fait directement dans ce fichier avec :
    # train_test_split(X, y, ...)
    #
    # Maintenant, on utilise DataSplitter.
    # Cela permet à Random Forest, XGBoost et les futurs modèles
    # d'utiliser exactement la même logique de séparation.
    # ==========================================================

    X_train, X_test, y_train, y_test = DataSplitter().split(
        df=df,
        target_column="target",
        columns_to_drop=["future_return_1"],
    )

    logger.info(f"X_train : {X_train.shape}")
    logger.info(f"X_test  : {X_test.shape}")
    logger.info(f"y_train : {y_train.shape}")
    logger.info(f"y_test  : {y_test.shape}")

    logger.info(f"Distribution target train :\n{y_train.value_counts()}")

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
            "features_count": X_train.shape[1],
            "splitter": "DataSplitter",
            "test_size": 0.20,
            "random_state": 42,
            "stratify": True,
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
                "features": X_train.columns.tolist(),
            },
            MODEL_PATH,
        )

        logger.info(f"Modèle sauvegardé : {MODEL_PATH}")
        logger.info("Training Random Forest V2 terminé avec succès.")


if __name__ == "__main__":
    train_model_v2()
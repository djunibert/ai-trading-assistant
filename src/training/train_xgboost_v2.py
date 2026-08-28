"""
Training XGBoost V2.

Entraîne un modèle XGBoost avec le dataset ML V2.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.mlops.mlflow_manager import MLflowManager
from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_ml_v2.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgboost_v2.pkl"


def train_xgboost_v2() -> None:
    logger.info("Chargement du dataset ML V2")

    df = pd.read_csv(DATASET_PATH)

    if "target" not in df.columns:
        raise ValueError("La colonne target est manquante.")

    # XGBoost préfère des classes 0, 1, 2 au lieu de -1, 0, 1
    target_mapping = {
        -1: 0,  # SELL
        0: 1,   # NO_TRADE
        1: 2,   # BUY
    }

    inverse_target_mapping = {
        0: -1,
        1: 0,
        2: 1,
    }

    y = df["target"].map(target_mapping)

    X = df.drop(
        columns=["target", "future_return_1"],
        errors="ignore",
    )

    logger.info(f"Dataset : {df.shape}")
    logger.info(f"Nombre de features : {X.shape[1]}")
    logger.info(f"Distribution target :\n{df['target'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    mlflow_manager = MLflowManager(
        experiment_name="AI Trading System V2"
    )

    with mlflow_manager.start_run(
        run_name="xgboost_v2_market_structure"
    ):
        logger.info("Entraînement du modèle XGBoost V2")

        model.fit(X_train, y_train)

        y_pred_encoded = model.predict(X_test)

        y_test_original = y_test.map(inverse_target_mapping)
        y_pred_original = pd.Series(y_pred_encoded).map(inverse_target_mapping)

        accuracy = accuracy_score(y_test_original, y_pred_original)
        f1_macro = f1_score(y_test_original, y_pred_original, average="macro")
        f1_weighted = f1_score(
            y_test_original,
            y_pred_original,
            average="weighted",
        )

        logger.info(f"Accuracy : {accuracy:.4f}")
        logger.info(f"F1 macro : {f1_macro:.4f}")
        logger.info(f"F1 weighted : {f1_weighted:.4f}")
        logger.info("\n" + classification_report(y_test_original, y_pred_original))

        mlflow_manager.log_params({
            "model_type": "XGBClassifier",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "dataset": str(DATASET_PATH),
            "features_count": X.shape[1],
        })

        mlflow_manager.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

        mlflow_manager.log_xgboost_model(
            model=model,
            artifact_name="xgboost_v2",
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "features": X.columns.tolist(),
                "target_mapping": target_mapping,
                "inverse_target_mapping": inverse_target_mapping,
            },
            MODEL_PATH,
        )

        logger.info(f"Modèle sauvegardé : {MODEL_PATH}")
        logger.info("Training XGBoost V2 terminé avec succès.")


if __name__ == "__main__":
    train_xgboost_v2()
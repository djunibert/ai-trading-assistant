"""
Module : train_model.py

Entraîne un modèle Random Forest V1 avec MLflowManager.
"""

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from src.mlops.mlflow_manager import MLflowManager
from src.utils.paths import FINAL_DIR, MODELS_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)


def train_random_forest_v1() -> None:
    dataset_file = FINAL_DIR / "dataset_ml_v1.csv"

    logger.info(f"Lecture du dataset : {dataset_file}")

    df = pd.read_csv(dataset_file)

    if "target" not in df.columns:
        raise ValueError("La colonne target est manquante.")

    features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "return_1",
        "volatility_20",
        "cpi",
        "nfp",
        "fed_rate",
        "unemployment_rate",
    ]

    features = [col for col in features if col in df.columns]

    df = df.dropna(subset=features + ["target"])

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    mlflow_manager = MLflowManager(
        experiment_name="AI Trading System V1"
    )

    with mlflow_manager.start_run(run_name="random_forest_v1"):

        mlflow_manager.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 8,
            "features": ",".join(features),
        })

        logger.info("Entraînement du modèle...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logger.info(f"Accuracy : {accuracy}")
        logger.info(f"F1-score : {f1}")

        print(classification_report(y_test, y_pred))

        mlflow_manager.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
        })

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_file = MODELS_DIR / "random_forest_v1.pkl"

        joblib.dump(
            {
                "model": model,
                "features": features,
            },
            model_file
        )

        mlflow_manager.log_sklearn_model(
            model=model,
            model_name="random_forest_v1"
        )

        logger.info(f"Modèle sauvegardé : {model_file}")


if __name__ == "__main__":
    train_random_forest_v1()
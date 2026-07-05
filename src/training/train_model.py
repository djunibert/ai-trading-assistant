"""
Module : train_model.py

Entraîne un modèle Random Forest V1 avec une target à 3 classes :

0 = SELL
1 = NO_TRADE
2 = BUY
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
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

    logger.info("Distribution de la target :")
    logger.info(f"\n{df['target'].value_counts().sort_index()}")

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    mlflow_manager = MLflowManager(
        experiment_name="AI Trading System V1"
    )

    with mlflow_manager.start_run(run_name="random_forest_v1_multiclass"):
        mlflow_manager.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "class_weight": "balanced_subsample",
            "features": ",".join(features),
            "target": "0=SELL,1=NO_TRADE,2=BUY",
        })

        logger.info("Entraînement du modèle...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        logger.info(f"Accuracy : {accuracy}")
        logger.info(f"F1 macro : {f1_macro}")
        logger.info(f"F1 weighted : {f1_weighted}")

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0,
            )
        )

        cm = confusion_matrix(y_test, y_pred)

        logger.info("Matrice de confusion :")
        logger.info(f"\n{cm}")

        mlflow_manager.log_metrics({
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        })

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        model_file = MODELS_DIR / "random_forest_v1.pkl"

        joblib.dump(
            {
                "model": model,
                "features": features,
                "target_mapping": {
                    0: "SELL",
                    1: "NO_TRADE",
                    2: "BUY",
                },
            },
            model_file,
        )

        with open("reports/classification_report.json", "w") as file:
            json.dump(report, file, indent=4)

        pd.DataFrame(cm).to_csv(
            "reports/confusion_matrix.csv",
            index=False,
            encoding="utf-8-sig",
        )

        mlflow_manager.log_sklearn_model(
            model=model,
            model_name="random_forest_v1",
        )

        logger.info(f"Modèle sauvegardé : {model_file}")


if __name__ == "__main__":
    train_random_forest_v1()

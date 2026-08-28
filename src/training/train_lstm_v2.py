"""
Training LSTM V2.

Pipeline complet :

Dataset ML
    ↓
Construction des séquences
    ↓
Train / Validation / Test
    ↓
LSTM
    ↓
MLflow
    ↓
Sauvegarde
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.deep_learning.sequence_builder import SequenceBuilder
from src.deep_learning.lstm_model import build_lstm_model
from src.deep_learning.lstm_trainer import LSTMTrainer
from src.mlops.mlflow_manager import MLflowManager
from src.utils.logger import get_logger


logger = get_logger(__name__)

DATASET_PATH = Path("data/final/dataset_lstm_v2.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_v2.keras"
SCALER_PATH = MODEL_DIR / "lstm_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "lstm_features.pkl"

SEQUENCE_LENGTH = 32
EPOCHS = 10
BATCH_SIZE = 64


def train_lstm_v2() -> None:
    """
    Entraîne et sauvegarde le modèle LSTM V2.
    """

    logger.info("Chargement du dataset LSTM V2")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    logger.info(f"Dimensions du dataset : {df.shape}")

    builder = SequenceBuilder(
        sequence_length=SEQUENCE_LENGTH
    )

    X, y, feature_names, scaler = builder.build(
        df=df,
        target_column="target",
        columns_to_exclude=["future_return_1"],
    )

    logger.info(f"Shape X : {X.shape}")
    logger.info(f"Shape y : {y.shape}")
    logger.info(f"Nombre de features : {len(feature_names)}")

    # Le LSTM attend des classes numériques positives :
    # SELL = 0
    # NO_TRADE = 1
    # BUY = 2
    target_mapping = {
        -1: 0,
        0: 1,
        1: 2,
    }

    y = np.vectorize(target_mapping.get)(y).astype(np.int32)

    # Premier découpage :
    # 80 % entraînement/validation
    # 20 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Deuxième découpage :
    # 80 % entraînement
    # 20 % validation dans la portion train
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    logger.info(f"Train : {X_train.shape}")
    logger.info(f"Validation : {X_val.shape}")
    logger.info(f"Test : {X_test.shape}")

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        n_features=X.shape[2],
        n_classes=3,
    )

    model.summary()

    trainer = LSTMTrainer(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    mlflow_manager = MLflowManager(
        experiment_name="AI Trading System V2"
    )

    with mlflow_manager.start_run(
        run_name="lstm_v2"
    ):
        logger.info("Début de l'entraînement LSTM V2")

        trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        logger.info(f"Loss : {loss:.4f}")
        logger.info(f"Accuracy : {accuracy:.4f}")

        mlflow_manager.log_params({
            "model_type": "LSTM",
            "sequence_length": SEQUENCE_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "features_count": len(feature_names),
            "dataset": str(DATASET_PATH),
            "test_size": 0.20,
            "validation_size": 0.20,
            "random_state": 42,
        })

        mlflow_manager.log_metrics({
            "accuracy": float(accuracy),
            "loss": float(loss),
        })

        # Sauvegarde du modèle et des objets nécessaires
        # pour refaire des prédictions plus tard.
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(feature_names, FEATURES_PATH)

        logger.info(f"Modèle sauvegardé : {MODEL_PATH}")
        logger.info(f"Scaler sauvegardé : {SCALER_PATH}")
        logger.info(f"Features sauvegardées : {FEATURES_PATH}")
        logger.info("Entraînement LSTM V2 terminé avec succès.")


if __name__ == "__main__":
    train_lstm_v2()
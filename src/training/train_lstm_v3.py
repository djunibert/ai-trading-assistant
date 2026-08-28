"""
Entraînement LSTM V3 par symbole et timeframe.

Le script utilise directement le dataset Analysis V3.

Exemples :

python -m src.training.train_lstm_v3 --symbol GC --timeframe 5m

python -m src.training.train_lstm_v3 --symbol GC --timeframe 15m
"""

from __future__ import annotations

import argparse
import json
#from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Input,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.mlops.mlflow_manager import MLflowManager
from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)

SEQUENCE_LENGTH = 32
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def encode_target(y: np.ndarray) -> np.ndarray:
    """
    Convertit les classes :

    SELL     -1 → 0
    NO_TRADE  0 → 1
    BUY       1 → 2
    """

    mapping = {
        -1: 0,
        0: 1,
        1: 2,
    }

    encoded = np.array(
        [mapping[int(value)] for value in y],
        dtype=np.int32,
    )

    return encoded


def decode_target(y: np.ndarray) -> np.ndarray:
    """
    Reconvertit les classes :

    0 → SELL
    1 → NO_TRADE
    2 → BUY
    """

    mapping = {
        0: -1,
        1: 0,
        2: 1,
    }

    decoded = np.array(
        [mapping[int(value)] for value in y],
        dtype=np.int32,
    )

    return decoded


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforme les lignes en séquences temporelles.
    """

    if len(X) <= sequence_length:
        raise ValueError(
            "Le dataset ne contient pas assez de lignes "
            "pour créer les séquences."
        )

    X_sequences = []
    y_sequences = []

    for index in range(sequence_length, len(X)):
        X_sequences.append(
            X[index - sequence_length:index]
        )

        y_sequences.append(
            y[index]
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(y_sequences, dtype=np.int32),
    )


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Sépare les données dans l'ordre chronologique.

    Aucun mélange aléatoire n'est effectué.
    Cela évite d'utiliser des données futures
    pendant l'entraînement.
    """

    total_rows = len(X)

    train_end = int(
        total_rows * TRAIN_RATIO
    )

    validation_end = int(
        total_rows
        * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_validation = X[
        train_end:validation_end
    ]

    y_validation = y[
        train_end:validation_end
    ]

    X_test = X[validation_end:]
    y_test = y[validation_end:]

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def build_lstm_model(
    sequence_length: int,
    feature_count: int,
) -> Sequential:
    """
    Construit le modèle LSTM V3.
    """

    model = Sequential(
        [
            Input(
                shape=(
                    sequence_length,
                    feature_count,
                )
            ),

            LSTM(
                units=128,
                return_sequences=True,
            ),

            BatchNormalization(),
            Dropout(0.30),

            LSTM(
                units=64,
                return_sequences=False,
            ),

            BatchNormalization(),
            Dropout(0.30),

            Dense(
                units=32,
                activation="relu",
            ),

            Dropout(0.20),

            Dense(
                units=3,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def calculate_class_weights(
    y_train: np.ndarray,
) -> dict[int, float]:
    """
    Compense le déséquilibre entre SELL, NO_TRADE et BUY.
    """

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    return {
        int(class_value): float(weight)
        for class_value, weight in zip(
            classes,
            weights,
            strict=True,
        )
    }


def train_lstm_v3(
    symbol: str,
    timeframe: str,
) -> None:
    """
    Entraîne le LSTM V3.
    """

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="lstm",
    )

    df = trainer.load_analysis_dataset()

    # La classe BaseTrainer sélectionne les colonnes numériques
    # et retire datetime, target, future_return_1 et les identifiants.
    X_df, y_series = trainer.prepare_tabular_features(
        df
    )

    feature_names = X_df.columns.tolist()

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = encode_target(
        y_series.to_numpy()
    )

    (
        X_train_raw,
        X_validation_raw,
        X_test_raw,
        y_train_raw,
        y_validation_raw,
        y_test_raw,
    ) = chronological_split(
        X=X,
        y=y,
    )

    # Le scaler est entraîné uniquement avec les données train.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_validation_scaled = scaler.transform(
        X_validation_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    X_train, y_train = create_sequences(
        X=X_train_scaled,
        y=y_train_raw,
        sequence_length=SEQUENCE_LENGTH,
    )

    X_validation, y_validation = create_sequences(
        X=X_validation_scaled,
        y=y_validation_raw,
        sequence_length=SEQUENCE_LENGTH,
    )

    X_test, y_test = create_sequences(
        X=X_test_scaled,
        y=y_test_raw,
        sequence_length=SEQUENCE_LENGTH,
    )

    logger.info(
        f"Symbole : {trainer.symbol}"
    )

    logger.info(
        f"Timeframe : {trainer.timeframe}"
    )

    logger.info(
        f"Features : {len(feature_names)}"
    )

    logger.info(
        f"Train : {X_train.shape}"
    )

    logger.info(
        f"Validation : {X_validation.shape}"
    )

    logger.info(
        f"Test : {X_test.shape}"
    )

    class_weights = calculate_class_weights(
        y_train
    )

    logger.info(
        f"Poids des classes : {class_weights}"
    )

    model = build_lstm_model(
        sequence_length=SEQUENCE_LENGTH,
        feature_count=len(feature_names),
    )

    model.summary()

    model_path = (
        trainer.model_dir
        / "lstm_v3.keras"
    )

    scaler_path = (
        trainer.model_dir
        / "scaler_v3.pkl"
    )

    features_path = (
        trainer.model_dir
        / "features_v3.pkl"
    )

    metadata_path = (
        trainer.model_dir
        / "metadata_v3.json"
    )

    metrics_path = (
        trainer.report_dir
        / "metrics_v3.json"
    )

    classification_report_path = (
        trainer.report_dir
        / "classification_report_v3.txt"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001,
        ),

        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    mlflow_manager = MLflowManager(
        experiment_name=(
            f"AI Trading System V3 - "
            f"{trainer.symbol}"
        )
    )

    run_name = (
        f"lstm_v3_"
        f"{trainer.symbol}_"
        f"{trainer.timeframe}"
    )

    with mlflow_manager.start_run(
        run_name=run_name
    ):
        logger.info(
            "Début de l'entraînement LSTM V3."
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            shuffle=False,
            verbose=1,
        )

        test_loss, test_accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0,
        )

        probabilities = model.predict(
            X_test,
            verbose=0,
        )

        predictions_encoded = np.argmax(
            probabilities,
            axis=1,
        )

        y_test_original = decode_target(
            y_test
        )

        predictions_original = decode_target(
            predictions_encoded
        )

        accuracy = accuracy_score(
            y_test_original,
            predictions_original,
        )

        f1_macro = f1_score(
            y_test_original,
            predictions_original,
            average="macro",
            zero_division=0,
        )

        f1_weighted = f1_score(
            y_test_original,
            predictions_original,
            average="weighted",
            zero_division=0,
        )

        report = classification_report(
            y_test_original,
            predictions_original,
            labels=[-1, 0, 1],
            target_names=[
                "SELL",
                "NO_TRADE",
                "BUY",
            ],
            zero_division=0,
        )

        logger.info(
            f"Loss : {test_loss:.4f}"
        )

        logger.info(
            f"Accuracy : {accuracy:.4f}"
        )

        logger.info(
            f"F1 macro : {f1_macro:.4f}"
        )

        logger.info(
            f"F1 weighted : {f1_weighted:.4f}"
        )

        logger.info(
            "\n" + report
        )

        mlflow_manager.log_params({
            "model_type": "LSTM",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "sequence_length": SEQUENCE_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "features_count": len(feature_names),
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "class_weight": True,
            "chronological_split": True,
        })

        mlflow_manager.log_metrics({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        })

        model.save(
            model_path
        )

        joblib.dump(
            scaler,
            scaler_path,
        )

        joblib.dump(
            feature_names,
            features_path,
        )

        metadata = {
            "model_type": "LSTM",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_count": len(feature_names),
            "target_mapping": {
                "-1": 0,
                "0": 1,
                "1": 2,
            },
        }

        metrics = {
            "loss": float(test_loss),
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        metrics_path.write_text(
            json.dumps(
                metrics,
                indent=4,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        classification_report_path.write_text(
            report,
            encoding="utf-8",
        )

    logger.info(
        f"Modèle sauvegardé : {model_path}"
    )

    logger.info(
        f"Scaler sauvegardé : {scaler_path}"
    )

    logger.info(
        f"Features sauvegardées : {features_path}"
    )

    logger.info(
        "Entraînement LSTM V3 terminé."
    )


def parse_arguments() -> argparse.Namespace:
    """
    Lit les paramètres PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner LSTM V3 pour un symbole "
            "et un timeframe."
        )
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à entraîner.",
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=[
            "1m",
            "5m",
            "15m",
            "30m",
            "1h",
            "4h",
            "1d",
        ],
        help="Timeframe à entraîner.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    train_lstm_v3(
        symbol=arguments.symbol,
        timeframe=arguments.timeframe,
    )
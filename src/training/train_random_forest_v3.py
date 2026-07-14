"""
Entraînement Random Forest V3 par symbole et timeframe.

Le script utilise :
- dataset_analysis_v3
- BaseTrainer
- ChronologicalSplitter
- MLflow
- sauvegarde du modèle et des métadonnées

Exemples :

python -m src.training.train_random_forest_v3 --symbol GC --timeframe 5m

python -m src.training.train_random_forest_v3 --symbol GC --timeframe 15m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.data.chronological_splitter import ChronologicalSplitter
from src.mlops.mlflow_manager import MLflowManager
from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

N_ESTIMATORS = 300
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2


def train_random_forest_v3(
    symbol: str,
    timeframe: str,
) -> None:
    """
    Entraîne un modèle Random Forest V3.
    """

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="random_forest",
    )

    df = trainer.load_analysis_dataset()

    X, y = trainer.prepare_tabular_features(df)

    splitter = ChronologicalSplitter(
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    split = splitter.split(
        X=X,
        y=y,
    )

    X_train = split.X_train
    X_validation = split.X_validation
    X_test = split.X_test

    y_train = split.y_train
    y_validation = split.y_validation
    y_test = split.y_test

    logger.info(f"Symbole : {trainer.symbol}")
    logger.info(f"Timeframe : {trainer.timeframe}")
    logger.info(f"Nombre de features : {X.shape[1]}")
    logger.info(f"X_train : {X_train.shape}")
    logger.info(f"X_validation : {X_validation.shape}")
    logger.info(f"X_test : {X_test.shape}")
    logger.info(
        f"Distribution train :\n{y_train.value_counts()}"
    )
    logger.info(
        f"Distribution validation :\n{y_validation.value_counts()}"
    )
    logger.info(
        f"Distribution test :\n{y_test.value_counts()}"
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=None,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    run_name = (
        f"random_forest_v3_"
        f"{trainer.symbol}_"
        f"{trainer.timeframe}"
    )

    mlflow_manager = MLflowManager(
        experiment_name=(
            f"AI Trading System V3 - {trainer.symbol}"
        )
    )

    with mlflow_manager.start_run(
        run_name=run_name
    ):
        logger.info(
            "Début de l'entraînement Random Forest V3."
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_predictions = model.predict(
            X_validation
        )

        test_predictions = model.predict(
            X_test
        )

        validation_accuracy = accuracy_score(
            y_validation,
            validation_predictions,
        )

        validation_f1_macro = f1_score(
            y_validation,
            validation_predictions,
            average="macro",
            zero_division=0,
        )

        validation_f1_weighted = f1_score(
            y_validation,
            validation_predictions,
            average="weighted",
            zero_division=0,
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions,
        )

        test_f1_macro = f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0,
        )

        test_f1_weighted = f1_score(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0,
        )

        validation_report = classification_report(
            y_validation,
            validation_predictions,
            labels=[-1, 0, 1],
            target_names=[
                "SELL",
                "NO_TRADE",
                "BUY",
            ],
            zero_division=0,
        )

        test_report = classification_report(
            y_test,
            test_predictions,
            labels=[-1, 0, 1],
            target_names=[
                "SELL",
                "NO_TRADE",
                "BUY",
            ],
            zero_division=0,
        )

        logger.info(
            f"Validation Accuracy : "
            f"{validation_accuracy:.4f}"
        )

        logger.info(
            f"Validation F1 macro : "
            f"{validation_f1_macro:.4f}"
        )

        logger.info(
            f"Validation F1 weighted : "
            f"{validation_f1_weighted:.4f}"
        )

        logger.info(
            "\nRapport validation :\n"
            + validation_report
        )

        logger.info(
            f"Test Accuracy : {test_accuracy:.4f}"
        )

        logger.info(
            f"Test F1 macro : {test_f1_macro:.4f}"
        )

        logger.info(
            f"Test F1 weighted : "
            f"{test_f1_weighted:.4f}"
        )

        logger.info(
            "\nRapport test :\n"
            + test_report
        )

        mlflow_manager.log_params({
            "model_type": "RandomForestClassifier",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "dataset": str(trainer.dataset_path),
            "n_estimators": N_ESTIMATORS,
            "max_depth": "None",
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": "balanced",
            "features_count": X_train.shape[1],
            "split_type": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
        })

        mlflow_manager.log_metrics({
            "validation_accuracy": float(
                validation_accuracy
            ),
            "validation_f1_macro": float(
                validation_f1_macro
            ),
            "validation_f1_weighted": float(
                validation_f1_weighted
            ),
            "test_accuracy": float(
                test_accuracy
            ),
            "test_f1_macro": float(
                test_f1_macro
            ),
            "test_f1_weighted": float(
                test_f1_weighted
            ),
        })

        mlflow_manager.log_sklearn_model(
            model=model,
            artifact_name=run_name,
        )

        model_path = (
            trainer.model_dir
            / "random_forest_v3.pkl"
        )

        metadata_path = (
            trainer.model_dir
            / "metadata_v3.json"
        )

        metrics_path = (
            trainer.report_dir
            / "metrics_v3.json"
        )

        validation_report_path = (
            trainer.report_dir
            / "classification_report_validation_v3.txt"
        )

        test_report_path = (
            trainer.report_dir
            / "classification_report_test_v3.txt"
        )

        model_bundle = {
            "model": model,
            "features": X_train.columns.tolist(),
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "split_type": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "metrics": {
                "validation_accuracy": validation_accuracy,
                "validation_f1_macro": validation_f1_macro,
                "validation_f1_weighted": (
                    validation_f1_weighted
                ),
                "test_accuracy": test_accuracy,
                "test_f1_macro": test_f1_macro,
                "test_f1_weighted": (
                    test_f1_weighted
                ),
            },
        }

        joblib.dump(
            model_bundle,
            model_path,
        )

        metadata = {
            "model_type": "RandomForestClassifier",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "feature_count": X_train.shape[1],
            "features": X_train.columns.tolist(),
            "split_type": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
        }

        metrics = {
            "validation_accuracy": float(
                validation_accuracy
            ),
            "validation_f1_macro": float(
                validation_f1_macro
            ),
            "validation_f1_weighted": float(
                validation_f1_weighted
            ),
            "test_accuracy": float(
                test_accuracy
            ),
            "test_f1_macro": float(
                test_f1_macro
            ),
            "test_f1_weighted": float(
                test_f1_weighted
            ),
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

        validation_report_path.write_text(
            validation_report,
            encoding="utf-8",
        )

        test_report_path.write_text(
            test_report,
            encoding="utf-8",
        )

        logger.info(
            f"Modèle sauvegardé : {model_path}"
        )

        logger.info(
            f"Métadonnées sauvegardées : {metadata_path}"
        )

        logger.info(
            f"Métriques sauvegardées : {metrics_path}"
        )

    logger.info(
        "Entraînement Random Forest V3 terminé."
    )


def parse_arguments() -> argparse.Namespace:
    """
    Lit les paramètres de la ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner Random Forest V3 "
            "avec un split chronologique."
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

    train_random_forest_v3(
        symbol=arguments.symbol,
        timeframe=arguments.timeframe,
    )
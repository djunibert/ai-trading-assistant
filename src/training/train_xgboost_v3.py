"""
Entraînement XGBoost V3 avec découpage chronologique.

Répartition :
- 70 % entraînement
- 15 % validation
- 15 % test

Exemples :

python -m src.training.train_xgboost_v3 --symbol GC --timeframe 5m
python -m src.training.train_xgboost_v3 --symbol GC --timeframe 15m
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from xgboost import XGBClassifier

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
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.80
COLSAMPLE_BYTREE = 0.80


TARGET_MAPPING = {
    -1: 0,
    0: 1,
    1: 2,
}

INVERSE_TARGET_MAPPING = {
    0: -1,
    1: 0,
    2: 1,
}


def encode_target(
    target: pd.Series,
) -> pd.Series:
    """
    Convertit :
    SELL -1 vers 0
    NO_TRADE 0 vers 1
    BUY 1 vers 2
    """

    encoded = target.map(TARGET_MAPPING)

    if encoded.isna().any():
        invalid_values = target[
            encoded.isna()
        ].unique()

        raise ValueError(
            f"Classes target invalides : {invalid_values}"
        )

    return encoded.astype(int)


def decode_target(
    target: pd.Series,
) -> pd.Series:
    """
    Reconvertit les classes XGBoost vers -1, 0 et 1.
    """

    decoded = target.map(INVERSE_TARGET_MAPPING)

    if decoded.isna().any():
        invalid_values = target[
            decoded.isna()
        ].unique()

        raise ValueError(
            f"Classes XGBoost invalides : {invalid_values}"
        )

    return decoded.astype(int)


def train_xgboost_v3(
    symbol: str,
    timeframe: str,
) -> None:
    """
    Entraîne XGBoost V3 pour un symbole et un timeframe.
    """

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="xgboost",
    )

    df = trainer.load_analysis_dataset()

    X, y_original = trainer.prepare_tabular_features(df)

    splitter = ChronologicalSplitter(
        train_ratio=TRAIN_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )

    split = splitter.split(
        X=X,
        y=y_original,
    )

    X_train = split.X_train
    X_validation = split.X_validation
    X_test = split.X_test

    y_train_original = split.y_train
    y_validation_original = split.y_validation
    y_test_original = split.y_test

    y_train = encode_target(y_train_original)
    y_validation = encode_target(y_validation_original)
    #y_test = encode_target(y_test_original)

    logger.info(f"Symbole : {trainer.symbol}")
    logger.info(f"Timeframe : {trainer.timeframe}")
    logger.info(f"Nombre de features : {X.shape[1]}")

    logger.info(f"X_train : {X_train.shape}")
    logger.info(f"X_validation : {X_validation.shape}")
    logger.info(f"X_test : {X_test.shape}")

    logger.info(
        f"Distribution train :\n"
        f"{y_train_original.value_counts()}"
    )

    logger.info(
        f"Distribution validation :\n"
        f"{y_validation_original.value_counts()}"
    )

    logger.info(
        f"Distribution test :\n"
        f"{y_test_original.value_counts()}"
    )

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    run_name = (
        f"xgboost_v3_"
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
            "Début de l'entraînement XGBoost V3."
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        validation_predictions_encoded = pd.Series(
            model.predict(X_validation),
            index=X_validation.index,
        )

        test_predictions_encoded = pd.Series(
            model.predict(X_test),
            index=X_test.index,
        )

        validation_predictions = decode_target(
            validation_predictions_encoded
        )

        test_predictions = decode_target(
            test_predictions_encoded
        )

        validation_accuracy = accuracy_score(
            y_validation_original,
            validation_predictions,
        )

        validation_f1_macro = f1_score(
            y_validation_original,
            validation_predictions,
            average="macro",
            zero_division=0,
        )

        validation_f1_weighted = f1_score(
            y_validation_original,
            validation_predictions,
            average="weighted",
            zero_division=0,
        )

        test_accuracy = accuracy_score(
            y_test_original,
            test_predictions,
        )

        test_f1_macro = f1_score(
            y_test_original,
            test_predictions,
            average="macro",
            zero_division=0,
        )

        test_f1_weighted = f1_score(
            y_test_original,
            test_predictions,
            average="weighted",
            zero_division=0,
        )

        validation_report = classification_report(
            y_validation_original,
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
            y_test_original,
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
            "model_type": "XGBClassifier",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "dataset": str(trainer.dataset_path),
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
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

        mlflow_manager.log_xgboost_model(
            model=model,
            artifact_name=run_name,
        )

        model_path = (
            trainer.model_dir
            / "xgboost_v3.pkl"
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
            "target_mapping": TARGET_MAPPING,
            "inverse_target_mapping": (
                INVERSE_TARGET_MAPPING
            ),
            "split_type": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "metrics": {
                "validation_accuracy": (
                    validation_accuracy
                ),
                "validation_f1_macro": (
                    validation_f1_macro
                ),
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
            "model_type": "XGBClassifier",
            "symbol": trainer.symbol,
            "timeframe": trainer.timeframe,
            "feature_count": X_train.shape[1],
            "features": X_train.columns.tolist(),
            "split_type": "chronological",
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "random_state": RANDOM_STATE,
            "target_mapping": TARGET_MAPPING,
            "inverse_target_mapping": (
                INVERSE_TARGET_MAPPING
            ),
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
            f"Métadonnées sauvegardées : "
            f"{metadata_path}"
        )

        logger.info(
            f"Métriques sauvegardées : "
            f"{metrics_path}"
        )

    logger.info(
        "Entraînement XGBoost V3 terminé."
    )


def parse_arguments() -> argparse.Namespace:
    """
    Lit les arguments de la ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner XGBoost V3 "
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

    train_xgboost_v3(
        symbol=arguments.symbol,
        timeframe=arguments.timeframe,
    )
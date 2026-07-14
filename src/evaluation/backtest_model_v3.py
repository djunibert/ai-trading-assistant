"""
Backtest des modèles V3 par symbole et timeframe.

Modèles pris en charge :
- Random Forest
- XGBoost
- LSTM

Exemples :

python -m src.evaluation.backtest_model_v3 --symbol GC --timeframe 5m --model random_forest

python -m src.evaluation.backtest_model_v3 --symbol GC --timeframe 5m --model xgboost

python -m src.evaluation.backtest_model_v3 --symbol GC --timeframe 15m --model lstm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger


logger = get_logger(__name__)


RANDOM_STATE = 42
TEST_SIZE = 0.20

LSTM_TRAIN_RATIO = 0.70
LSTM_VALIDATION_RATIO = 0.15

VALID_TIMEFRAMES = {
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
}

VALID_MODELS = {
    "random_forest",
    "xgboost",
    "lstm",
}


def get_analysis_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """Retourne le chemin du dataset Analysis V3."""

    return (
        Path("data/final/analysis")
        / symbol
        / f"dataset_analysis_v3_{symbol}_{timeframe}.csv"
    )


def get_ml_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """Retourne le chemin du dataset ML V3."""

    return (
        Path("data/final/ml")
        / symbol
        / f"dataset_ml_v3_{symbol}_{timeframe}.csv"
    )


def get_model_directory(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> Path:
    """Retourne le dossier du modèle."""

    return (
        Path("models")
        / symbol
        / timeframe
        / model_name
    )


def get_model_path(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> Path:
    """Retourne le chemin du modèle."""

    model_directory = get_model_directory(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    if model_name == "random_forest":
        return model_directory / "random_forest_v3.pkl"

    if model_name == "xgboost":
        return model_directory / "xgboost_v3.pkl"

    if model_name == "lstm":
        return model_directory / "lstm_v3.keras"

    raise ValueError(
        f"Modèle non pris en charge : {model_name}"
    )


def get_output_directory(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> Path:
    """Retourne le dossier des rapports."""

    return (
        Path("reports/model_backtesting")
        / symbol
        / timeframe
        / model_name
    )


def load_analysis_dataset(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Charge le dataset Analysis."""

    dataset_path = get_analysis_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset Analysis introuvable : {dataset_path}"
        )

    logger.info(
        f"Chargement du dataset Analysis : {dataset_path}"
    )

    df = pd.read_csv(dataset_path)

    required_columns = {
        "target",
        "future_return_1",
    }

    missing_columns = required_columns.difference(
        df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce",
            utc=True,
        )

    return df


def load_ml_dataset(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Charge le dataset ML des modèles tabulaires."""

    dataset_path = get_ml_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset ML introuvable : {dataset_path}"
        )

    logger.info(
        f"Chargement du dataset ML : {dataset_path}"
    )

    df = pd.read_csv(dataset_path)

    if "target" not in df.columns:
        raise ValueError(
            "La colonne target est absente du dataset ML."
        )

    return df


def load_tabular_model_bundle(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any]:
    """Charge Random Forest ou XGBoost."""

    model_path = get_model_path(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Le fichier modèle doit contenir un dictionnaire."
        )

    if "model" not in bundle or "features" not in bundle:
        raise ValueError(
            "Le bundle doit contenir model et features."
        )

    return bundle


def load_lstm_bundle(
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    """Charge le modèle LSTM, le scaler et les features."""

    model_directory = get_model_directory(
        symbol=symbol,
        timeframe=timeframe,
        model_name="lstm",
    )

    model_path = model_directory / "lstm_v3.keras"
    scaler_path = model_directory / "scaler_v3.pkl"
    features_path = model_directory / "features_v3.pkl"
    metadata_path = model_directory / "metadata_v3.json"

    required_paths = [
        model_path,
        scaler_path,
        features_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"Fichiers LSTM manquants : {missing_paths}"
        )

    sequence_length = 32

    if metadata_path.exists():
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        sequence_length = int(
            metadata.get(
                "sequence_length",
                sequence_length,
            )
        )

    return {
        "model": load_model(model_path),
        "scaler": joblib.load(scaler_path),
        "features": joblib.load(features_path),
        "sequence_length": sequence_length,
    }


def create_tabular_test_data(
    ml_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recrée le split utilisé par RF et XGBoost."""

    X = ml_df.drop(
        columns=[
            "target",
            "future_return_1",
        ],
        errors="ignore",
    )

    y = ml_df["target"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def decode_xgboost_predictions(
    predictions: np.ndarray,
    inverse_mapping: dict | None,
) -> np.ndarray:
    """Convertit 0, 1, 2 vers -1, 0, 1."""

    mapping = inverse_mapping or {
        0: -1,
        1: 0,
        2: 1,
    }

    decoded = pd.Series(
        predictions
    ).map(mapping)

    if decoded.isna().any():
        raise ValueError(
            "Certaines prédictions XGBoost sont invalides."
        )

    return decoded.to_numpy(
        dtype=int
    )


def predict_tabular_model(
    symbol: str,
    timeframe: str,
    model_name: str,
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produit le DataFrame de prédictions RF ou XGBoost."""

    ml_df = load_ml_dataset(
        symbol=symbol,
        timeframe=timeframe,
    )

    if len(ml_df) != len(analysis_df):
        raise ValueError(
            "Les datasets ML et Analysis "
            "n'ont pas le même nombre de lignes."
        )

    X_test, y_test = create_tabular_test_data(
        ml_df
    )

    bundle = load_tabular_model_bundle(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    model = bundle["model"]
    feature_names = bundle["features"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features manquantes : {missing_features}"
        )

    X_model = X_test[feature_names]

    predictions = model.predict(
        X_model
    )

    if model_name == "xgboost":
        predictions = decode_xgboost_predictions(
            predictions=predictions,
            inverse_mapping=bundle.get(
                "inverse_target_mapping"
            ),
        )

    selected_analysis = analysis_df.loc[
        X_test.index
    ].copy()

    result_df = pd.DataFrame(
        index=X_test.index
    )

    if "datetime" in selected_analysis.columns:
        result_df["datetime"] = (
            selected_analysis["datetime"]
        )

    if "close" in selected_analysis.columns:
        result_df["close"] = (
            selected_analysis["close"]
        )

    result_df["actual_target"] = (
        y_test.loc[X_test.index]
        .astype(int)
    )

    result_df["prediction"] = predictions

    result_df["future_return"] = (
        selected_analysis["future_return_1"]
        .astype(float)
    )

    return result_df


def encode_lstm_target(
    values: np.ndarray,
) -> np.ndarray:
    """Encode -1, 0, 1 vers 0, 1, 2."""

    mapping = {
        -1: 0,
        0: 1,
        1: 2,
    }

    return np.asarray(
        [
            mapping[int(value)]
            for value in values
        ],
        dtype=np.int32,
    )


def decode_lstm_target(
    values: np.ndarray,
) -> np.ndarray:
    """Décode 0, 1, 2 vers -1, 0, 1."""

    mapping = {
        0: -1,
        1: 0,
        2: 1,
    }

    return np.asarray(
        [
            mapping[int(value)]
            for value in values
        ],
        dtype=np.int32,
    )


def create_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    source_indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les séquences LSTM."""

    X_sequences = []
    y_sequences = []
    sequence_indices = []

    for index in range(
        sequence_length,
        len(X),
    ):
        X_sequences.append(
            X[index - sequence_length:index]
        )

        y_sequences.append(
            y[index]
        )

        sequence_indices.append(
            source_indices[index]
        )

    return (
        np.asarray(
            X_sequences,
            dtype=np.float32,
        ),
        np.asarray(
            y_sequences,
            dtype=np.int32,
        ),
        np.asarray(
            sequence_indices,
            dtype=int,
        ),
    )


def predict_lstm_model(
    symbol: str,
    timeframe: str,
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produit les prédictions LSTM sur le test chronologique."""

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="lstm",
    )

    X_df, y_series = trainer.prepare_tabular_features(
        analysis_df
    )

    bundle = load_lstm_bundle(
        symbol=symbol,
        timeframe=timeframe,
    )

    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["features"]
    sequence_length = bundle["sequence_length"]

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features LSTM manquantes : {missing_features}"
        )

    X_df = X_df[feature_names]

    X = X_df.to_numpy(
        dtype=np.float32
    )

    y = encode_lstm_target(
        y_series.to_numpy()
    )

    source_indices = np.arange(
        len(analysis_df)
    )

    total_rows = len(X)

    train_end = int(
        total_rows * LSTM_TRAIN_RATIO
    )

    validation_end = int(
        total_rows
        * (
            LSTM_TRAIN_RATIO
            + LSTM_VALIDATION_RATIO
        )
    )

    X_test_raw = X[validation_end:]
    y_test_raw = y[validation_end:]
    test_indices = source_indices[
        validation_end:
    ]

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    (
        X_test,
        y_test,
        sequence_indices,
    ) = create_lstm_sequences(
        X=X_test_scaled,
        y=y_test_raw,
        source_indices=test_indices,
        sequence_length=sequence_length,
    )

    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    predictions_encoded = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = decode_lstm_target(
        predictions_encoded
    )

    actual_targets = decode_lstm_target(
        y_test
    )

    selected_analysis = analysis_df.iloc[
        sequence_indices
    ].copy()

    result_df = pd.DataFrame(
        index=sequence_indices
    )

    if "datetime" in selected_analysis.columns:
        result_df["datetime"] = (
            selected_analysis["datetime"]
            .to_numpy()
        )

    if "close" in selected_analysis.columns:
        result_df["close"] = (
            selected_analysis["close"]
            .to_numpy()
        )

    result_df["actual_target"] = (
        actual_targets
    )

    result_df["prediction"] = (
        predictions
    )

    result_df["future_return"] = (
        selected_analysis["future_return_1"]
        .astype(float)
        .to_numpy()
    )

    return result_df


def complete_backtest_dataframe(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ajoute les rendements et résultats du backtest."""

    result_df = result_df.copy()

    result_df["strategy_return"] = (
        result_df["prediction"]
        * result_df["future_return"]
    )

    result_df["trade_taken"] = (
        result_df["prediction"] != 0
    )

    result_df["is_correct"] = (
        result_df["prediction"]
        == result_df["actual_target"]
    )

    result_df["is_win"] = (
        result_df["strategy_return"] > 0
    )

    return result_df


def empty_metrics() -> dict[str, Any]:
    """Retourne des métriques vides."""

    return {
        "total_observations": 0,
        "total_trades": 0,
        "buy_trades": 0,
        "sell_trades": 0,
        "wins": 0,
        "losses": 0,
        "breakeven": 0,
        "win_rate_percent": 0.0,
        "signal_accuracy_percent": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "net_return_sum": 0.0,
        "compounded_return": 0.0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
    }


def calculate_backtest_metrics(
    result_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calcule les principales métriques de trading."""

    trades_df = result_df[
        result_df["trade_taken"]
    ].copy()

    if trades_df.empty:
        return empty_metrics(), trades_df

    total_trades = len(trades_df)

    wins = int(
        (
            trades_df["strategy_return"] > 0
        ).sum()
    )

    losses = int(
        (
            trades_df["strategy_return"] < 0
        ).sum()
    )

    breakeven = int(
        (
            trades_df["strategy_return"] == 0
        ).sum()
    )

    gross_profit = float(
        trades_df.loc[
            trades_df["strategy_return"] > 0,
            "strategy_return",
        ].sum()
    )

    gross_loss = abs(
        float(
            trades_df.loc[
                trades_df["strategy_return"] < 0,
                "strategy_return",
            ].sum()
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0.0
    )

    expectancy = float(
        trades_df["strategy_return"].mean()
    )

    average_win = (
        float(
            trades_df.loc[
                trades_df["strategy_return"] > 0,
                "strategy_return",
            ].mean()
        )
        if wins > 0
        else 0.0
    )

    average_loss = (
        float(
            trades_df.loc[
                trades_df["strategy_return"] < 0,
                "strategy_return",
            ].mean()
        )
        if losses > 0
        else 0.0
    )

    trades_df["equity_curve"] = (
        1.0
        + trades_df["strategy_return"]
    ).cumprod()

    trades_df["running_max"] = (
        trades_df["equity_curve"]
        .cummax()
    )

    trades_df["drawdown"] = (
        trades_df["equity_curve"]
        / trades_df["running_max"]
        - 1.0
    )

    max_drawdown = float(
        trades_df["drawdown"].min()
    )

    compounded_return = float(
        trades_df["equity_curve"].iloc[-1]
        - 1.0
    )

    return_std = float(
        trades_df["strategy_return"].std()
    )

    sharpe_ratio = (
        expectancy
        / return_std
        * np.sqrt(252)
        if return_std > 0
        else 0.0
    )

    metrics = {
        "total_observations": int(
            len(result_df)
        ),
        "total_trades": int(total_trades),
        "buy_trades": int(
            (
                trades_df["prediction"] == 1
            ).sum()
        ),
        "sell_trades": int(
            (
                trades_df["prediction"] == -1
            ).sum()
        ),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_percent": round(
            wins / total_trades * 100,
            2,
        ),
        "signal_accuracy_percent": round(
            result_df["is_correct"].mean()
            * 100,
            2,
        ),
        "gross_profit": round(
            gross_profit,
            8,
        ),
        "gross_loss": round(
            gross_loss,
            8,
        ),
        "net_return_sum": round(
            float(
                trades_df[
                    "strategy_return"
                ].sum()
            ),
            8,
        ),
        "compounded_return": round(
            compounded_return,
            8,
        ),
        "profit_factor": round(
            profit_factor,
            4,
        ),
        "expectancy": round(
            expectancy,
            8,
        ),
        "average_win": round(
            average_win,
            8,
        ),
        "average_loss": round(
            average_loss,
            8,
        ),
        "max_drawdown": round(
            max_drawdown,
            8,
        ),
        "sharpe_ratio": round(
            sharpe_ratio,
            4,
        ),
    }

    return metrics, trades_df


def save_reports(
    symbol: str,
    timeframe: str,
    model_name: str,
    metrics: dict[str, Any],
    result_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Sauvegarde les rapports du backtest."""

    output_directory = get_output_directory(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_directory
        / "backtest_metrics.csv"
    )

    predictions_path = (
        output_directory
        / "backtest_predictions.csv"
    )

    trades_path = (
        output_directory
        / "backtest_trades.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    result_df.to_csv(
        predictions_path,
        index=True,
        index_label="source_index",
        encoding="utf-8-sig",
    )

    trades_df.to_csv(
        trades_path,
        index=True,
        index_label="source_index",
        encoding="utf-8-sig",
    )

    return (
        metrics_path,
        predictions_path,
        trades_path,
    )


def log_backtest_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    """Affiche les métriques du backtest."""

    logger.info("=" * 70)
    logger.info("RÉSULTATS DU BACKTEST")
    logger.info("=" * 70)

    logger.info(f"Modèle : {model_name}")
    logger.info(f"Symbole : {symbol}")
    logger.info(f"Timeframe : {timeframe}")

    logger.info(
        f"Observations : "
        f"{metrics['total_observations']}"
    )

    logger.info(
        f"Trades : {metrics['total_trades']}"
    )

    logger.info(
        f"Win rate : "
        f"{metrics['win_rate_percent']} %"
    )

    logger.info(
        f"Accuracy : "
        f"{metrics['signal_accuracy_percent']} %"
    )

    logger.info(
        f"Profit factor : "
        f"{metrics['profit_factor']}"
    )

    logger.info(
        f"Rendement composé : "
        f"{metrics['compounded_return']}"
    )

    logger.info(
        f"Max drawdown : "
        f"{metrics['max_drawdown']}"
    )

    logger.info(
        f"Sharpe ratio : "
        f"{metrics['sharpe_ratio']}"
    )


def backtest_model_v3(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any]:
    """Exécute le backtest complet."""

    symbol = symbol.upper()
    timeframe = timeframe.lower()
    model_name = model_name.lower()

    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Timeframe invalide : {timeframe}"
        )

    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Modèle invalide : {model_name}"
        )

    analysis_df = load_analysis_dataset(
        symbol=symbol,
        timeframe=timeframe,
    )

    if model_name == "lstm":
        result_df = predict_lstm_model(
            symbol=symbol,
            timeframe=timeframe,
            analysis_df=analysis_df,
        )

    else:
        result_df = predict_tabular_model(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
            analysis_df=analysis_df,
        )

    result_df = complete_backtest_dataframe(
        result_df
    )

    metrics, trades_df = calculate_backtest_metrics(
        result_df=result_df,
    )

    (
        metrics_path,
        predictions_path,
        trades_path,
    ) = save_reports(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        metrics=metrics,
        result_df=result_df,
        trades_df=trades_df,
    )

    log_backtest_metrics(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        metrics=metrics,
    )

    logger.info(
        f"Métriques : {metrics_path}"
    )

    logger.info(
        f"Prédictions : {predictions_path}"
    )

    logger.info(
        f"Trades : {trades_path}"
    )

    return {
        "metrics": metrics,
        "predictions": result_df,
        "trades": trades_df,
    }


def parse_arguments() -> argparse.Namespace:
    """Lit les paramètres PowerShell."""

    parser = argparse.ArgumentParser(
        description=(
            "Backtester un modèle V3 "
            "pour un symbole et un timeframe."
        )
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à tester.",
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=sorted(VALID_TIMEFRAMES),
        help="Timeframe à tester.",
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(VALID_MODELS),
        help="Modèle à tester.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    backtest_model_v3(
        symbol=arguments.symbol,
        timeframe=arguments.timeframe,
        model_name=arguments.model,
    )
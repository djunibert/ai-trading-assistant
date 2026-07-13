"""
Backtest des modèles V3 par symbole et timeframe.

Modèles pris en charge :
- Random Forest
- XGBoost

Exemples PowerShell :

python -m src.evaluation.backtest_model_v3 `
    --symbol GC `
    --timeframe 5m `
    --model random_forest

python -m src.evaluation.backtest_model_v3 `
    --symbol GC `
    --timeframe 15m `
    --model xgboost

Principe du backtest :

SELL     = -1
NO_TRADE = 0
BUY      = 1

Rendement de la stratégie :

prediction * future_return_1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


logger = get_logger(__name__)


# ==========================================================
# Configuration
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

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
}


# ==========================================================
# Chemins
# ==========================================================

def get_ml_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Retourne le chemin du dataset numérique utilisé
    pour entraîner le modèle.
    """

    return (
        Path("data/final/ml")
        / symbol
        / f"dataset_ml_v3_{symbol}_{timeframe}.csv"
    )


def get_analysis_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Retourne le chemin du dataset d'analyse.

    Ce dataset contient notamment :
    - datetime
    - future_return_1
    - symbol
    - timeframe
    """

    return (
        Path("data/final/analysis")
        / symbol
        / f"dataset_analysis_v3_{symbol}_{timeframe}.csv"
    )


def get_model_path(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> Path:
    """
    Retourne le chemin du modèle demandé.
    """

    if model_name == "random_forest":
        return (
            Path("models")
            / symbol
            / timeframe
            / "random_forest"
            / "random_forest_v3.pkl"
        )

    if model_name == "xgboost":
        return (
            Path("models")
            / symbol
            / timeframe
            / "xgboost"
            / "xgboost_v3.pkl"
        )

    raise ValueError(
        f"Modèle non pris en charge : {model_name}"
    )


def get_output_directory(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> Path:
    """
    Retourne le dossier des rapports du backtest.
    """

    return (
        Path("reports/model_backtesting")
        / symbol
        / timeframe
        / model_name
    )


# ==========================================================
# Chargement des données
# ==========================================================

def load_datasets(
    symbol: str,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge les datasets ML et Analysis.
    """

    ml_path = get_ml_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    analysis_path = get_analysis_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not ml_path.exists():
        raise FileNotFoundError(
            f"Dataset ML introuvable : {ml_path}"
        )

    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Dataset Analysis introuvable : {analysis_path}"
        )

    logger.info(f"Dataset ML : {ml_path}")
    logger.info(f"Dataset Analysis : {analysis_path}")

    ml_df = pd.read_csv(ml_path)
    analysis_df = pd.read_csv(analysis_path)

    validate_datasets(
        ml_df=ml_df,
        analysis_df=analysis_df,
    )

    return ml_df, analysis_df


def validate_datasets(
    ml_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> None:
    """
    Vérifie la cohérence des datasets.
    """

    if "target" not in ml_df.columns:
        raise ValueError(
            "La colonne target est absente du dataset ML."
        )

    if "future_return_1" not in analysis_df.columns:
        raise ValueError(
            "La colonne future_return_1 est absente "
            "du dataset Analysis."
        )

    if len(ml_df) != len(analysis_df):
        raise ValueError(
            "Les datasets ML et Analysis n'ont pas "
            "le même nombre de lignes. "
            f"ML={len(ml_df)}, Analysis={len(analysis_df)}"
        )

    if "datetime" in analysis_df.columns:
        analysis_df["datetime"] = pd.to_datetime(
            analysis_df["datetime"],
            errors="coerce",
            utc=True,
        )


# ==========================================================
# Jeu de test
# ==========================================================

def create_test_split(
    ml_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Recrée exactement le même jeu de test que celui utilisé
    pendant l'entraînement des modèles V3.

    Important :
    le random_state, test_size et stratify doivent rester
    identiques aux scripts d'entraînement.
    """

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


# ==========================================================
# Chargement du modèle
# ==========================================================

def load_model_bundle(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Charge le modèle et ses métadonnées.
    """

    model_path = get_model_path(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {model_path}"
        )

    logger.info(f"Chargement du modèle : {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise TypeError(
            "Le fichier modèle doit contenir un dictionnaire."
        )

    required_keys = {
        "model",
        "features",
    }

    missing_keys = required_keys.difference(
        bundle.keys()
    )

    if missing_keys:
        raise ValueError(
            f"Clés manquantes dans le modèle : {missing_keys}"
        )

    return bundle


# ==========================================================
# Prédictions
# ==========================================================

def prepare_model_features(
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Place les features dans le même ordre que pendant
    l'entraînement.
    """

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            "Features manquantes dans le dataset : "
            f"{missing_features}"
        )

    return X_test[feature_names].copy()


def predict_classes(
    model_name: str,
    bundle: dict[str, Any],
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Produit des prédictions dans le format commun :

    SELL     = -1
    NO_TRADE = 0
    BUY      = 1
    """

    model = bundle["model"]

    raw_predictions = model.predict(
        X_test
    )

    if model_name == "random_forest":
        return np.asarray(
            raw_predictions,
            dtype=int,
        )

    if model_name == "xgboost":
        inverse_mapping = bundle.get(
            "inverse_target_mapping",
            {
                0: -1,
                1: 0,
                2: 1,
            },
        )

        decoded_predictions = pd.Series(
            raw_predictions,
            index=X_test.index,
        ).map(inverse_mapping)

        if decoded_predictions.isna().any():
            raise ValueError(
                "Certaines prédictions XGBoost ne peuvent "
                "pas être reconverties."
            )

        return decoded_predictions.to_numpy(
            dtype=int
        )

    raise ValueError(
        f"Modèle non pris en charge : {model_name}"
    )


# ==========================================================
# Construction des résultats
# ==========================================================

def build_backtest_dataframe(
    analysis_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Construit le DataFrame de backtest.

    Les indices de X_test permettent de retrouver exactement
    les mêmes lignes dans le dataset Analysis.
    """

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
            .astype(float)
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

    # BUY :
    # +1 × rendement futur
    #
    # SELL :
    # -1 × rendement futur
    #
    # NO_TRADE :
    # 0 × rendement futur
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

    return result_df.sort_index()


# ==========================================================
# Métriques
# ==========================================================

def calculate_backtest_metrics(
    result_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Calcule les métriques principales de trading.
    """

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

    net_return_sum = float(
        trades_df["strategy_return"].sum()
    )

    expectancy = float(
        trades_df["strategy_return"].mean()
    )

    average_win = float(
        trades_df.loc[
            trades_df["strategy_return"] > 0,
            "strategy_return",
        ].mean()
    ) if wins > 0 else 0.0

    average_loss = float(
        trades_df.loc[
            trades_df["strategy_return"] < 0,
            "strategy_return",
        ].mean()
    ) if losses > 0 else 0.0

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
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

    win_rate = (
        wins / total_trades * 100
        if total_trades > 0
        else 0.0
    )

    signal_accuracy = float(
        result_df["is_correct"].mean()
        * 100
    )

    buy_trades = int(
        (
            trades_df["prediction"] == 1
        ).sum()
    )

    sell_trades = int(
        (
            trades_df["prediction"] == -1
        ).sum()
    )

    metrics = {
        "total_observations": int(len(result_df)),
        "total_trades": int(total_trades),
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_percent": round(win_rate, 2),
        "signal_accuracy_percent": round(
            signal_accuracy,
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
            net_return_sum,
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


def empty_metrics() -> dict[str, Any]:
    """
    Retourne des métriques vides.
    """

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


# ==========================================================
# Sauvegarde
# ==========================================================

def save_reports(
    symbol: str,
    timeframe: str,
    model_name: str,
    metrics: dict[str, Any],
    result_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """
    Sauvegarde :
    - les métriques
    - toutes les prédictions
    - seulement les trades
    """

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


# ==========================================================
# Affichage
# ==========================================================

def log_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    Affiche les principales métriques.
    """

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
        f"BUY : {metrics['buy_trades']}"
    )

    logger.info(
        f"SELL : {metrics['sell_trades']}"
    )

    logger.info(
        f"Wins : {metrics['wins']}"
    )

    logger.info(
        f"Losses : {metrics['losses']}"
    )

    logger.info(
        f"Win rate : "
        f"{metrics['win_rate_percent']} %"
    )

    logger.info(
        f"Accuracy signaux : "
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
        f"Expectancy : "
        f"{metrics['expectancy']}"
    )

    logger.info(
        f"Max drawdown : "
        f"{metrics['max_drawdown']}"
    )

    logger.info(
        f"Sharpe ratio : "
        f"{metrics['sharpe_ratio']}"
    )


# ==========================================================
# Pipeline principal
# ==========================================================

def backtest_model_v3(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Exécute le backtest complet.
    """

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

    ml_df, analysis_df = load_datasets(
        symbol=symbol,
        timeframe=timeframe,
    )

    X_test, y_test = create_test_split(
        ml_df=ml_df,
    )

    bundle = load_model_bundle(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    feature_names = bundle["features"]

    X_model = prepare_model_features(
        X_test=X_test,
        feature_names=feature_names,
    )

    predictions = predict_classes(
        model_name=model_name,
        bundle=bundle,
        X_test=X_model,
    )

    result_df = build_backtest_dataframe(
        analysis_df=analysis_df,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
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

    log_metrics(
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


# ==========================================================
# Arguments PowerShell
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """
    Lit les paramètres de la ligne de commande.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Backtester un modèle V3 "
            "pour un symbole et un timeframe."
        )
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à tester. Valeur par défaut : GC.",
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
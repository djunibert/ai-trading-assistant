"""
Backtest d'un modèle V3 par symbole et timeframe.

Exemples PowerShell :

python -m src.evaluation.backtest_model_v3 `
    --symbol GC `
    --timeframe 5m `
    --model random_forest

python -m src.evaluation.backtest_model_v3 `
    --symbol GC `
    --timeframe 15m `
    --model random_forest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


logger = get_logger(__name__)

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
}


def get_ml_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Retourne le chemin du dataset ML.
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
    Retourne le chemin du dataset Analysis.
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
    Retourne le chemin du modèle sauvegardé.
    """

    if model_name == "random_forest":
        return (
            Path("models")
            / symbol
            / timeframe
            / "random_forest"
            / "random_forest_v3.pkl"
        )

    raise ValueError(
        f"Modèle non supporté : {model_name}"
    )


def load_data(
    symbol: str,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Charge :
    - le dataset ML pour les features
    - le dataset Analysis pour future_return_1
    - le même jeu de test que pendant l'entraînement
    """

    ml_dataset_path = get_ml_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    analysis_dataset_path = get_analysis_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not ml_dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset ML introuvable : {ml_dataset_path}"
        )

    if not analysis_dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset Analysis introuvable : {analysis_dataset_path}"
        )

    logger.info(
        f"Chargement du dataset ML : {ml_dataset_path}"
    )

    logger.info(
        f"Chargement du dataset Analysis : {analysis_dataset_path}"
    )

    ml_df = pd.read_csv(ml_dataset_path)
    analysis_df = pd.read_csv(analysis_dataset_path)

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
            "le même nombre de lignes."
        )

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

    return analysis_df, X_test, y_test


def load_model_bundle(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict:
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

    logger.info(
        f"Chargement du modèle : {model_path}"
    )

    bundle = joblib.load(model_path)

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


def calculate_trading_metrics(
    future_returns: pd.Series,
    predictions: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """
    Calcule un backtest simple à partir des prédictions.

    SELL = -1
    NO_TRADE = 0
    BUY = 1

    Rendement stratégie :
    prediction × future_return_1
    """

    result_df = pd.DataFrame({
        "future_return": future_returns.to_numpy(),
        "prediction": predictions,
    })

    result_df["strategy_return"] = (
        result_df["prediction"]
        * result_df["future_return"]
    )

    result_df["trade_taken"] = (
        result_df["prediction"] != 0
    )

    trades_df = result_df[
        result_df["trade_taken"]
    ].copy()

    if trades_df.empty:
        empty_metrics = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_return": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

        return empty_metrics, trades_df

    trades_df["is_win"] = (
        trades_df["strategy_return"] > 0
    )

    wins = int(
        trades_df["is_win"].sum()
    )

    losses = int(
        (~trades_df["is_win"]).sum()
    )

    total_trades = len(trades_df)

    gross_profit = trades_df.loc[
        trades_df["strategy_return"] > 0,
        "strategy_return",
    ].sum()

    gross_loss = abs(
        trades_df.loc[
            trades_df["strategy_return"] < 0,
            "strategy_return",
        ].sum()
    )

    net_return = trades_df[
        "strategy_return"
    ].sum()

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0.0
    )

    expectancy = trades_df[
        "strategy_return"
    ].mean()

    trades_df["equity_curve"] = (
        1
        + trades_df["strategy_return"]
    ).cumprod()

    trades_df["running_max"] = (
        trades_df["equity_curve"]
        .cummax()
    )

    trades_df["drawdown"] = (
        trades_df["equity_curve"]
        / trades_df["running_max"]
        - 1
    )

    max_drawdown = trades_df[
        "drawdown"
    ].min()

    return_std = trades_df[
        "strategy_return"
    ].std()

    sharpe_ratio = (
        trades_df["strategy_return"].mean()
        / return_std
        * np.sqrt(252)
        if return_std > 0
        else 0.0
    )

    metrics = {
        "total_trades": int(total_trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(
            wins / total_trades * 100,
            2,
        ),
        "gross_profit": round(
            float(gross_profit),
            6,
        ),
        "gross_loss": round(
            float(gross_loss),
            6,
        ),
        "net_return": round(
            float(net_return),
            6,
        ),
        "profit_factor": round(
            float(profit_factor),
            4,
        ),
        "expectancy": round(
            float(expectancy),
            6,
        ),
        "max_drawdown": round(
            float(max_drawdown),
            6,
        ),
        "sharpe_ratio": round(
            float(sharpe_ratio),
            4,
        ),
    }

    return metrics, trades_df


def save_backtest_reports(
    symbol: str,
    timeframe: str,
    model_name: str,
    metrics: dict,
    trades_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Sauvegarde les métriques et l'historique des trades.
    """

    output_dir = (
        Path("reports/model_backtesting")
        / symbol
        / timeframe
        / model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / "backtest_metrics.csv"
    )

    trades_path = (
        output_dir
        / "backtest_trades.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    trades_df.to_csv(
        trades_path,
        index=False,
        encoding="utf-8-sig",
    )

    return metrics_path, trades_path


def backtest_model_v3(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> None:
    """
    Exécute le backtest du modèle demandé.
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

    analysis_df, X_test, _ = load_data(
        symbol=symbol,
        timeframe=timeframe,
    )

    bundle = load_model_bundle(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
    )

    model = bundle["model"]
    features = bundle["features"]

    missing_features = [
        feature
        for feature in features
        if feature not in X_test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features manquantes : {missing_features}"
        )

    X_test = X_test[features]

    predictions = model.predict(
        X_test
    )

    future_returns = analysis_df.loc[
        X_test.index,
        "future_return_1",
    ]

    metrics, trades_df = calculate_trading_metrics(
        future_returns=future_returns,
        predictions=predictions,
    )

    metrics_path, trades_path = save_backtest_reports(
        symbol=symbol,
        timeframe=timeframe,
        model_name=model_name,
        metrics=metrics,
        trades_df=trades_df,
    )

    logger.info(
        f"Modèle : {model_name}"
    )
    logger.info(
        f"Symbole : {symbol}"
    )
    logger.info(
        f"Timeframe : {timeframe}"
    )
    logger.info(
        f"Total trades : {metrics['total_trades']}"
    )
    logger.info(
        f"Wins : {metrics['wins']}"
    )
    logger.info(
        f"Losses : {metrics['losses']}"
    )
    logger.info(
        f"Win rate : {metrics['win_rate']} %"
    )
    logger.info(
        f"Profit factor : {metrics['profit_factor']}"
    )
    logger.info(
        f"Net return : {metrics['net_return']}"
    )
    logger.info(
        f"Expectancy : {metrics['expectancy']}"
    )
    logger.info(
        f"Max drawdown : {metrics['max_drawdown']}"
    )
    logger.info(
        f"Sharpe ratio : {metrics['sharpe_ratio']}"
    )
    logger.info(
        f"Métriques sauvegardées : {metrics_path}"
    )
    logger.info(
        f"Trades sauvegardés : {trades_path}"
    )


def parse_arguments() -> argparse.Namespace:
    """
    Lit les arguments PowerShell.
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
        help="Symbole à tester. Exemple : GC.",
    )

    parser.add_argument(
        "--timeframe",
        required=True,
        choices=sorted(VALID_TIMEFRAMES),
        help="Timeframe à tester.",
    )

    parser.add_argument(
        "--model",
        default="random_forest",
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
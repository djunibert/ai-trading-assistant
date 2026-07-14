"""
Comparaison des backtests V3.

Compare les modèles selon :
- timeframe
- nombre de trades
- win rate
- profit factor
- rendement composé
- expectancy
- drawdown
- Sharpe ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)

REPORTS_ROOT = Path("reports/model_backtesting")

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


def load_model_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict | None:
    """
    Charge les métriques d'un modèle.
    """

    metrics_path = (
        REPORTS_ROOT
        / symbol
        / timeframe
        / model_name
        / "backtest_metrics.csv"
    )

    if not metrics_path.exists():
        logger.warning(
            f"Rapport manquant : {metrics_path}"
        )
        return None

    metrics_df = pd.read_csv(metrics_path)

    if metrics_df.empty:
        logger.warning(
            f"Rapport vide : {metrics_path}"
        )
        return None

    metrics = metrics_df.iloc[0].to_dict()

    metrics["symbol"] = symbol
    metrics["timeframe"] = timeframe
    metrics["model"] = model_name

    return metrics


def calculate_score(
    row: pd.Series,
) -> float:
    """
    Calcule un score simple de sélection.

    Le score favorise :
    - profit factor élevé
    - expectancy positive
    - Sharpe élevé
    - faible drawdown
    """

    profit_factor = float(
        row.get("profit_factor", 0.0)
    )

    expectancy = float(
        row.get("expectancy", 0.0)
    )

    sharpe_ratio = float(
        row.get("sharpe_ratio", 0.0)
    )

    max_drawdown = abs(
        float(row.get("max_drawdown", 0.0))
    )

    win_rate = float(
        row.get("win_rate_percent", 0.0)
    ) / 100

    score = (
        profit_factor * 0.30
        + sharpe_ratio * 0.25
        + expectancy * 1000 * 0.20
        + win_rate * 0.15
        - max_drawdown * 0.10
    )

    return round(score, 6)


def compare_backtests(
    symbol: str,
    timeframes: list[str],
) -> pd.DataFrame:
    """
    Compare les modèles disponibles.
    """

    symbol = symbol.upper()

    results = []

    for timeframe in timeframes:
        timeframe = timeframe.lower()

        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Timeframe invalide : {timeframe}"
            )

        for model_name in sorted(VALID_MODELS):
            metrics = load_model_metrics(
                symbol=symbol,
                timeframe=timeframe,
                model_name=model_name,
            )

            if metrics is not None:
                results.append(metrics)

    if not results:
        raise ValueError(
            "Aucun rapport de backtest disponible."
        )

    comparison_df = pd.DataFrame(results)

    comparison_df["selection_score"] = (
        comparison_df.apply(
            calculate_score,
            axis=1,
        )
    )

    preferred_columns = [
        "symbol",
        "timeframe",
        "model",
        "total_observations",
        "total_trades",
        "buy_trades",
        "sell_trades",
        "wins",
        "losses",
        "win_rate_percent",
        "signal_accuracy_percent",
        "profit_factor",
        "expectancy",
        "net_return_sum",
        "compounded_return",
        "max_drawdown",
        "sharpe_ratio",
        "selection_score",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in comparison_df.columns
    ]

    comparison_df = comparison_df[
        available_columns
    ]

    comparison_df = comparison_df.sort_values(
        by=[
            "selection_score",
            "profit_factor",
            "sharpe_ratio",
        ],
        ascending=False,
    ).reset_index(drop=True)

    output_dir = (
        Path("reports/model_comparison")
        / symbol
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "backtest_comparison_v3.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    logger.info("Comparaison des backtests V3")
    logger.info(f"\n{comparison_df}")
    logger.info(
        f"Rapport sauvegardé : {output_path}"
    )

    if not comparison_df.empty:
        champion = comparison_df.iloc[0]

        logger.info("=" * 70)
        logger.info("CHAMPION ACTUEL")
        logger.info("=" * 70)
        logger.info(
            f"Modèle : {champion['model']}"
        )
        logger.info(
            f"Timeframe : {champion['timeframe']}"
        )
        logger.info(
            f"Score : {champion['selection_score']}"
        )
        logger.info(
            f"Profit factor : {champion['profit_factor']}"
        )
        logger.info(
            f"Sharpe : {champion['sharpe_ratio']}"
        )

    return comparison_df


def parse_arguments() -> argparse.Namespace:
    """
    Lit les arguments PowerShell.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Comparer les backtests V3 "
            "des différents modèles."
        )
    )

    parser.add_argument(
        "--symbol",
        default="GC",
        help="Symbole à comparer.",
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["5m", "15m"],
        choices=sorted(VALID_TIMEFRAMES),
        help="Timeframes à comparer.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    compare_backtests(
        symbol=arguments.symbol,
        timeframes=arguments.timeframes,
    )
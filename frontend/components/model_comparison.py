"""
Composant de comparaison des modèles.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.utils.constants import (
    AVAILABLE_MODELS,
    AVAILABLE_SYMBOLS,
    AVAILABLE_TIMEFRAMES,
    REPORTS_ROOT,
)


def load_backtest_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any] | None:
    """Charge les métriques d'un rapport de backtest."""

    metrics_path = (
        REPORTS_ROOT
        / symbol
        / timeframe
        / model_name
        / "backtest_metrics.csv"
    )

    if not metrics_path.exists():
        return None

    metrics_df = pd.read_csv(metrics_path)

    if metrics_df.empty:
        return None

    return metrics_df.iloc[0].to_dict()


def show_model_comparison() -> None:
    """Compare les modèles disponibles."""

    st.subheader("Comparaison des modèles")

    symbol_column, timeframe_column = st.columns(2)

    with symbol_column:
        symbol = st.selectbox(
            "Symbole comparé",
            options=AVAILABLE_SYMBOLS,
            index=0,
            key="comparison_symbol",
        )

    with timeframe_column:
        timeframe = st.selectbox(
            "Timeframe comparé",
            options=AVAILABLE_TIMEFRAMES,
            index=0,
            key="comparison_timeframe",
        )

    results: list[dict[str, Any]] = []

    for model_name in AVAILABLE_MODELS:
        metrics = load_backtest_metrics(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
        )

        if metrics is None:
            continue

        results.append(
            {
                "Modèle": model_name,
                "Accuracy (%)": float(
                    metrics.get(
                        "signal_accuracy_percent",
                        0,
                    )
                ),
                "Trades": int(
                    metrics.get(
                        "total_trades",
                        0,
                    )
                ),
                "Win rate (%)": float(
                    metrics.get(
                        "win_rate_percent",
                        0,
                    )
                ),
                "Profit factor": float(
                    metrics.get(
                        "profit_factor",
                        0,
                    )
                ),
                "Rendement composé (%)": (
                    float(
                        metrics.get(
                            "compounded_return",
                            0,
                        )
                    )
                    * 100
                ),
                "Drawdown (%)": (
                    float(
                        metrics.get(
                            "max_drawdown",
                            0,
                        )
                    )
                    * 100
                ),
                "Sharpe": float(
                    metrics.get(
                        "sharpe_ratio",
                        0,
                    )
                ),
            }
        )

    if not results:
        st.warning(
            "Aucun rapport de backtest disponible."
        )
        return

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by=[
            "Sharpe",
            "Profit factor",
        ],
        ascending=False,
    ).reset_index(drop=True)

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )

    champion = comparison_df.iloc[0]

    st.success(
        "Modèle champion selon le rendement ajusté au risque : "
        f"{champion['Modèle']}"
    )

    column_1, column_2, column_3, column_4 = (
        st.columns(4)
    )

    with column_1:
        st.metric(
            "Win rate",
            f"{champion['Win rate (%)']:.2f} %",
        )

    with column_2:
        st.metric(
            "Profit factor",
            f"{champion['Profit factor']:.2f}",
        )

    with column_3:
        st.metric(
            "Sharpe",
            f"{champion['Sharpe']:.2f}",
        )

    with column_4:
        st.metric(
            "Drawdown",
            f"{champion['Drawdown (%)']:.2f} %",
        )

"""
Composant de prédiction et d'analyse historique.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from frontend.utils.api_client import call_api
from frontend.utils.constants import (
    AVAILABLE_SYMBOLS,
    AVAILABLE_TIMEFRAMES,
)


def display_single_prediction(
    prediction: dict[str, Any],
) -> None:
    """Affiche une prédiction unique."""

    signal = prediction.get("signal", "UNKNOWN")
    confidence = prediction.get("confidence")
    close_price = prediction.get("close")
    datetime_value = prediction.get("datetime", "N/A")

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        if signal == "BUY":
            st.success("Signal : BUY")
        elif signal == "SELL":
            st.error("Signal : SELL")
        elif signal == "NO_TRADE":
            st.warning("Signal : NO TRADE")
        else:
            st.info(f"Signal : {signal}")

    with column_2:
        st.metric(
            "Confiance",
            (
                f"{float(confidence) * 100:.2f} %"
                if confidence is not None
                else "N/A"
            ),
        )

    with column_3:
        st.metric(
            "Prix",
            (
                f"{float(close_price):.2f}"
                if close_price is not None
                else "N/A"
            ),
        )

    st.caption(f"Observation : {datetime_value}")

    actual_target = prediction.get("actual_target")
    future_return = prediction.get("future_return_1")

    if actual_target is not None or future_return is not None:
        detail_1, detail_2 = st.columns(2)

        with detail_1:
            if actual_target is not None:
                mapping = {
                    -1: "SELL",
                    0: "NO_TRADE",
                    1: "BUY",
                }
                st.write(
                    "Cible réelle :",
                    mapping.get(int(actual_target), actual_target),
                )

        with detail_2:
            if future_return is not None:
                st.write(
                    "Rendement futur :",
                    round(float(future_return), 6),
                )


def calculate_historical_metrics(
    predictions_df: pd.DataFrame,
) -> dict[str, float]:
    """Calcule les principales métriques historiques."""

    empty_metrics = {
        "trades": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "win_rate": 0.0,
        "net_return": 0.0,
        "compounded_return": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
    }

    required_columns = {
        "prediction",
        "future_return_1",
    }

    if (
        predictions_df.empty
        or not required_columns.issubset(predictions_df.columns)
    ):
        return empty_metrics

    working_df = predictions_df.copy()

    working_df["prediction"] = pd.to_numeric(
        working_df["prediction"],
        errors="coerce",
    ).fillna(0)

    working_df["future_return_1"] = pd.to_numeric(
        working_df["future_return_1"],
        errors="coerce",
    ).fillna(0)

    working_df["strategy_return"] = (
        working_df["prediction"]
        * working_df["future_return_1"]
    )

    trades_df = working_df[
        working_df["prediction"] != 0
    ].copy()

    if trades_df.empty:
        return empty_metrics

    wins = int(
        (trades_df["strategy_return"] > 0).sum()
    )
    losses = int(
        (trades_df["strategy_return"] < 0).sum()
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

    trades_df["equity_curve"] = (
        1.0 + trades_df["strategy_return"]
    ).cumprod()

    trades_df["running_max"] = (
        trades_df["equity_curve"].cummax()
    )

    trades_df["drawdown"] = (
        trades_df["equity_curve"]
        / trades_df["running_max"]
        - 1.0
    )

    total_trades = len(trades_df)

    return {
        "trades": float(total_trades),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": (
            wins / total_trades * 100
            if total_trades > 0
            else 0.0
        ),
        "net_return": float(
            trades_df["strategy_return"].sum()
        ),
        "compounded_return": float(
            trades_df["equity_curve"].iloc[-1] - 1.0
        ),
        "max_drawdown": float(
            trades_df["drawdown"].min()
        ),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        ),
    }


def display_historical_predictions(
    result: dict[str, Any],
) -> None:
    """Affiche les prédictions historiques."""

    predictions = result.get("predictions", [])
    signal_counts = result.get("signal_counts", {})

    if not predictions:
        st.warning("Aucune prédiction disponible.")
        return

    st.success(
        f"{len(predictions)} bougies analysées"
    )

    buy_count = int(signal_counts.get("BUY", 0))
    sell_count = int(signal_counts.get("SELL", 0))
    no_trade_count = int(
        signal_counts.get("NO_TRADE", 0)
    )

    column_1, column_2, column_3, column_4 = (
        st.columns(4)
    )

    with column_1:
        st.metric("BUY", buy_count)

    with column_2:
        st.metric("SELL", sell_count)

    with column_3:
        st.metric("NO TRADE", no_trade_count)

    with column_4:
        st.metric(
            "Entrées possibles",
            buy_count + sell_count,
        )

    predictions_df = pd.DataFrame(predictions)

    if "datetime" in predictions_df.columns:
        predictions_df["datetime"] = pd.to_datetime(
            predictions_df["datetime"],
            errors="coerce",
            utc=True,
        )

    if {
        "prediction",
        "future_return_1",
    }.issubset(predictions_df.columns):
        predictions_df["prediction"] = pd.to_numeric(
            predictions_df["prediction"],
            errors="coerce",
        ).fillna(0)

        predictions_df["future_return_1"] = pd.to_numeric(
            predictions_df["future_return_1"],
            errors="coerce",
        ).fillna(0)

        predictions_df["strategy_return"] = (
            predictions_df["prediction"]
            * predictions_df["future_return_1"]
        )

        predictions_df["equity_curve"] = (
            1.0
            + predictions_df[
                "strategy_return"
            ].fillna(0)
        ).cumprod()

        metrics = calculate_historical_metrics(
            predictions_df
        )

        st.subheader("Résultats historiques")

        metric_1, metric_2, metric_3, metric_4 = (
            st.columns(4)
        )

        with metric_1:
            st.metric(
                "Win rate",
                f"{metrics['win_rate']:.2f} %",
            )

        with metric_2:
            st.metric(
                "Profit factor",
                f"{metrics['profit_factor']:.2f}",
            )

        with metric_3:
            st.metric(
                "Rendement composé",
                (
                    f"{metrics['compounded_return'] * 100:.2f} %"
                ),
            )

        with metric_4:
            st.metric(
                "Max drawdown",
                (
                    f"{metrics['max_drawdown'] * 100:.2f} %"
                ),
            )

        if "datetime" in predictions_df.columns:
            st.subheader("Courbe de performance")

            chart_df = (
                predictions_df[
                    [
                        "datetime",
                        "equity_curve",
                    ]
                ]
                .dropna(subset=["datetime"])
                .set_index("datetime")
            )

            if not chart_df.empty:
                st.line_chart(chart_df)

    st.subheader("Points d'entrée historiques")

    if "signal" not in predictions_df.columns:
        st.warning(
            "La colonne signal est absente des résultats."
        )
        return

    trade_rows = predictions_df[
        predictions_df["signal"].isin(
            ["BUY", "SELL"]
        )
    ].copy()

    if trade_rows.empty:
        st.info(
            "Aucun signal BUY ou SELL sur cette période."
        )
    else:
        columns_to_show = [
            column
            for column in [
                "datetime",
                "close",
                "signal",
                "confidence",
                "actual_target",
                "future_return_1",
                "strategy_return",
            ]
            if column in trade_rows.columns
        ]

        st.dataframe(
            trade_rows[columns_to_show],
            width="stretch",
            hide_index=True,
        )

    with st.expander(
        "Afficher toutes les prédictions"
    ):
        st.dataframe(
            predictions_df,
            width="stretch",
            hide_index=True,
        )


def show_prediction_section() -> dict[str, Any] | None:
    """Affiche les trois modes de prédiction."""

    st.subheader("Prédiction du marché")

    symbol_column, timeframe_column = st.columns(2)

    with symbol_column:
        symbol = st.selectbox(
            "Symbole",
            options=AVAILABLE_SYMBOLS,
            index=0,
            key="prediction_symbol",
        )

    with timeframe_column:
        timeframe = st.selectbox(
            "Timeframe",
            options=AVAILABLE_TIMEFRAMES,
            index=0,
            key="prediction_timeframe",
        )

    prediction_mode = st.radio(
        "Mode de prédiction",
        options=[
            "Dernière bougie",
            "Date précise",
            "Période historique",
        ],
        horizontal=True,
    )

    if prediction_mode == "Dernière bougie":
        if not st.button(
            "Générer la dernière prédiction",
            type="primary",
            width="stretch",
        ):
            return None

        try:
            with st.spinner(
                "Analyse de la dernière bougie..."
            ):
                prediction = call_api(
                    "GET",
                    (
                        f"/predict/latest/"
                        f"{symbol}/{timeframe}"
                    ),
                )
        except RuntimeError as error:
            st.error(str(error))
            return None

        display_single_prediction(prediction)
        return prediction

    if prediction_mode == "Date précise":
        selected_datetime = st.text_input(
            "Date et heure UTC",
            value="2026-07-10T20:30:00+00:00",
        )

        if not st.button(
            "Prédire cette bougie",
            type="primary",
            width="stretch",
        ):
            return None

        try:
            with st.spinner(
                "Analyse de la bougie sélectionnée..."
            ):
                prediction = call_api(
                    "GET",
                    (
                        f"/predict/at/"
                        f"{symbol}/{timeframe}"
                    ),
                    params={
                        "datetime": selected_datetime,
                    },
                )
        except RuntimeError as error:
            st.error(str(error))
            return None

        display_single_prediction(prediction)
        return prediction

    start_datetime = st.text_input(
        "Date de début UTC",
        value="2026-06-20T00:00:00Z",
    )

    end_datetime = st.text_input(
        "Date de fin UTC",
        value="2026-07-10T20:50:00Z",
    )

    limit = st.number_input(
        "Nombre maximal de bougies",
        min_value=10,
        max_value=5000,
        value=3500,
        step=100,
    )

    if not st.button(
        "Analyser la période",
        type="primary",
        width="stretch",
    ):
        return None

    try:
        with st.spinner(
            "Analyse de la période historique..."
        ):
            result = call_api(
                "GET",
                (
                    f"/predict/range/"
                    f"{symbol}/{timeframe}"
                ),
                params={
                    "start": start_datetime,
                    "end": end_datetime,
                    "limit": int(limit),
                },
            )
    except RuntimeError as error:
        st.error(str(error))
        return None

    display_historical_predictions(result)
    return None

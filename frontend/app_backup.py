"""
Interface Streamlit du système IA de trading.

Fonctionnalités :
- État de FastAPI
- Informations sur le modèle actif
- Dernière prédiction
- Prédiction à une date précise
- Analyse historique sur une période
- Affichage des points d'entrée BUY et SELL
- Courbe de performance
- Gestion du risque
- Comparaison Random Forest, XGBoost et LSTM
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

REQUEST_TIMEOUT = 120

REPORTS_ROOT = Path(
    "reports/model_backtesting"
)

AVAILABLE_SYMBOLS = [
    "GC",
    "SI",
    "NQ",
    "ES",
]

AVAILABLE_TIMEFRAMES = [
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
]

AVAILABLE_MODELS = [
    "random_forest",
    "xgboost",
    "lstm",
]


st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
)


def call_api(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Envoie une requête HTTP à FastAPI.
    """

    api_url = st.session_state.api_url.rstrip("/")

    url = f"{api_url}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )

    except requests.ConnectionError as error:
        raise RuntimeError(
            "Impossible de joindre FastAPI. "
            "Vérifie que l'API fonctionne sur le port 8000."
        ) from error

    except requests.Timeout as error:
        raise RuntimeError(
            "La requête vers FastAPI a dépassé le délai."
        ) from error

    try:
        response_data = response.json()

    except requests.JSONDecodeError:
        response_data = {
            "detail": response.text,
        }

    if not response.ok:
        detail = response_data.get(
            "detail",
            response_data,
        )

        raise RuntimeError(
            f"Erreur API {response.status_code} : {detail}"
        )

    return response_data


def show_header() -> None:
    """
    Affiche l'en-tête principal.
    """

    st.title(
        "AI Trading Assistant System"
    )

    st.write(
        "Plateforme de prédiction, d'analyse historique, "
        "de comparaison des modèles et de gestion du risque."
    )


def show_sidebar() -> None:
    """
    Affiche la barre latérale.
    """

    with st.sidebar:
        st.header(
            "Configuration"
        )

        st.session_state.api_url = st.text_input(
            "Adresse FastAPI",
            value=st.session_state.api_url,
        )

        st.write(
            "Services"
        )

        st.link_button(
            "Documentation FastAPI",
            (
                f"{st.session_state.api_url.rstrip('/')}"
                "/docs"
            ),
            use_container_width=True,
        )

        st.link_button(
            "MLflow",
            "http://localhost:5000",
            use_container_width=True,
        )

        st.write("")

        st.caption(
            "Random Forest, XGBoost et LSTM"
        )


def show_api_status() -> None:
    """
    Affiche l'état de FastAPI et du modèle actif.
    """

    st.subheader(
        "État du système"
    )

    api_column, model_column = st.columns(2)

    try:
        health = call_api(
            "GET",
            "/health",
        )

        with api_column:
            st.success(
                "FastAPI est disponible"
            )

            st.metric(
                "État de l'API",
                health.get(
                    "status",
                    "unknown",
                ),
            )

            st.write(
                "Version :",
                health.get(
                    "version",
                    "N/A",
                ),
            )

            st.write(
                "Modèle présent :",
                health.get(
                    "model_exists",
                    False,
                ),
            )

    except RuntimeError as error:
        with api_column:
            st.error(
                str(error)
            )

        return

    try:
        model_information = call_api(
            "GET",
            "/model",
        )

        with model_column:
            if model_information.get(
                "loaded"
            ):
                st.success(
                    "Modèle chargé"
                )

                st.write(
                    "Type :",
                    model_information.get(
                        "model_type",
                        "N/A",
                    ),
                )

                st.write(
                    "Symbole :",
                    model_information.get(
                        "symbol",
                        "N/A",
                    ),
                )

                st.write(
                    "Timeframe :",
                    model_information.get(
                        "timeframe",
                        "N/A",
                    ),
                )

                st.write(
                    "Nombre de features :",
                    model_information.get(
                        "feature_count",
                        "N/A",
                    ),
                )

                st.write(
                    "Type de split :",
                    model_information.get(
                        "split_type",
                        "N/A",
                    ),
                )

            else:
                st.warning(
                    "Aucun modèle chargé"
                )

                st.write(
                    "Chemin :",
                    model_information.get(
                        "path",
                        "N/A",
                    ),
                )

    except RuntimeError as error:
        with model_column:
            st.error(
                str(error)
            )


def display_single_prediction(
    prediction: dict[str, Any],
) -> None:
    """
    Affiche une prédiction unique.
    """

    signal = prediction.get(
        "signal",
        "UNKNOWN",
    )

    confidence = prediction.get(
        "confidence"
    )

    close_price = prediction.get(
        "close"
    )

    datetime_value = prediction.get(
        "datetime",
        "N/A",
    )

    result_column_1, result_column_2, result_column_3 = (
        st.columns(3)
    )

    with result_column_1:
        if signal == "BUY":
            st.success(
                "Signal : BUY"
            )

        elif signal == "SELL":
            st.error(
                "Signal : SELL"
            )

        elif signal == "NO_TRADE":
            st.warning(
                "Signal : NO TRADE"
            )

        else:
            st.info(
                f"Signal : {signal}"
            )

    with result_column_2:
        st.metric(
            "Confiance",
            (
                f"{confidence * 100:.2f} %"
                if confidence is not None
                else "N/A"
            ),
        )

    with result_column_3:
        st.metric(
            "Prix",
            (
                f"{close_price:.2f}"
                if close_price is not None
                else "N/A"
            ),
        )

    st.caption(
        f"Observation : {datetime_value}"
    )

    actual_target = prediction.get(
        "actual_target"
    )

    future_return = prediction.get(
        "future_return_1"
    )

    if (
        actual_target is not None
        or future_return is not None
    ):
        detail_column_1, detail_column_2 = (
            st.columns(2)
        )

        with detail_column_1:
            if actual_target is not None:
                target_mapping = {
                    -1: "SELL",
                    0: "NO_TRADE",
                    1: "BUY",
                }

                st.write(
                    "Cible réelle :",
                    target_mapping.get(
                        actual_target,
                        actual_target,
                    ),
                )

        with detail_column_2:
            if future_return is not None:
                st.write(
                    "Rendement futur :",
                    round(
                        float(
                            future_return
                        ),
                        6,
                    ),
                )


def calculate_historical_metrics(
    predictions_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calcule les métriques simples sur les prédictions historiques.
    """

    if predictions_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_return": 0.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    predictions_df = predictions_df.copy()

    predictions_df["strategy_return"] = (
        predictions_df["prediction"]
        * predictions_df["future_return_1"]
    )

    trades_df = predictions_df[
        predictions_df["prediction"] != 0
    ].copy()

    if trades_df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_return": 0.0,
            "compounded_return": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

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

    total_trades = len(
        trades_df
    )

    return {
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / total_trades * 100
            if total_trades > 0
            else 0.0
        ),
        "net_return": float(
            trades_df[
                "strategy_return"
            ].sum()
        ),
        "compounded_return": float(
            trades_df[
                "equity_curve"
            ].iloc[-1]
            - 1
        ),
        "max_drawdown": float(
            trades_df[
                "drawdown"
            ].min()
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
    """
    Affiche plusieurs prédictions historiques.
    """

    predictions = result.get(
        "predictions",
        [],
    )

    signal_counts = result.get(
        "signal_counts",
        {},
    )

    if not predictions:
        st.warning(
            "Aucune prédiction disponible."
        )

        return

    st.success(
        f"{len(predictions)} bougies analysées"
    )

    buy_count = int(
        signal_counts.get(
            "BUY",
            0,
        )
    )

    sell_count = int(
        signal_counts.get(
            "SELL",
            0,
        )
    )

    no_trade_count = int(
        signal_counts.get(
            "NO_TRADE",
            0,
        )
    )

    trade_count = (
        buy_count
        + sell_count
    )

    column_1, column_2, column_3, column_4 = (
        st.columns(4)
    )

    with column_1:
        st.metric(
            "BUY",
            buy_count,
        )

    with column_2:
        st.metric(
            "SELL",
            sell_count,
        )

    with column_3:
        st.metric(
            "NO TRADE",
            no_trade_count,
        )

    with column_4:
        st.metric(
            "Entrées possibles",
            trade_count,
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    predictions_df["datetime"] = pd.to_datetime(
        predictions_df["datetime"],
        errors="coerce",
        utc=True,
    )

    if {
        "prediction",
        "future_return_1",
    }.issubset(
        predictions_df.columns
    ):
        predictions_df["strategy_return"] = (
            predictions_df["prediction"]
            * predictions_df["future_return_1"]
        )

        predictions_df["equity_curve"] = (
            1
            + predictions_df[
                "strategy_return"
            ].fillna(0)
        ).cumprod()

        metrics = calculate_historical_metrics(
            predictions_df
        )

        st.subheader(
            "Résultats historiques"
        )

        metric_column_1, metric_column_2, metric_column_3, metric_column_4 = (
            st.columns(4)
        )

        with metric_column_1:
            st.metric(
                "Win rate",
                f"{metrics['win_rate']:.2f} %",
            )

        with metric_column_2:
            st.metric(
                "Profit factor",
                f"{metrics['profit_factor']:.2f}",
            )

        with metric_column_3:
            st.metric(
                "Rendement composé",
                (
                    f"{metrics['compounded_return'] * 100:.2f} %"
                ),
            )

        with metric_column_4:
            st.metric(
                "Max drawdown",
                (
                    f"{metrics['max_drawdown'] * 100:.2f} %"
                ),
            )

        st.subheader(
            "Courbe de performance"
        )

        chart_df = (
            predictions_df[
                [
                    "datetime",
                    "equity_curve",
                ]
            ]
            .dropna(
                subset=[
                    "datetime"
                ]
            )
            .set_index(
                "datetime"
            )
        )

        if not chart_df.empty:
            st.line_chart(
                chart_df
            )

    st.subheader(
        "Points d'entrée historiques"
    )

    trade_rows = predictions_df[
        predictions_df["signal"].isin(
            [
                "BUY",
                "SELL",
            ]
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
            trade_rows[
                columns_to_show
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "Afficher toutes les prédictions"
    ):
        st.dataframe(
            predictions_df,
            use_container_width=True,
            hide_index=True,
        )


def show_prediction_section() -> dict[str, Any] | None:
    """
    Affiche les trois modes de prédiction.
    """

    st.subheader(
        "Prédiction du marché"
    )

    symbol_column, timeframe_column = (
        st.columns(2)
    )

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
            use_container_width=True,
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
            st.error(
                str(error)
            )

            return None

        display_single_prediction(
            prediction
        )

        return prediction

    if prediction_mode == "Date précise":
        selected_datetime = st.text_input(
            "Date et heure UTC",
            value=(
                "2026-07-10T20:30:00+00:00"
            ),
        )

        if not st.button(
            "Prédire cette bougie",
            type="primary",
            use_container_width=True,
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
            st.error(
                str(error)
            )

            return None

        display_single_prediction(
            prediction
        )

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
        use_container_width=True,
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
                    "limit": int(
                        limit
                    ),
                },
            )

    except RuntimeError as error:
        st.error(
            str(error)
        )

        return None

    display_historical_predictions(
        result
    )

    return None


def load_backtest_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any] | None:
    """
    Charge les métriques d'un rapport de backtest.
    """

    metrics_path = (
        REPORTS_ROOT
        / symbol
        / timeframe
        / model_name
        / "backtest_metrics.csv"
    )

    if not metrics_path.exists():
        return None

    metrics_df = pd.read_csv(
        metrics_path
    )

    if metrics_df.empty:
        return None

    return metrics_df.iloc[
        0
    ].to_dict()


def show_model_comparison() -> None:
    """
    Compare Random Forest, XGBoost et LSTM.
    """

    st.subheader(
        "Comparaison des modèles"
    )

    symbol_column, timeframe_column = (
        st.columns(2)
    )

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

    results = []

    for model_name in AVAILABLE_MODELS:
        metrics = load_backtest_metrics(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
        )

        if metrics is None:
            continue

        results.append({
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
        })

    if not results:
        st.warning(
            "Aucun rapport de backtest disponible."
        )

        return

    comparison_df = pd.DataFrame(
        results
    )

    comparison_df = comparison_df.sort_values(
        by=[
            "Sharpe",
            "Profit factor",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    champion = comparison_df.iloc[
        0
    ]

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

    st.caption(
        "Random Forest produit généralement plus de trades. "
        "XGBoost est plus sélectif et présente actuellement "
        "un meilleur rendement ajusté au risque sur GC 5m."
    )


def show_risk_section(
    prediction: dict[str, Any] | None,
) -> None:
    """
    Affiche la section de gestion du risque.
    """

    st.subheader(
        "Gestion du risque"
    )

    default_signal = (
        prediction.get(
            "signal",
            "BUY",
        )
        if prediction
        else "BUY"
    )

    signal_options = [
        "BUY",
        "SELL",
        "NO_TRADE",
    ]

    signal_index = (
        signal_options.index(
            default_signal
        )
        if default_signal in signal_options
        else 0
    )

    column_1, column_2, column_3 = (
        st.columns(3)
    )

    with column_1:
        signal = st.selectbox(
            "Signal",
            options=signal_options,
            index=signal_index,
        )

        account_balance = st.number_input(
            "Solde du compte",
            min_value=1000.0,
            value=100000.0,
            step=1000.0,
        )

    with column_2:
        default_entry_price = 3300.0

        if (
            prediction
            and prediction.get(
                "close"
            ) is not None
        ):
            default_entry_price = float(
                prediction[
                    "close"
                ]
            )

        entry_price = st.number_input(
            "Prix d'entrée",
            min_value=0.01,
            value=default_entry_price,
            step=0.10,
        )

        atr_14 = st.number_input(
            "ATR 14",
            min_value=0.01,
            value=8.0,
            step=0.10,
        )

    with column_3:
        risk_percent = st.number_input(
            "Risque par trade (%)",
            min_value=0.10,
            max_value=5.0,
            value=0.50,
            step=0.10,
        )

        risk_reward_ratio = st.number_input(
            "Ratio risque/rendement",
            min_value=0.50,
            value=2.0,
            step=0.25,
        )

    confidence = 0.70

    if (
        prediction
        and prediction.get(
            "confidence"
        ) is not None
    ):
        confidence = float(
            prediction[
                "confidence"
            ]
        )

    if not st.button(
        "Calculer le risque",
        use_container_width=True,
    ):
        return

    payload = {
        "signal": signal,
        "account_balance": account_balance,
        "risk_percent": risk_percent,
        "entry_price": entry_price,
        "atr_14": atr_14,
        "confidence": confidence,
        "risk_reward_ratio": risk_reward_ratio,
    }

    try:
        risk_result = call_api(
            "POST",
            "/risk/calculate",
            json=payload,
        )

    except RuntimeError as error:
        st.error(
            str(error)
        )

        return

    st.success(
        "Calcul du risque terminé"
    )

    if not isinstance(
        risk_result,
        dict,
    ):
        st.write(
            risk_result
        )

        return

    with st.expander(
        "Afficher le résultat complet",
        expanded=True,
    ):
        st.json(
            risk_result
        )


def main() -> None:
    """
    Point d'entrée de l'application Streamlit.
    """

    if "api_url" not in st.session_state:
        st.session_state.api_url = (
            DEFAULT_API_URL
        )

    show_header()

    show_sidebar()

    show_api_status()

    st.divider()

    prediction = show_prediction_section()

    st.divider()

    show_model_comparison()

    st.divider()

    show_risk_section(
        prediction
    )


if __name__ == "__main__":
    main()
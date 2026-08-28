


# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
)

ADMIN_PASSWORD = os.getenv(
    "ADMIN_PASSWORD",
    "admin123",
)

MLFLOW_URL = os.getenv(
    "MLFLOW_URL",
    "http://127.0.0.1:5000",
)

REQUEST_TIMEOUT = 30

REPORTS_ROOT = Path(
    "reports/model_backtesting"
)

MONITORING_ROOT = Path(
    "reports/monitoring"
)


st.set_page_config(
    page_title="Administration",
    page_icon="⚙️",
    layout="wide",
)


def check_admin_access() -> bool:
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if st.session_state.admin_authenticated:
        return True

    st.title("Accès administrateur")

    password = st.text_input(
        "Mot de passe",
        type="password",
    )

    if st.button(
        "Connexion",
        type="primary",
        width="stretch",
    ):
        if password == ADMIN_PASSWORD:
            st.session_state.admin_authenticated = True
            st.rerun()

        else:
            st.error(
                "Mot de passe incorrect."
            )

    return False


def call_api(
    endpoint: str,
) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{DEFAULT_API_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        raise RuntimeError(
            f"Erreur API : {error}"
        ) from error


def check_service(
    url: str,
) -> bool:
    try:
        response = requests.get(
            url,
            timeout=5,
        )

        return response.ok

    except requests.RequestException:
        return False


def show_system_status() -> None:
    st.subheader(
        "État des services"
    )

    api_column, mlflow_column, model_column = st.columns(3)

    api_available = check_service(
        f"{DEFAULT_API_URL}/health"
    )

    mlflow_available = check_service(
        f"{MLFLOW_URL}/health"
    )

    with api_column:
        st.metric(
            "FastAPI",
            "Disponible" if api_available else "Indisponible",
        )

    with mlflow_column:
        st.metric(
            "MLflow",
            "Disponible" if mlflow_available else "Indisponible",
        )

    try:
        model_information = call_api(
            "/model"
        )

        with model_column:
            st.metric(
                "Modèle actif",
                model_information.get(
                    "model_type",
                    "Inconnu",
                ),
            )

            st.caption(
                f"Features : "
                f"{model_information.get('feature_count', 'N/A')}"
            )

    except RuntimeError:
        with model_column:
            st.metric(
                "Modèle actif",
                "Indisponible",
            )


def show_evidently_section() -> None:
    st.subheader(
        "Monitoring Evidently"
    )

    symbol_column, timeframe_column = st.columns(2)

    with symbol_column:
        symbol = st.selectbox(
            "Symbole",
            [
                "GC",
                "SI",
                "NQ",
                "ES",
            ],
            key="admin_evidently_symbol",
        )

    with timeframe_column:
        timeframe = st.selectbox(
            "Timeframe",
            [
                "5m",
                "15m",
                "30m",
                "1h",
                "4h",
                "1d",
            ],
            key="admin_evidently_timeframe",
        )

    report_directory = (
        MONITORING_ROOT
        / symbol
        / timeframe
    )

    html_path = (
        report_directory
        / "evidently_report.html"
    )

    json_path = (
        report_directory
        / "evidently_report.json"
    )

    if not html_path.exists():
        st.warning(
            "Aucun rapport Evidently disponible."
        )

        st.code(
            (
                "python -m src.monitoring.evidently_report "
                f"--symbol {symbol} "
                f"--timeframe {timeframe}"
            ),
            language="powershell",
        )

        return

    st.success(
        "Rapport Evidently disponible"
    )

    info_column_1, info_column_2 = st.columns(2)

    with info_column_1:
        st.metric(
            "Taille HTML",
            f"{html_path.stat().st_size / 1024 / 1024:.2f} Mo",
        )

    with info_column_2:
        st.metric(
            "Rapport JSON",
            "Disponible" if json_path.exists() else "Absent",
        )

    with html_path.open("rb") as report_file:
        st.download_button(
            "Télécharger le rapport HTML",
            data=report_file.read(),
            file_name=(
                f"evidently_{symbol}_{timeframe}.html"
            ),
            mime="text/html",
            width="stretch",
        )

    if json_path.exists():
        try:
            report_data = json.loads(
                json_path.read_text(
                    encoding="utf-8"
                )
            )

            with st.expander(
                "Afficher le rapport JSON"
            ):
                st.json(
                    report_data
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as error:
            st.error(
                f"Impossible de lire le rapport JSON : {error}"
            )


def load_backtest_metrics(
    symbol: str,
    timeframe: str,
    model_name: str,
) -> dict[str, Any] | None:
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


def show_backtest_section() -> None:
    st.subheader(
        "Backtesting"
    )

    symbol_column, timeframe_column = st.columns(2)

    with symbol_column:
        symbol = st.selectbox(
            "Symbole du backtest",
            [
                "GC",
                "SI",
                "NQ",
                "ES",
            ],
            key="admin_backtest_symbol",
        )

    with timeframe_column:
        timeframe = st.selectbox(
            "Timeframe du backtest",
            [
                "5m",
                "15m",
                "30m",
                "1h",
                "4h",
                "1d",
            ],
            key="admin_backtest_timeframe",
        )

    models = [
        "random_forest",
        "xgboost",
        "lstm",
    ]

    results = []

    for model_name in models:
        metrics = load_backtest_metrics(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
        )

        if metrics is None:
            continue

        results.append({
            "Modèle": model_name,
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
        by="Sharpe",
        ascending=False,
    ).reset_index(
        drop=True
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True,
    )

    champion = comparison_df.iloc[
        0
    ]

    st.success(
        f"Modèle champion : {champion['Modèle']}"
    )


def show_mlflow_section() -> None:
    st.subheader(
        "MLflow"
    )

    mlflow_available = check_service(
        f"{MLFLOW_URL}/health"
    )

    if mlflow_available:
        st.success(
            "MLflow est disponible"
        )

    else:
        st.error(
            "MLflow est indisponible"
        )

    st.link_button(
        "Ouvrir MLflow",
        MLFLOW_URL,
        width="stretch",
    )


def show_dvc_section() -> None:
    st.subheader(
        "DVC"
    )

    st.write(
        "DVC versionne les données, les modèles "
        "et les artefacts volumineux."
    )

    st.code(
        (
            "dvc status\n"
            "dvc remote list\n"
            "dvc push\n"
            "dvc pull"
        ),
        language="powershell",
    )


def main() -> None:
    if not check_admin_access():
        st.stop()

    st.title(
        "Tableau de bord administrateur"
    )

    if st.button(
        "Déconnexion"
    ):
        st.session_state.admin_authenticated = False
        st.rerun()

    show_system_status()

    st.divider()

    show_evidently_section()

    st.divider()

    show_backtest_section()

    st.divider()

    show_mlflow_section()

    st.divider()

    show_dvc_section()


main()
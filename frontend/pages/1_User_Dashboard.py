"""
Tableau de bord utilisateur du système IA de trading.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from frontend.components.api_status import show_api_status
from frontend.components.header import show_header
from frontend.components.model_comparison import show_model_comparison
from frontend.components.prediction import show_prediction_section
from frontend.components.risk import show_risk_section
from frontend.components.sidebar import show_sidebar
from frontend.utils.constants import DEFAULT_API_URL


st.set_page_config(
    page_title="Tableau de bord utilisateur",
    page_icon="📈",
    layout="wide",
)


def main() -> None:
    """Point d'entrée du tableau de bord utilisateur."""

    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL

    show_header()
    show_sidebar()
    show_api_status()

    st.divider()

    prediction = show_prediction_section()

    st.divider()

    show_model_comparison()

    st.divider()

    show_risk_section(prediction)


if __name__ == "__main__":
    main()

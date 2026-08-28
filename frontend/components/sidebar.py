import streamlit as st

from frontend.utils.constants import DEFAULT_MLFLOW_URL


def show_sidebar() -> None:
    with st.sidebar:
        st.header("Configuration")

        st.session_state.api_url = st.text_input(
            "Adresse FastAPI",
            value=st.session_state.api_url,
        )

        st.write("Services")

        st.link_button(
            "Documentation FastAPI",
            f"{st.session_state.api_url.rstrip('/')}/docs",
            width="stretch",
        )

        st.link_button(
            "MLflow",
            DEFAULT_MLFLOW_URL,
            width="stretch",
        )

        st.caption("Random Forest, XGBoost et LSTM")
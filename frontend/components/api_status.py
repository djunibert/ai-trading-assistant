from __future__ import annotations

import streamlit as st

from frontend.utils.api_client import call_api


def show_api_status() -> None:
    st.subheader("État du système")

    api_column, model_column = st.columns(2)

    try:
        health = call_api(
            "GET",
            "/health",
        )

        with api_column:
            st.success("FastAPI est disponible")

            st.metric(
                "État de l’API",
                health.get("status", "unknown"),
            )

            st.write(
                "Version :",
                health.get("version", "N/A"),
            )

            st.write(
                "Modèle présent :",
                health.get("model_exists", False),
            )

    except RuntimeError as error:
        with api_column:
            st.error(str(error))

        return

    try:
        model_information = call_api(
            "GET",
            "/model",
        )

        with model_column:
            if model_information.get("loaded"):
                st.success("Modèle chargé")

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
                st.warning("Aucun modèle chargé")

                st.write(
                    "Chemin :",
                    model_information.get(
                        "path",
                        "N/A",
                    ),
                )

    except RuntimeError as error:
        with model_column:
            st.error(str(error))
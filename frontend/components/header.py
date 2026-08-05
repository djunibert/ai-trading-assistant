import streamlit as st


def show_header() -> None:
    st.title("AI Trading Assistant System")

    st.write(
        "Plateforme de prédiction, d’analyse historique, "
        "de comparaison des modèles et de gestion du risque."
    )
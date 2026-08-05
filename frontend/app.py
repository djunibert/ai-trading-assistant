import streamlit as st


st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
)

st.title("AI Trading Assistant System")

st.write(
    "Utilisez le menu à gauche pour accéder "
    "au tableau de bord utilisateur ou administrateur."
)

st.subheader("Tableau de bord utilisateur")

st.write(
    "Prédictions, analyse historique, comparaison des modèles "
    "et gestion du risque."
)

st.subheader("Tableau de bord administrateur")

st.write(
    "Monitoring Evidently, MLflow, DVC, backtesting "
    "et état des services."
)
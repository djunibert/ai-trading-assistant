"""
Composant de gestion du risque.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.utils.api_client import call_api
from frontend.utils.constants import SIGNAL_OPTIONS


def show_risk_section(
    prediction: dict[str, Any] | None,
) -> None:
    """Affiche la section de gestion du risque."""

    st.subheader("Gestion du risque")

    default_signal = (
        prediction.get("signal", "BUY")
        if prediction
        else "BUY"
    )

    signal_index = (
        SIGNAL_OPTIONS.index(default_signal)
        if default_signal in SIGNAL_OPTIONS
        else 0
    )

    column_1, column_2, column_3 = st.columns(3)

    with column_1:
        signal = st.selectbox(
            "Signal",
            options=SIGNAL_OPTIONS,
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
            and prediction.get("close") is not None
        ):
            default_entry_price = float(
                prediction["close"]
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
        and prediction.get("confidence") is not None
    ):
        confidence = float(
            prediction["confidence"]
        )

    if not st.button(
        "Calculer le risque",
        width="stretch",
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
        st.error(str(error))
        return

    st.success("Calcul du risque terminé")

    if not isinstance(risk_result, dict):
        st.write(risk_result)
        return

    with st.expander(
        "Afficher le résultat complet",
        expanded=True,
    ):
        st.json(risk_result)

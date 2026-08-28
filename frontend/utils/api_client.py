from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from frontend.utils.constants import REQUEST_TIMEOUT


def call_api(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> dict[str, Any]:
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
            "Impossible de joindre FastAPI."
        ) from error

    except requests.Timeout as error:
        raise RuntimeError(
            "La requête vers FastAPI a dépassé le délai."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Erreur réseau : {error}"
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


def check_service(
    url: str,
    timeout: int = 5,
) -> bool:
    try:
        response = requests.get(
            url,
            timeout=timeout,
        )

        return response.ok

    except requests.RequestException:
        return False
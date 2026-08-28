"""
API FastAPI du système IA de trading.

Endpoints :
- GET  /health
- GET  /model
- GET  /predict/latest/{symbol}/{timeframe}
- GET  /predict/at/{symbol}/{timeframe}
- GET  /predict/range/{symbol}/{timeframe}
- POST /risk/calculate
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.risk.risk_manager import calculate_risk
from src.training.base_trainer import BaseTrainer


APP_VERSION = "3.1.0"

DEFAULT_MODEL_PATH = (
    "models/GC/5m/random_forest/random_forest_v3.pkl"
)

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
)

SIGNAL_MAPPING = {
    -1: "SELL",
    0: "NO_TRADE",
    1: "BUY",
}


app = FastAPI(
    title="AI Trading System API",
    version=APP_VERSION,
)


class RiskInput(BaseModel):
    signal: str

    account_balance: float = Field(
        gt=0
    )

    risk_percent: float = Field(
        gt=0,
        le=100,
    )

    entry_price: float = Field(
        gt=0
    )

    atr_14: float = Field(
        gt=0
    )

    confidence: float = Field(
        ge=0,
        le=1,
    )

    risk_reward_ratio: float = Field(
        default=2.0,
        gt=0,
    )


@lru_cache(maxsize=1)
def load_model_package() -> dict[str, Any]:
    """
    Charge le bundle du modèle actif.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    package = joblib.load(
        MODEL_PATH
    )

    if not isinstance(package, dict):
        raise ValueError(
            "Le fichier modèle doit contenir un dictionnaire."
        )

    required_keys = {
        "model",
        "features",
    }

    missing_keys = required_keys.difference(
        package.keys()
    )

    if missing_keys:
        raise ValueError(
            f"Clés manquantes : {missing_keys}"
        )

    return package


def get_dataset_path(
    symbol: str,
    timeframe: str,
) -> Path:
    """
    Retourne le chemin du dataset Analysis V3.
    """

    normalized_symbol = symbol.upper()
    normalized_timeframe = timeframe.lower()

    return (
        Path("data/final/analysis")
        / normalized_symbol
        / (
            f"dataset_analysis_v3_"
            f"{normalized_symbol}_{normalized_timeframe}.csv"
        )
    )


def load_analysis_dataset(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """
    Charge et valide le dataset Analysis V3.
    """

    dataset_path = get_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset introuvable : {dataset_path}",
        )

    analysis_df = pd.read_csv(
        dataset_path
    )

    if analysis_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Le dataset est vide.",
        )

    if "datetime" not in analysis_df.columns:
        raise HTTPException(
            status_code=400,
            detail="La colonne datetime est absente.",
        )

    if "close" not in analysis_df.columns:
        raise HTTPException(
            status_code=400,
            detail="La colonne close est absente.",
        )

    analysis_df["datetime"] = pd.to_datetime(
        analysis_df["datetime"],
        utc=True,
        errors="coerce",
    )

    analysis_df = analysis_df.dropna(
        subset=["datetime"]
    ).reset_index(drop=True)

    return analysis_df


def prepare_model_features(
    analysis_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    expected_features: list[str],
) -> pd.DataFrame:
    """
    Reconstruit les features exactement comme pendant
    l'entraînement du modèle.
    """

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="api",
    )

    X, _ = trainer.prepare_tabular_features(
        analysis_df
    )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in X.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Features manquantes",
                "features": missing_features,
            },
        )

    return X[
        expected_features
    ].copy()


def predict_dataframe(
    model: Any,
    features_df: pd.DataFrame,
) -> tuple[list[int], list[float | None]]:
    """
    Produit les prédictions et les confiances.
    """

    raw_predictions = model.predict(
        features_df
    )

    predictions = [
        int(value)
        for value in raw_predictions
    ]

    confidences: list[float | None] = [
        None
    ] * len(predictions)

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = model.predict_proba(
            features_df
        )

        confidences = [
            float(row.max())
            for row in probabilities
        ]

    return predictions, confidences


def build_prediction_result(
    row: pd.Series,
    prediction: int,
    confidence: float | None,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    """
    Construit la réponse JSON d'une prédiction.
    """

    result = {
        "symbol": symbol.upper(),
        "timeframe": timeframe.lower(),
        "datetime": str(
            row["datetime"]
        ),
        "close": float(
            row["close"]
        ),
        "prediction": prediction,
        "signal": SIGNAL_MAPPING.get(
            prediction,
            "UNKNOWN",
        ),
        "confidence": (
            round(confidence, 4)
            if confidence is not None
            else None
        ),
    }

    if "target" in row.index:
        result["actual_target"] = int(
            row["target"]
        )

    if "future_return_1" in row.index:
        result["future_return_1"] = float(
            row["future_return_1"]
        )

    return result


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/model")
def model_information() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        return {
            "loaded": False,
            "path": str(MODEL_PATH),
        }

    package = load_model_package()

    return {
        "loaded": True,
        "path": str(MODEL_PATH),
        "model_type": type(
            package["model"]
        ).__name__,
        "symbol": package.get(
            "symbol"
        ),
        "timeframe": package.get(
            "timeframe"
        ),
        "feature_count": len(
            package["features"]
        ),
        "split_type": package.get(
            "split_type"
        ),
    }


@app.get("/predict/latest/{symbol}/{timeframe}")
def predict_latest(
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    """
    Produit une prédiction sur la dernière bougie.
    """

    package = load_model_package()

    analysis_df = load_analysis_dataset(
        symbol=symbol,
        timeframe=timeframe,
    )

    X = prepare_model_features(
        analysis_df=analysis_df,
        symbol=symbol,
        timeframe=timeframe,
        expected_features=package["features"],
    )

    latest_features = X.iloc[
        [-1]
    ]

    predictions, confidences = predict_dataframe(
        model=package["model"],
        features_df=latest_features,
    )

    latest_row = analysis_df.iloc[
        -1
    ]

    return build_prediction_result(
        row=latest_row,
        prediction=predictions[0],
        confidence=confidences[0],
        symbol=symbol,
        timeframe=timeframe,
    )


@app.get("/predict/at/{symbol}/{timeframe}")
def predict_at(
    symbol: str,
    timeframe: str,
    datetime_value: str = Query(
        alias="datetime",
        description=(
            "Date UTC exacte. "
            "Exemple : 2026-07-10T20:30:00+00:00"
        ),
    ),
) -> dict[str, Any]:
    """
    Produit une prédiction pour une bougie précise.
    """

    package = load_model_package()

    analysis_df = load_analysis_dataset(
        symbol=symbol,
        timeframe=timeframe,
    )

    requested_datetime = pd.to_datetime(
        datetime_value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(
        requested_datetime
    ):
        raise HTTPException(
            status_code=422,
            detail="Format de date invalide.",
        )

    matching_indices = analysis_df.index[
        analysis_df["datetime"]
        == requested_datetime
    ].tolist()

    if not matching_indices:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune bougie trouvée pour cette date."
            ),
        )

    source_index = matching_indices[0]

    X = prepare_model_features(
        analysis_df=analysis_df,
        symbol=symbol,
        timeframe=timeframe,
        expected_features=package["features"],
    )

    selected_features = X.loc[
        [source_index]
    ]

    predictions, confidences = predict_dataframe(
        model=package["model"],
        features_df=selected_features,
    )

    selected_row = analysis_df.loc[
        source_index
    ]

    return build_prediction_result(
        row=selected_row,
        prediction=predictions[0],
        confidence=confidences[0],
        symbol=symbol,
        timeframe=timeframe,
    )


@app.get("/predict/range/{symbol}/{timeframe}")
def predict_range(
    symbol: str,
    timeframe: str,
    start: str = Query(
        description=(
            "Date de début UTC."
        )
    ),
    end: str = Query(
        description=(
            "Date de fin UTC."
        )
    ),
    limit: int = Query(
        default=500,
        ge=1,
        le=5000,
    ),
) -> dict[str, Any]:
    """
    Produit plusieurs prédictions sur une période historique.
    """

    package = load_model_package()

    analysis_df = load_analysis_dataset(
        symbol=symbol,
        timeframe=timeframe,
    )

    start_datetime = pd.to_datetime(
        start,
        utc=True,
        errors="coerce",
    )

    end_datetime = pd.to_datetime(
        end,
        utc=True,
        errors="coerce",
    )

    if pd.isna(
        start_datetime
    ) or pd.isna(
        end_datetime
    ):
        raise HTTPException(
            status_code=422,
            detail="Date de début ou de fin invalide.",
        )

    if start_datetime > end_datetime:
        raise HTTPException(
            status_code=422,
            detail=(
                "La date de début doit précéder "
                "la date de fin."
            ),
        )

    selected_df = analysis_df[
        analysis_df["datetime"].between(
            start_datetime,
            end_datetime,
        )
    ].copy()

    if selected_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune donnée trouvée sur cette période."
            ),
        )

    selected_df = selected_df.tail(
        limit
    )

    X = prepare_model_features(
        analysis_df=analysis_df,
        symbol=symbol,
        timeframe=timeframe,
        expected_features=package["features"],
    )

    selected_features = X.loc[
        selected_df.index
    ]

    predictions, confidences = predict_dataframe(
        model=package["model"],
        features_df=selected_features,
    )

    results = []

    for position, source_index in enumerate(
        selected_df.index
    ):
        row = selected_df.loc[
            source_index
        ]

        results.append(
            build_prediction_result(
                row=row,
                prediction=predictions[position],
                confidence=confidences[position],
                symbol=symbol,
                timeframe=timeframe,
            )
        )

    signal_counts = {
        "BUY": sum(
            result["signal"] == "BUY"
            for result in results
        ),
        "SELL": sum(
            result["signal"] == "SELL"
            for result in results
        ),
        "NO_TRADE": sum(
            result["signal"] == "NO_TRADE"
            for result in results
        ),
    }

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.lower(),
        "start": str(
            start_datetime
        ),
        "end": str(
            end_datetime
        ),
        "count": len(
            results
        ),
        "signal_counts": signal_counts,
        "predictions": results,
    }


@app.post("/risk/calculate")
def risk_calculation(
    data: RiskInput,
) -> Any:
    normalized_signal = data.signal.upper()

    if normalized_signal not in {
        "BUY",
        "SELL",
        "NO_TRADE",
    }:
        raise HTTPException(
            status_code=422,
            detail=(
                "Le signal doit être BUY, SELL "
                "ou NO_TRADE."
            ),
        )

    return calculate_risk(
        signal=normalized_signal,
        account_balance=data.account_balance,
        risk_percent=data.risk_percent,
        entry_price=data.entry_price,
        atr_14=data.atr_14,
        confidence=data.confidence,
        risk_reward_ratio=data.risk_reward_ratio,
    )
"""
API FastAPI du système IA de trading.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.risk.risk_manager import calculate_risk
from src.training.base_trainer import BaseTrainer


APP_VERSION = "3.0.0"

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
    account_balance: float = Field(gt=0)
    risk_percent: float = Field(gt=0, le=100)
    entry_price: float = Field(gt=0)
    atr_14: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    risk_reward_ratio: float = Field(default=2.0, gt=0)


@lru_cache(maxsize=1)
def load_model_package() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    required_keys = {"model", "features"}

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
    return (
        Path("data/final/analysis")
        / symbol.upper()
        / (
            f"dataset_analysis_v3_"
            f"{symbol.upper()}_{timeframe.lower()}.csv"
        )
    )


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
        "symbol": package.get("symbol"),
        "timeframe": package.get("timeframe"),
        "feature_count": len(
            package["features"]
        ),
    }


@app.get("/predict/latest/{symbol}/{timeframe}")
def predict_latest(
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    package = load_model_package()

    dataset_path = get_dataset_path(
        symbol=symbol,
        timeframe=timeframe,
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dataset introuvable : {dataset_path}",
        )

    analysis_df = pd.read_csv(dataset_path)

    if analysis_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Le dataset est vide.",
        )

    trainer = BaseTrainer(
        symbol=symbol,
        timeframe=timeframe,
        model_name="api",
    )

    X, _ = trainer.prepare_tabular_features(
        analysis_df
    )

    expected_features = package["features"]

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

    latest_features = X[
        expected_features
    ].iloc[[-1]]

    model = package["model"]

    prediction = int(
        model.predict(latest_features)[0]
    )

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            latest_features
        )[0]

        confidence = float(
            probabilities.max()
        )

    latest_row = analysis_df.iloc[-1]

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.lower(),
        "datetime": latest_row.get("datetime"),
        "close": float(latest_row["close"]),
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


@app.post("/risk/calculate")
def risk_calculation(
    data: RiskInput,
) -> Any:
    return calculate_risk(
        signal=data.signal.upper(),
        account_balance=data.account_balance,
        risk_percent=data.risk_percent,
        entry_price=data.entry_price,
        atr_14=data.atr_14,
        confidence=data.confidence,
        risk_reward_ratio=data.risk_reward_ratio,
    )
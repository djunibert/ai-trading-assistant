"""
Tests du moteur de signal.
"""

from src.trading.signal_engine import SignalEngine


def test_generate_buy_signal() -> None:
    engine = SignalEngine(
        minimum_confidence=0.50,
        minimum_score_difference=0.10,
    )

    result = engine.generate_signal(
        symbol="GC",
        timeframe="5m",
        predictions={
            "random_forest": 1,
            "xgboost": 1,
            "lstm": 0,
        },
        confidences={
            "random_forest": 0.70,
            "xgboost": 0.80,
            "lstm": 0.60,
        },
        model_weights={
            "random_forest": 1.0,
            "xgboost": 1.5,
            "lstm": 0.5,
        },
        market_structure_score=70,
        buy_setup_score=80,
        sell_setup_score=10,
        risk_trade_allowed=True,
    )

    assert result.signal == "BUY"
    assert result.confidence >= 0.50


def test_trade_rejected_by_risk_engine() -> None:
    engine = SignalEngine()

    result = engine.generate_signal(
        symbol="GC",
        timeframe="15m",
        predictions={
            "random_forest": 1,
            "xgboost": 1,
        },
        risk_trade_allowed=False,
    )

    assert result.signal == "NO_TRADE"
"""
Moteur de signal de trading multi-modèles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class TradingSignal:
    """
    Résultat final produit par le moteur de signal.
    """

    symbol: str
    timeframe: str
    signal: str
    confidence: float
    buy_score: float
    sell_score: float
    reason: str


class SignalEngine:
    """
    Combine les prédictions de plusieurs modèles.

    Valeurs attendues :
    -1 = SELL
     0 = NO_TRADE
     1 = BUY
    """

    def __init__(
        self,
        minimum_confidence: float = 0.60,
        minimum_score_difference: float = 0.15,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError(
                "minimum_confidence doit être entre 0 et 1."
            )

        if minimum_score_difference < 0:
            raise ValueError(
                "minimum_score_difference doit être positif."
            )

        self.minimum_confidence = minimum_confidence
        self.minimum_score_difference = minimum_score_difference

    def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        predictions: Mapping[str, int],
        confidences: Mapping[str, float] | None = None,
        model_weights: Mapping[str, float] | None = None,
        market_structure_score: float = 0.0,
        buy_setup_score: float = 0.0,
        sell_setup_score: float = 0.0,
        risk_trade_allowed: bool = True,
    ) -> TradingSignal:
        """
        Génère un signal final.
        """

        if not predictions:
            raise ValueError(
                "Au moins une prédiction est requise."
            )

        self._validate_predictions(predictions)

        confidences = confidences or {}
        model_weights = model_weights or {}

        if not risk_trade_allowed:
            return TradingSignal(
                symbol=symbol.upper(),
                timeframe=timeframe.lower(),
                signal="NO_TRADE",
                confidence=0.0,
                buy_score=0.0,
                sell_score=0.0,
                reason="Trade refusé par le moteur de risque.",
            )

        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0

        for model_name, prediction in predictions.items():
            weight = float(
                model_weights.get(model_name, 1.0)
            )

            confidence = float(
                confidences.get(model_name, 1.0)
            )

            confidence = min(
                max(confidence, 0.0),
                1.0,
            )

            weighted_score = weight * confidence
            total_weight += weight

            if prediction == 1:
                buy_score += weighted_score

            elif prediction == -1:
                sell_score += weighted_score

        if total_weight <= 0:
            raise ValueError(
                "Le poids total des modèles doit être supérieur à zéro."
            )

        buy_score /= total_weight
        sell_score /= total_weight

        buy_structure_bonus = (
            max(buy_setup_score, 0.0) / 100
            + max(market_structure_score, 0.0) / 100
        ) * 0.20

        sell_structure_bonus = (
            max(sell_setup_score, 0.0) / 100
            + max(-market_structure_score, 0.0) / 100
        ) * 0.20

        buy_score = min(
            buy_score + buy_structure_bonus,
            1.0,
        )

        sell_score = min(
            sell_score + sell_structure_bonus,
            1.0,
        )

        score_difference = abs(
            buy_score - sell_score
        )

        if score_difference < self.minimum_score_difference:
            return TradingSignal(
                symbol=symbol.upper(),
                timeframe=timeframe.lower(),
                signal="NO_TRADE",
                confidence=max(buy_score, sell_score),
                buy_score=buy_score,
                sell_score=sell_score,
                reason="Différence insuffisante entre BUY et SELL.",
            )

        if buy_score > sell_score:
            signal = "BUY"
            confidence = buy_score
        else:
            signal = "SELL"
            confidence = sell_score

        if confidence < self.minimum_confidence:
            return TradingSignal(
                symbol=symbol.upper(),
                timeframe=timeframe.lower(),
                signal="NO_TRADE",
                confidence=confidence,
                buy_score=buy_score,
                sell_score=sell_score,
                reason="Confiance inférieure au seuil minimum.",
            )

        return TradingSignal(
            symbol=symbol.upper(),
            timeframe=timeframe.lower(),
            signal=signal,
            confidence=confidence,
            buy_score=buy_score,
            sell_score=sell_score,
            reason="Consensus des modèles et de la structure de marché.",
        )

    @staticmethod
    def _validate_predictions(
        predictions: Mapping[str, int],
    ) -> None:
        """
        Vérifie que les prédictions valent -1, 0 ou 1.
        """

        valid_values = {-1, 0, 1}

        invalid_predictions = {
            model_name: prediction
            for model_name, prediction in predictions.items()
            if prediction not in valid_values
        }

        if invalid_predictions:
            raise ValueError(
                f"Prédictions invalides : {invalid_predictions}"
            )
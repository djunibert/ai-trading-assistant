"""
Liquidity Pool Engine

Détecte et suit les zones de liquidité.

Auteur : Junior Hébert
Projet : AI Trading System
"""

from __future__ import annotations

import pandas as pd

from src.market_structure.models.liquidity_pool import LiquidityPool


class LiquidityPoolEngine:
    """
    Détecte les Buy Side et Sell Side Liquidity Pools.

    Les pools sont créés à partir des Equal High / Equal Low.
    """

    def __init__(self):
        self.active_pool: LiquidityPool | None = None

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # ==========================================================
        # Colonnes créées
        # ==========================================================

        df["liquidity_pool_direction"] = "NONE"
        df["liquidity_pool_price"] = 0.0
        df["liquidity_pool_strength"] = 0
        df["liquidity_pool_age"] = 0
        df["liquidity_pool_distance"] = 0.0
        df["liquidity_pool_swept"] = False
        df["liquidity_pool_active"] = False
        df["liquidity_pool_score"] = 0.0

        self.active_pool = None

        # ==========================================================
        # Parcours des bougies
        # ==========================================================

        for i in range(len(df)):

            # ------------------------------------------------------
            # Création Buy Side Liquidity
            # ------------------------------------------------------

            if bool(df.loc[i, "equal_high"]):

                self.active_pool = LiquidityPool(
                    direction="BUY_SIDE",
                    price=float(df.loc[i, "last_swing_high"]),
                    created_index=i,
                    strength=2,
                )

            # ------------------------------------------------------
            # Création Sell Side Liquidity
            # ------------------------------------------------------

            elif bool(df.loc[i, "equal_low"]):

                self.active_pool = LiquidityPool(
                    direction="SELL_SIDE",
                    price=float(df.loc[i, "last_swing_low"]),
                    created_index=i,
                    strength=2,
                )

            # ------------------------------------------------------
            # Aucun pool actif
            # ------------------------------------------------------

            if self.active_pool is None:
                continue

            if not self.active_pool.active:
                continue

            # ------------------------------------------------------
            # Mise à jour
            # ------------------------------------------------------

            self.active_pool.age = i - self.active_pool.created_index

            high = float(df.loc[i, "high"])
            low = float(df.loc[i, "low"])
            close = float(df.loc[i, "close"])

            # ------------------------------------------------------
            # Buy Side
            # ------------------------------------------------------

            if self.active_pool.direction == "BUY_SIDE":

                self.active_pool.swept = (
                    high > self.active_pool.price
                    and close < self.active_pool.price
                )

                distance = self.active_pool.price - close

            # ------------------------------------------------------
            # Sell Side
            # ------------------------------------------------------

            else:

                self.active_pool.swept = (
                    low < self.active_pool.price
                    and close > self.active_pool.price
                )

                distance = close - self.active_pool.price

            # ------------------------------------------------------
            # Si sweep alors le pool devient inactif
            # ------------------------------------------------------

            if self.active_pool.swept:
                self.active_pool.active = False

            # ------------------------------------------------------
            # Calcul du score
            # ------------------------------------------------------

            score = 0

            if self.active_pool.active:
                score += 25

            if self.active_pool.strength >= 2:
                score += 25

            if self.active_pool.age <= 20:
                score += 25

            if not self.active_pool.swept:
                score += 25

            self.active_pool.score = score

            # ------------------------------------------------------
            # Sauvegarde dans le DataFrame
            # ------------------------------------------------------

            df.loc[i, "liquidity_pool_direction"] = (
                self.active_pool.direction
            )

            df.loc[i, "liquidity_pool_price"] = (
                self.active_pool.price
            )

            df.loc[i, "liquidity_pool_strength"] = (
                self.active_pool.strength
            )

            df.loc[i, "liquidity_pool_age"] = (
                self.active_pool.age
            )

            df.loc[i, "liquidity_pool_distance"] = (
                distance
            )

            df.loc[i, "liquidity_pool_swept"] = (
                self.active_pool.swept
            )

            df.loc[i, "liquidity_pool_active"] = (
                self.active_pool.active
            )

            df.loc[i, "liquidity_pool_score"] = (
                self.active_pool.score
            )

        return df
"""
Order Block Engine V2.

Détecte des Order Blocks avec :
- zone top / bottom
- taille
- distance au prix
- mitigation
- confluence avec BOS
- confluence avec FVG
- confluence avec Liquidity Sweep
- score de qualité
"""

import pandas as pd


class OrderBlockEngine:
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ob_v2"] = 0
        df["ob_v2_direction"] = "NONE"
        df["ob_v2_top"] = None
        df["ob_v2_bottom"] = None
        df["ob_v2_size"] = 0.0
        df["ob_v2_size_atr"] = 0.0
        df["ob_v2_distance"] = 0.0
        df["ob_v2_mitigated"] = False
        df["ob_v2_confluence_bos"] = False
        df["ob_v2_confluence_fvg"] = False
        df["ob_v2_confluence_liquidity"] = False
        df["ob_v2_score"] = 0.0

        for i in range(self.lookback, len(df)):

            # Bullish Order Block :
            # dernière bougie bearish avant un BOS bullish
            if df.loc[i, "bos_v2"] == 1:
                recent = df.iloc[i - self.lookback:i]
                bearish_candles = recent[recent["close"] < recent["open"]]

                if not bearish_candles.empty:
                    ob = bearish_candles.iloc[-1]

                    top = ob["high"]
                    bottom = ob["low"]

                    df.loc[i, "ob_v2"] = 1
                    df.loc[i, "ob_v2_direction"] = "BULLISH"
                    df.loc[i, "ob_v2_top"] = top
                    df.loc[i, "ob_v2_bottom"] = bottom
                    df.loc[i, "ob_v2_size"] = top - bottom

            # Bearish Order Block :
            # dernière bougie bullish avant un BOS bearish
            if df.loc[i, "bos_v2"] == -1:
                recent = df.iloc[i - self.lookback:i]
                bullish_candles = recent[recent["close"] > recent["open"]]

                if not bullish_candles.empty:
                    ob = bullish_candles.iloc[-1]

                    top = ob["high"]
                    bottom = ob["low"]

                    df.loc[i, "ob_v2"] = -1
                    df.loc[i, "ob_v2_direction"] = "BEARISH"
                    df.loc[i, "ob_v2_top"] = top
                    df.loc[i, "ob_v2_bottom"] = bottom
                    df.loc[i, "ob_v2_size"] = top - bottom

        # Taille normalisée par ATR
        df["ob_v2_size_atr"] = (
            df["ob_v2_size"] / df["atr_14"]
        ).replace([float("inf"), -float("inf")], 0).fillna(0)

        # Distance au prix actuel
        df["ob_v2_midpoint"] = (
            pd.to_numeric(df["ob_v2_top"], errors="coerce")
            + pd.to_numeric(df["ob_v2_bottom"], errors="coerce")
        ) / 2

        df["ob_v2_distance"] = (
            df["close"] - df["ob_v2_midpoint"]
        ).abs().fillna(0)

        # Mitigation :
        # bullish OB mitigated si le prix revient toucher la zone
        # bearish OB mitigated si le prix revient toucher la zone
        df["last_ob_top"] = (
            pd.to_numeric(df["ob_v2_top"], errors="coerce")
            .where(df["ob_v2"] != 0)
            .ffill()
        )

        df["last_ob_bottom"] = (
            pd.to_numeric(df["ob_v2_bottom"], errors="coerce")
            .where(df["ob_v2"] != 0)
            .ffill()
        )

        df["last_ob_direction"] = (
            df["ob_v2_direction"]
            .where(df["ob_v2"] != 0)
            .replace("NONE", pd.NA)
            .ffill()
            .fillna("NONE")
        )

        df["ob_v2_mitigated"] = (
            (df["last_ob_direction"] == "BULLISH")
            & (df["low"] <= df["last_ob_top"])
            & (df["low"] >= df["last_ob_bottom"])
        ) | (
            (df["last_ob_direction"] == "BEARISH")
            & (df["high"] >= df["last_ob_bottom"])
            & (df["high"] <= df["last_ob_top"])
        )

        # Confluences
        df["ob_v2_confluence_bos"] = df["ob_v2"] != 0

        if "fvg" in df.columns:
            df["recent_fvg"] = (
                df["fvg"]
                .ne(0)
                .rolling(window=5)
                .max()
                .fillna(False)
                .astype(bool)
            )
            df["ob_v2_confluence_fvg"] = (
                (df["ob_v2"] != 0)
                & df["recent_fvg"]
            )

        if "liquidity_sweep" in df.columns:
            df["recent_liquidity_sweep_for_ob"] = (
                df["liquidity_sweep"]
                .ne(0)
                .rolling(window=5)
                .max()
                .fillna(False)
                .astype(bool)
            )
            df["ob_v2_confluence_liquidity"] = (
                (df["ob_v2"] != 0)
                & df["recent_liquidity_sweep_for_ob"]
            )

        # Score
        df.loc[df["ob_v2"] != 0, "ob_v2_score"] += 20
        df.loc[df["ob_v2_size_atr"].between(0.2, 2.0), "ob_v2_score"] += 20
        df.loc[df["ob_v2_confluence_bos"], "ob_v2_score"] += 20
        df.loc[df["ob_v2_confluence_fvg"], "ob_v2_score"] += 20
        df.loc[df["ob_v2_confluence_liquidity"], "ob_v2_score"] += 20

        df.loc[df["ob_v2"] == 0, "ob_v2_score"] = 0

        return df
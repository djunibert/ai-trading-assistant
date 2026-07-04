"""
Détection des sessions de trading.

Sessions en heure de New York :
- Asia
- London
- New York
- London/New York overlap
"""

import pandas as pd


class SessionDetector:
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month

        df["asian_session"] = df["hour"].between(18, 23) | df["hour"].between(0, 2)
        df["london_session"] = df["hour"].between(3, 11)
        df["new_york_session"] = df["hour"].between(8, 16)
        df["london_new_york_overlap"] = df["hour"].between(8, 11)

        df["session"] = "OTHER"

        df.loc[df["asian_session"], "session"] = "ASIA"
        df.loc[df["london_session"], "session"] = "LONDON"
        df.loc[df["new_york_session"], "session"] = "NEW_YORK"
        df.loc[df["london_new_york_overlap"], "session"] = "LONDON_NY_OVERLAP"

        return df
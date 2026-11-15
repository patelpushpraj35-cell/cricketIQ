"""Role-specific view of the shared six-category IPL ranking engine."""

import pandas as pd

from engine.ipl_player_ranking import calculate_ipl_player_rankings


def calculate_bowling_ipl_ratings(balls: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    rankings = calculate_ipl_player_rankings(balls, matches)
    bowlers = [player for player in rankings["players"] if player["bowling"]["wickets"] is not None]
    return pd.json_normalize(bowlers, sep="_")

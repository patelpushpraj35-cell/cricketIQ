"""Role-specific view of the shared six-category IPL ranking engine."""

import pandas as pd

from engine.ipl_player_ranking import calculate_ipl_player_rankings


def calculate_batting_ipl_ratings(balls: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    rankings = calculate_ipl_player_rankings(balls, matches)
    batters = [player for player in rankings["players"] if player["batting"]["runs"] is not None]
    return pd.json_normalize(batters, sep="_")

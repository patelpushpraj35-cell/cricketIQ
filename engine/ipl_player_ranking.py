"""Coverage-aware IPL player rankings built from the supplied ball-by-ball data."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import re
import json
from typing import Any

import numpy as np
import pandas as pd


CATEGORY_WEIGHTS = OrderedDict([
    ("traditional", 0.15),
    ("pressure", 0.25),
    ("impact", 0.17),
    ("important_matches", 0.15),
    ("consistency", 0.10),
    ("game_intelligence", 0.08),
])

ROLE_RATE_COMPONENT_WEIGHT = {
    "batting": 0.10,
    "bowling": 0.10,
}

ROLE_FRAMEWORK_TOTALS_PERCENT = {
    "batter": 100.0,
    "bowler": 100.0,
    "all_rounder": 100.0,
}
ALL_ROUNDER_ROLE_BLEND = 0.50

ROLE_LEADERBOARD_MINIMUMS = {
    "batting_runs": 1000,
    "bowling_wickets": 50,
}

CATEGORY_NAMES = {
    "traditional": "Traditional Performance Statistics",
    "pressure": "Match Situation & Pressure Performance",
    "impact": "High-Impact & Match-Winning Performance",
    "important_matches": "Important Match Performance",
    "consistency": "Consistency & Adaptability",
    "game_intelligence": "Advanced Game Intelligence & Fielding",
}

MASTER_SCORE_NAMES = {
    "consistency_score": "Consistency Score",
    "pressure_score": "Pressure Performance Score",
    "match_impact_score": "Match Impact Score",
}

# Each entry is (metric key, role, higher-is-better). Unsupported source fields
# are included in coverage denominators but never imputed or scored.
CATEGORY_METRICS = {
    "traditional": [
        ("bat_runs_per_match", "batting", True),
        ("bat_average", "batting", True),
        ("bat_fifties_per_innings", "batting", True),
        ("bat_hundreds_per_innings", "batting", True),
        ("bat_boundary_run_pct", "batting", True),
        ("bat_fours_per_innings", "batting", True),
        ("bat_sixes_per_innings", "batting", True),
        ("bowl_wickets_per_innings", "bowling", True),
        ("bowl_average", "bowling", False),
        ("bowl_strike_rate", "bowling", False),
        ("bowl_dot_pct", "bowling", True),
        ("field_catches", "fielding", True),
        ("field_runouts", "fielding", True),
        ("field_stumpings", "fielding", True),
    ],
    "pressure": [
        ("bat_high_rrr_sr", "batting", True),
        ("bat_low_rrr_sr", "batting", True),
        ("bat_low_wickets_sr", "batting", True),
        ("bat_close_finish_runs_per_match", "batting", True),
        ("bat_last_over_sr", "batting", True),
        ("bat_recovery_sr", "batting", True),
        ("bowl_high_rrr_economy", "bowling", False),
        ("bowl_low_rrr_economy", "bowling", False),
        ("bowl_low_wickets_wickets_per_match", "bowling", True),
        ("bowl_close_finish_economy", "bowling", False),
        ("bowl_last_over_economy", "bowling", False),
        ("bowl_low_total_defenses", "bowling", True),
    ],
    "impact": [
        ("bat_winning_runs_per_match", "batting", True),
        ("bat_17_run_final_overs", "batting", True),
        ("bat_player_of_match", "batting", True),
        ("bowl_winning_wickets_per_match", "bowling", True),
        ("bowl_under_10_defenses", "bowling", True),
        ("bowl_well_set_dismissals", "bowling", True),
        ("bowl_partnership_breaks", "bowling", True),
        ("bowl_player_of_match", "bowling", True),
        ("field_runs_saved", "fielding", True),
    ],
    "important_matches": [
        ("bat_important_runs_per_innings", "batting", True),
        ("bat_important_strike_rate", "batting", True),
        ("bowl_important_wickets_per_innings", "bowling", True),
        ("bowl_important_economy", "bowling", False),
        ("important_match_contributions", "shared", True),
        ("must_win_performance", "shared", True),
    ],
    "consistency": [
        ("bat_last5_runs_per_innings", "batting", True),
        ("bat_last10_runs_per_innings", "batting", True),
        ("bat_season_consistency", "batting", True),
        ("bat_home_away_balance", "batting", True),
        ("bat_strong_opposition_runs_per_match", "batting", True),
        ("bat_positions_played", "batting", True),
        ("bat_adapt_pitch", "batting", True),
        ("bowl_last5_wickets_per_match", "bowling", True),
        ("bowl_last10_wickets_per_match", "bowling", True),
        ("bowl_season_consistency", "bowling", True),
        ("bowl_home_away_balance", "bowling", True),
        ("bowl_strong_opposition_wickets_per_match", "bowling", True),
        ("bowl_phase_roles_played", "bowling", True),
        ("bowl_adapt_pitch", "bowling", True),
    ],
    "game_intelligence": [
        ("bat_powerplay_sr", "batting", True),
        ("bat_middle_overs_sr", "batting", True),
        ("bat_death_overs_sr", "batting", True),
        ("bat_vs_pace_sr", "batting", True),
        ("bat_vs_spin_sr", "batting", True),
        ("bowl_powerplay_economy", "bowling", False),
        ("bowl_middle_overs_economy", "bowling", False),
        ("bowl_death_overs_economy", "bowling", False),
        ("bowl_vs_left_handed_economy", "bowling", False),
        ("bowl_vs_right_handed_economy", "bowling", False),
        ("bat_failure_rate", "batting", False),
        ("bowl_expensive_spell_rate", "bowling", False),
        ("field_dropped_catches", "fielding", False),
    ],
}

MASTER_METRICS = {
    "consistency_score": [
        ("bat_average", "batting", True),
        ("bat_last5_runs_per_innings", "batting", True),
        ("bat_last10_runs_per_innings", "batting", True),
        ("bat_season_consistency", "batting", True),
        ("bat_home_away_balance", "batting", True),
        ("bat_positions_played", "batting", True),
        ("bowl_wickets_per_innings", "bowling", True),
        ("bowl_last5_wickets_per_match", "bowling", True),
        ("bowl_last10_wickets_per_match", "bowling", True),
        ("bowl_season_consistency", "bowling", True),
        ("bowl_home_away_balance", "bowling", True),
        ("bowl_phase_roles_played", "bowling", True),
    ],
    "pressure_score": [
        ("bat_high_rrr_sr", "batting", True),
        ("bat_low_rrr_sr", "batting", True),
        ("bat_low_wickets_sr", "batting", True),
        ("bat_close_finish_runs_per_match", "batting", True),
        ("bat_last_over_sr", "batting", True),
        ("bat_recovery_sr", "batting", True),
        ("bowl_high_rrr_economy", "bowling", False),
        ("bowl_low_rrr_economy", "bowling", False),
        ("bowl_low_wickets_wickets_per_match", "bowling", True),
        ("bowl_close_finish_economy", "bowling", False),
        ("bowl_last_over_economy", "bowling", False),
        ("bowl_low_total_defenses", "bowling", True),
        ("bowl_important_wickets_per_innings", "bowling", True),
        ("bowl_important_economy", "bowling", False),
    ],
    "match_impact_score": [
        ("bat_winning_runs_per_match", "batting", True),
        ("bat_17_run_final_overs", "batting", True),
        ("bat_player_of_match", "batting", True),
        ("bowl_winning_wickets_per_match", "bowling", True),
        ("bowl_under_10_defenses", "bowling", True),
        ("bowl_well_set_dismissals", "bowling", True),
        ("bowl_partnership_breaks", "bowling", True),
        ("bowl_player_of_match", "bowling", True),
        ("field_runs_saved", "fielding", True),
    ],
}

UNAVAILABLE_METRICS = {
    "field_catches": "Fielder identity is not present in the supplied ball-by-ball data.",
    "field_runouts": "Run-out assisting/fielding player identity is not present in the data.",
    "field_stumpings": "Wicketkeeper identity is not present in the supplied data.",
    "field_runs_saved": "Runs saved by fielders are not recorded in the supplied data.",
    "field_dropped_catches": "Dropped catches are not recorded in the supplied data.",
    "bat_adapt_pitch": "Pitch-condition labels are not present in the supplied data.",
    "bowl_adapt_pitch": "Pitch-condition labels are not present in the supplied data.",
    "must_win_performance": "Must-win match labels are not present in the supplied data.",
}


def _clip_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    observed = numeric.dropna()
    if observed.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low, high = observed.quantile([0.05, 0.95]).tolist()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        scaled = pd.Series(50.0, index=values.index, dtype=float)
    else:
        scaled = ((numeric.clip(low, high) - low) / (high - low)) * 100.0
    if not higher_is_better:
        scaled = 100.0 - scaled
    return scaled.clip(0, 100)


def _equal_weight_distribution(keys: list[str], total: float = 100.0) -> dict[str, float]:
    if not keys:
        return {}
    share = total / len(keys)
    weights = {key: share for key in keys[:-1]}
    weights[keys[-1]] = total - sum(weights.values())
    return weights


def _role_parameter_weights_percent(keys: list[str], role: str) -> dict[str, float]:
    return _equal_weight_distribution(keys)


def _category_role_blend_percent(specs: list[tuple[str, str, bool]]) -> dict[str, float]:
    roles = list(dict.fromkeys(
        role for _, role, _ in specs if role in {"batting", "bowling", "shared"}
    ))
    if "shared" in roles and "batting" in roles and "bowling" in roles:
        return {"batting": 25.0, "bowling": 25.0, "shared": 50.0}
    return _equal_weight_distribution(roles)


def _role_path_category_blend_percent(
    specs: list[tuple[str, str, bool]],
    role: str,
) -> dict[str, float]:
    roles = {metric_role for _, metric_role, _ in specs}
    if role not in roles:
        return {}
    if "shared" in roles:
        return {role: 50.0, "shared": 50.0}
    return {role: 100.0}


def _equal_metric_mean(scores: pd.DataFrame, keys: list[str]) -> pd.Series:
    return scores[keys].mean(axis=1, skipna=True)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def _add_metric(
    values: dict[str, pd.Series],
    samples: dict[str, pd.Series],
    key: str,
    metric_values: pd.Series,
    metric_samples: pd.Series,
) -> None:
    values[key] = metric_values.astype(float)
    samples[key] = metric_samples.astype(float)


def _group_sum(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    return frame.groupby(key, dropna=True)[column].sum()


def _group_count(frame: pd.DataFrame, key: str, column: str) -> pd.Series:
    return frame.groupby(key, dropna=True)[column].count()


def _balanced_home_away(home: pd.Series, away: pd.Series) -> pd.Series:
    valid = home.notna() & away.notna()
    high = pd.concat([home, away], axis=1).max(axis=1)
    low = pd.concat([home, away], axis=1).min(axis=1)
    balance = 1.0 - (high - low).div(high.where(high > 0))
    return balance.where(valid)


def _match_close(result: Any) -> bool:
    if not isinstance(result, str):
        return False
    runs = re.search(r"won by\s+(\d+)\s+runs?", result, re.IGNORECASE)
    wickets = re.search(r"won\s+(\d+)\s+(?:wickets?|wkts?)", result, re.IGNORECASE)
    return bool((runs and int(runs.group(1)) <= 10) or (wickets and int(wickets.group(1)) <= 2))


def _load_ipl_data(
    data_path: str | Path | None = None,
    matches_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(__file__).resolve().parents[1]
    data_path = Path(data_path) if data_path else root / "bbb_advanced.feather"
    if not data_path.exists():
        fallback = root / "bbb_processed.feather"
        if not fallback.exists():
            raise FileNotFoundError(f"IPL ball-by-ball data not found: {data_path}")
        data_path = fallback
    balls = pd.read_feather(data_path)
    required = {"match_id", "innings", "overid", "striker_id", "striker_name", "bowler_id", "bowler_name", "runs"}
    missing = sorted(required.difference(balls.columns))
    if missing:
        raise ValueError("Ball-by-ball dataset is missing required columns: " + ", ".join(missing))
    if "comp_name" not in balls.columns:
        raise ValueError("Ball-by-ball dataset has no comp_name column; IPL matches cannot be isolated safely.")
    balls = balls[balls["comp_name"].fillna("").astype(str).str.match(r"^IPL \d{4}$")].copy()
    if balls.empty:
        raise ValueError("The supplied ball-by-ball dataset contains no IPL season records.")

    matches_path = Path(matches_path) if matches_path else root / "data_pack" / "Data Pack" / "matches.csv"
    comps_path = matches_path.parent / "competitions.csv"
    if not matches_path.exists() or not comps_path.exists():
        matches = pd.DataFrame(columns=["match_id"])
    else:
        matches = pd.read_csv(matches_path)
        competitions = pd.read_csv(comps_path)
        matches = matches.merge(
            competitions[["comp_id", "comp_name", "season"]],
            on="comp_id",
            how="left",
            suffixes=("", "_competition"),
        )
        matches = matches[matches["comp_name"].fillna("").astype(str).str.match(r"^IPL \d{4}$")].copy()
    return balls, matches


def calculate_ipl_player_rankings(
    balls: pd.DataFrame,
    matches: pd.DataFrame | None = None,
    minimum_matches: int = 5,
) -> dict[str, Any]:
    """Return all-player IPL scores, category data, metric coverage, and metadata."""
    data = balls.copy()
    if data.empty:
        raise ValueError("Cannot calculate IPL rankings from an empty ball-by-ball dataset.")
    if "comp_name" in data.columns:
        data = data[data["comp_name"].fillna("").astype(str).str.match(r"^IPL \d{4}$")].copy()
    if data.empty:
        raise ValueError("No IPL deliveries remain after filtering the input dataset.")

    if matches is not None and not matches.empty:
        metadata_columns = [column for column in (
            "match_id", "home_team_id", "win_team_id", "man_of_the_match",
            "result", "stage", "knock_out", "match_date", "season",
        ) if column in matches.columns]
        match_info = matches[metadata_columns].drop_duplicates("match_id")
        # Match-level source data is authoritative for winners, match stages, and awards.
        data = data.drop(columns=[column for column in metadata_columns if column != "match_id" and column in data.columns], errors="ignore")
        data = data.merge(match_info, on="match_id", how="left")

    if "season" not in data:
        data["season"] = data.get("comp_name", pd.Series("", index=data.index)).astype(str).str.extract(r"(\d{4})")[0]
    data["season"] = data["season"].astype(str)
    if "match_date" in data.columns:
        data["match_date"] = pd.to_datetime(data["match_date"], errors="coerce")
    elif "delivery_ist_time" in data.columns:
        data["match_date"] = pd.to_datetime(data["delivery_ist_time"], errors="coerce", utc=True)
    else:
        data["match_date"] = pd.NaT

    defaults: dict[str, Any] = {
        "extras_type": None, "is_four": False, "is_six": False, "is_wicket": False,
        "is_bowler_wicket": False, "is_legal": None, "striker_dismissed": None,
        "dismissed_player_id": np.nan, "is_bowler_wicket": False, "batting_order": np.nan,
        "pace_or_spin": None, "batting_type": None, "rrr_before": np.nan,
        "wickets_in_hand_before": np.nan, "cum_wickets_before": np.nan,
        "is_any_knockout": None, "knock_out": 0, "stage": None, "win_team_id": np.nan,
        "home_team_id": np.nan, "man_of_the_match": np.nan, "result": None,
        "is_dot_bat": None, "is_dot_bowl": None, "bowler_runs": None,
        "striker_id": np.nan, "striker_name": None, "bowler_id": np.nan, "bowler_name": None,
        "batting_team_id": np.nan, "bowling_team_id": np.nan,
        "ball_runs": None, "runs_needed_before": np.nan, "target": np.nan, "inn1_total": np.nan,
        "event_break_dangerous_stand": 0,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    data["is_legal"] = data["is_legal"].where(data["is_legal"].notna(), ~data["extras_type"].astype(str).str.lower().isin(["wide", "no ball", "noball", "overstep"])).astype(int)
    data["striker_dismissed"] = data["striker_dismissed"].where(
        data["striker_dismissed"].notna(),
        data["is_wicket"].astype(bool) & data["striker_id"].eq(data["dismissed_player_id"]),
    ).astype(int)
    data["is_bowler_wicket"] = data["is_bowler_wicket"].fillna(False).astype(int)
    data["is_four"] = data["is_four"].fillna(False).astype(int)
    data["is_six"] = data["is_six"].fillna(False).astype(int)
    data["runs"] = pd.to_numeric(data["runs"], errors="coerce")
    data["bowler_runs"] = pd.to_numeric(data["bowler_runs"], errors="coerce")
    data["batting_order"] = pd.to_numeric(data["batting_order"], errors="coerce")
    data["bat_balls"] = (~data["extras_type"].astype(str).str.lower().eq("wide")).astype(int)
    data["is_dot_bat"] = data["is_dot_bat"].where(
        data["is_dot_bat"].notna(),
        data["runs"].eq(0) & data["bat_balls"].eq(1),
    ).astype(int)
    data["is_dot_bowl"] = data["is_dot_bowl"].where(
        data["is_dot_bowl"].notna(),
        data["bowler_runs"].eq(0) & data["is_legal"].eq(1),
    ).astype(int)
    if "ball_runs" not in data.columns or data["ball_runs"].isna().all():
        extra_runs = pd.to_numeric(data.get("extras1_runs", 0), errors="coerce").fillna(0)
        extra_runs += pd.to_numeric(data.get("extras_2_runs", 0), errors="coerce").fillna(0)
        data["ball_runs"] = data["runs"].fillna(0) + extra_runs

    # IPL fixture membership, not delivery rows, determines a player's actual match sample.
    player_names: dict[Any, str] = {}
    for id_column, name_column in (("striker_id", "striker_name"), ("bowler_id", "bowler_name")):
        lookup = data[[id_column, name_column]].dropna().drop_duplicates(id_column)
        player_names.update(dict(zip(lookup[id_column], lookup[name_column].astype(str))))
    player_ids = pd.Index(sorted(player_names, key=lambda value: str(value)), name="player_id")
    values: dict[str, pd.Series] = {}
    samples: dict[str, pd.Series] = {}

    # Actual role-specific innings and match tables.
    batting = data[data["striker_id"].notna()].copy()
    batting["bat_balls"] = data.loc[batting.index, "bat_balls"]
    batting["boundary_runs"] = batting["is_four"] * 4 + batting["is_six"] * 6
    batter_innings = batting.groupby(["striker_id", "match_id", "innings"], dropna=True).agg(
        runs=("runs", "sum"),
        balls=("bat_balls", "sum"),
        dismissals=("striker_dismissed", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        boundary_runs=("boundary_runs", "sum"),
        batting_order=("batting_order", "median"),
        match_date=("match_date", "first"),
        season=("season", "first"),
        batting_team_id=("batting_team_id", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        home_team_id=("home_team_id", "first"),
        win_team_id=("win_team_id", "first"),
        man_of_the_match=("man_of_the_match", "first"),
        stage=("stage", "first"),
        knock_out=("knock_out", "max"),
    ).reset_index()
    batter_match = batter_innings.groupby(["striker_id", "match_id"], dropna=True).agg(
        runs=("runs", "sum"),
        balls=("balls", "sum"),
        dismissals=("dismissals", "sum"),
        match_date=("match_date", "first"),
        season=("season", "first"),
        batting_team_id=("batting_team_id", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        home_team_id=("home_team_id", "first"),
        win_team_id=("win_team_id", "first"),
        man_of_the_match=("man_of_the_match", "first"),
        stage=("stage", "first"),
        knock_out=("knock_out", "max"),
    ).reset_index()

    bowling = data[data["bowler_id"].notna()].copy()
    bowler_innings = bowling.groupby(["bowler_id", "match_id", "innings"], dropna=True).agg(
        legal_balls=("is_legal", "sum"),
        runs_conceded=("bowler_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_bowl", "sum") if "is_dot_bowl" in bowling.columns else ("bowler_runs", lambda series: (series == 0).sum()),
        match_date=("match_date", "first"),
        season=("season", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        batting_team_id=("batting_team_id", "first"),
        home_team_id=("home_team_id", "first"),
        win_team_id=("win_team_id", "first"),
        man_of_the_match=("man_of_the_match", "first"),
        stage=("stage", "first"),
        knock_out=("knock_out", "max"),
    ).reset_index()
    bowler_match = bowler_innings.groupby(["bowler_id", "match_id"], dropna=True).agg(
        legal_balls=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        match_date=("match_date", "first"),
        season=("season", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        batting_team_id=("batting_team_id", "first"),
        home_team_id=("home_team_id", "first"),
        win_team_id=("win_team_id", "first"),
        man_of_the_match=("man_of_the_match", "first"),
        stage=("stage", "first"),
        knock_out=("knock_out", "max"),
    ).reset_index()

    # Traditional batting statistics.
    batter_totals = batting.groupby("striker_id").agg(
        balls=("bat_balls", "sum"), runs=("runs", "sum"), outs=("striker_dismissed", "sum"),
        fours=("is_four", "sum"), sixes=("is_six", "sum"),
    )
    batter_totals["matches"] = batter_match.groupby("striker_id")["match_id"].nunique()
    batter_totals["innings"] = batter_innings.groupby("striker_id")["innings"].count()
    batter_totals["boundary_runs"] = batting.groupby("striker_id")["boundary_runs"].sum()
    batter_totals["not_outs"] = (batter_totals["innings"] - batter_totals["outs"]).clip(lower=0)
    batter_totals["avg"] = _safe_divide(batter_totals["runs"], batter_totals["outs"])
    batter_totals["sr"] = _safe_divide(batter_totals["runs"] * 100.0, batter_totals["balls"])
    batter_totals["boundary_pct"] = _safe_divide(batter_totals["boundary_runs"] * 100.0, batter_totals["runs"])
    bat_half_centuries = batter_innings.groupby("striker_id")["runs"].apply(lambda series: ((series >= 50) & (series < 100)).sum())
    bat_centuries = batter_innings.groupby("striker_id")["runs"].apply(lambda series: (series >= 100).sum())
    for key, series, sample in [
        ("bat_runs_per_match", _safe_divide(batter_totals["runs"], batter_totals["matches"]), batter_totals["matches"]),
        ("bat_average", batter_totals["avg"].where(batter_totals["outs"] > 0), batter_totals["outs"]),
        ("bat_strike_rate", batter_totals["sr"], batter_totals["balls"]),
        ("bat_fifties_per_innings", _safe_divide(bat_half_centuries, batter_totals["innings"]), batter_totals["innings"]),
        ("bat_hundreds_per_innings", _safe_divide(bat_centuries, batter_totals["innings"]), batter_totals["innings"]),
        ("bat_boundary_run_pct", batter_totals["boundary_pct"], batter_totals["runs"]),
        ("bat_fours_per_innings", _safe_divide(batter_totals["fours"], batter_totals["innings"]), batter_totals["innings"]),
        ("bat_sixes_per_innings", _safe_divide(batter_totals["sixes"], batter_totals["innings"]), batter_totals["innings"]),
    ]:
        _add_metric(values, samples, key, series, sample)

    # Traditional bowling statistics.
    bowler_totals = bowling.groupby("bowler_id").agg(
        legal_balls=("is_legal", "sum"), runs=("bowler_runs", "sum"),
        wickets=("is_bowler_wicket", "sum"),
        dots=("is_dot_bowl", "sum") if "is_dot_bowl" in bowling.columns else ("bowler_runs", lambda series: (series == 0).sum()),
    )
    bowler_totals["matches"] = bowler_match.groupby("bowler_id")["match_id"].nunique()
    bowler_totals["innings"] = bowler_innings.groupby("bowler_id")["innings"].count()
    bowler_totals["average"] = _safe_divide(bowler_totals["runs"], bowler_totals["wickets"])
    bowler_totals["economy"] = _safe_divide(bowler_totals["runs"] * 6.0, bowler_totals["legal_balls"])
    bowler_totals["sr"] = _safe_divide(bowler_totals["legal_balls"], bowler_totals["wickets"])
    bowler_totals["dot_pct"] = _safe_divide(bowler_totals["dots"] * 100.0, bowler_totals["legal_balls"])
    for key, series, sample in [
        ("bowl_wickets_per_innings", _safe_divide(bowler_totals["wickets"], bowler_totals["innings"]), bowler_totals["innings"]),
        ("bowl_average", bowler_totals["average"].where(bowler_totals["wickets"] > 0), bowler_totals["wickets"]),
        ("bowl_economy", bowler_totals["economy"], bowler_totals["legal_balls"]),
        ("bowl_strike_rate", bowler_totals["sr"].where(bowler_totals["wickets"] > 0), bowler_totals["wickets"]),
        ("bowl_dot_pct", bowler_totals["dot_pct"], bowler_totals["legal_balls"]),
    ]:
        _add_metric(values, samples, key, series, sample)

    def add_delivery_rate(
        key: str,
        frame: pd.DataFrame,
        role_id: str,
        mask: pd.Series,
        numerator: str,
        denominator: str,
        multiplier: float = 1.0,
        numerator_is_sum: bool = True,
    ) -> None:
        subset = frame.loc[mask.fillna(False)]
        if subset.empty:
            _add_metric(values, samples, key, pd.Series(dtype=float), pd.Series(dtype=float))
            return
        numerator_sum = subset.groupby(role_id)[numerator].sum() if numerator_is_sum else subset.groupby(role_id)[numerator].mean()
        denominator_sum = subset.groupby(role_id)[denominator].sum()
        _add_metric(values, samples, key, _safe_divide(numerator_sum * multiplier, denominator_sum), denominator_sum)

    # Situation metrics are directly derived from recorded ball-by-ball game state.
    rrr = pd.to_numeric(data["rrr_before"], errors="coerce")
    wickets_in_hand = pd.to_numeric(data["wickets_in_hand_before"], errors="coerce")
    add_delivery_rate("bat_high_rrr_sr", data, "striker_id", rrr >= 12, "runs", "bat_balls", 100.0)
    add_delivery_rate("bat_low_rrr_sr", data, "striker_id", (rrr < 6) & (data["innings"] == 2), "runs", "bat_balls", 100.0)
    add_delivery_rate("bat_low_wickets_sr", data, "striker_id", wickets_in_hand <= 4, "runs", "bat_balls", 100.0)
    add_delivery_rate("bowl_high_rrr_economy", data, "bowler_id", rrr >= 12, "bowler_runs", "is_legal", 6.0)
    add_delivery_rate("bowl_low_rrr_economy", data, "bowler_id", (rrr < 6) & (data["innings"] == 2), "bowler_runs", "is_legal", 6.0)
    low_wicket_bowling = data.loc[wickets_in_hand <= 4]
    low_wicket_wickets = low_wicket_bowling.groupby("bowler_id")["is_bowler_wicket"].sum()
    low_wicket_matches = low_wicket_bowling.groupby("bowler_id")["match_id"].nunique()
    _add_metric(values, samples, "bowl_low_wickets_wickets_per_match",
                _safe_divide(low_wicket_wickets, low_wicket_matches), low_wicket_matches)

    # Match close finishes are identified only from supplied match result margins.
    if "result" in data.columns:
        close_match_ids = data.loc[data["result"].map(_match_close), "match_id"].dropna().unique()
    else:
        close_match_ids = []
    close_mask = data["match_id"].isin(close_match_ids)
    close_batting = data.loc[close_mask]
    close_runs = close_batting.groupby("striker_id")["runs"].sum()
    close_matches = close_batting.groupby("striker_id")["match_id"].nunique()
    _add_metric(values, samples, "bat_close_finish_runs_per_match",
                _safe_divide(close_runs, close_matches), close_matches)
    add_delivery_rate("bowl_close_finish_economy", data, "bowler_id", close_mask, "bowler_runs", "is_legal", 6.0)

    # The last over is inferred from the last recorded over in each innings (including shortened innings).
    innings_last_over = data.groupby(["match_id", "innings"])["overid"].transform("max")
    last_over_mask = data["overid"].eq(innings_last_over)
    add_delivery_rate("bat_last_over_sr", data, "striker_id", last_over_mask, "runs", "bat_balls", 100.0)
    add_delivery_rate("bowl_last_over_economy", data, "bowler_id", last_over_mask, "bowler_runs", "is_legal", 6.0)

    recovery_mask = (pd.to_numeric(data["overid"], errors="coerce") <= 6) & (pd.to_numeric(data["cum_wickets_before"], errors="coerce") >= 2)
    add_delivery_rate("bat_recovery_sr", data, "striker_id", recovery_mask, "runs", "bat_balls", 100.0)

    # Winning contributions, Player-of-the-Match awards, and knockout records.
    bat_win = batter_match[batter_match["win_team_id"].eq(batter_match["batting_team_id"])]
    bowl_win = bowler_match[bowler_match["win_team_id"].eq(bowler_match["bowling_team_id"])]
    _add_metric(
        values, samples, "bat_winning_runs_per_match",
        _safe_divide(bat_win.groupby("striker_id")["runs"].sum(), bat_win.groupby("striker_id")["match_id"].nunique()),
        bat_win.groupby("striker_id")["match_id"].nunique(),
    )
    _add_metric(
        values, samples, "bowl_winning_wickets_per_match",
        _safe_divide(bowl_win.groupby("bowler_id")["wickets"].sum(), bowl_win.groupby("bowler_id")["match_id"].nunique()),
        bowl_win.groupby("bowler_id")["match_id"].nunique(),
    )
    final_over_scores = data.loc[last_over_mask].groupby(["match_id", "innings", "striker_id"])["runs"].sum()
    high_last_over = final_over_scores[final_over_scores >= 17]
    high_last_over_count = high_last_over.groupby("striker_id").size()
    final_over_balls = data.loc[last_over_mask].groupby(["match_id", "innings", "striker_id"])["bat_balls"].sum()
    final_over_innings = final_over_balls.groupby("striker_id").size()
    _add_metric(values, samples, "bat_17_run_final_overs",
                _safe_divide(high_last_over_count, final_over_innings), final_over_innings)

    if "man_of_the_match" in data.columns:
        pom_ids = pd.to_numeric(data["man_of_the_match"], errors="coerce").dropna().unique()
        pom_matches = data.loc[data["man_of_the_match"].isin(pom_ids), ["match_id", "man_of_the_match"]].drop_duplicates()
        pom_counts = pom_matches.groupby("man_of_the_match")["match_id"].nunique()
        bat_matches_played = batter_match.groupby("striker_id")["match_id"].nunique()
        bowl_matches_played = bowler_match.groupby("bowler_id")["match_id"].nunique()
        _add_metric(values, samples, "bat_player_of_match",
                    _safe_divide(pom_counts, bat_matches_played), bat_matches_played)
        _add_metric(values, samples, "bowl_player_of_match",
                    _safe_divide(pom_counts, bowl_matches_played), bowl_matches_played)

    # Low-score defenses and sub-10 final-over defenses require a known match winner.
    match_innings_total = data.groupby(["match_id", "innings"])["ball_runs"].sum().rename("innings_total").reset_index()
    first_totals = match_innings_total[match_innings_total["innings"].eq(1)].set_index("match_id")["innings_total"]
    second = data[data["innings"].eq(2)].copy()
    second["first_innings_total"] = second["match_id"].map(first_totals)
    low_total_match_ids = second.loc[second["first_innings_total"] < 140, "match_id"].unique()
    low_total_won = bowler_match[
        bowler_match["match_id"].isin(low_total_match_ids)
        & bowler_match["win_team_id"].eq(bowler_match["bowling_team_id"])
    ]
    low_total_opportunities = bowler_match[bowler_match["match_id"].isin(low_total_match_ids)].groupby("bowler_id")["match_id"].nunique()
    low_total_wins = low_total_won.groupby("bowler_id")["match_id"].nunique()
    _add_metric(values, samples, "bowl_low_total_defenses",
                _safe_divide(low_total_wins, low_total_opportunities), low_total_opportunities)

    final_over_chases = data.loc[(data["innings"].eq(2)) & last_over_mask].copy()
    final_over_chases["runs_needed_start"] = final_over_chases.groupby("match_id")["runs_needed_before"].transform("first") if "runs_needed_before" in final_over_chases else np.nan
    defense_events = final_over_chases.groupby(["match_id", "bowler_id"]).agg(
        required=("runs_needed_start", "first"),
        bowling_team_id=("bowling_team_id", "first"),
        winner=("win_team_id", "first"),
    ).dropna(subset=["required"])
    under_10 = defense_events[defense_events["required"] < 10]
    defended = under_10[under_10["winner"].eq(under_10["bowling_team_id"])].groupby("bowler_id").size()
    under_10_opportunities = under_10.groupby("bowler_id").size()
    _add_metric(values, samples, "bowl_under_10_defenses",
                _safe_divide(defended, under_10_opportunities), under_10_opportunities)

    # Well-set dismissals and strong partnership breaks use recorded batter/wicket state.
    data = data.sort_values(["match_id", "innings", "overid", "over_ball_num"], kind="stable")
    batter_runs_before = data.groupby(["match_id", "innings", "striker_id"])["runs"].transform(lambda series: series.cumsum().shift(1, fill_value=0))
    well_set = (data["is_bowler_wicket"].astype(bool)) & (batter_runs_before >= 30)
    well_set_dismissals = data.loc[well_set].groupby("bowler_id").size()
    set_wicket_samples = data.loc[batter_runs_before >= 30].groupby("bowler_id").size()
    _add_metric(values, samples, "bowl_well_set_dismissals", well_set_dismissals, set_wicket_samples)
    partnership_breaks = data.loc[data["event_break_dangerous_stand"].astype(bool)].groupby("bowler_id").size()
    partnership_samples = data.groupby("bowler_id")["match_id"].count()
    _add_metric(values, samples, "bowl_partnership_breaks",
                _safe_divide(partnership_breaks, partnership_samples), partnership_samples)

    # Important matches are only those explicitly marked by supplied stage/knockout fields.
    if "is_any_knockout" in data.columns:
        important_mask = data["is_any_knockout"].fillna(0).astype(bool)
    else:
        stage = data["stage"].fillna("").astype(str).str.lower()
        important_mask = data["knock_out"].fillna(0).astype(bool) | stage.isin(["final", "semi final", "semi-final", "qualifier", "eliminator"])
    important_batting = batter_innings[
        batter_innings["match_id"].isin(data.loc[important_mask, "match_id"].unique())
    ]
    important_bowling = bowler_innings[
        bowler_innings["match_id"].isin(data.loc[important_mask, "match_id"].unique())
    ]
    _add_metric(values, samples, "bat_important_runs_per_innings",
                _safe_divide(important_batting.groupby("striker_id")["runs"].sum(), important_batting.groupby("striker_id")["innings"].count()),
                important_batting.groupby("striker_id")["innings"].count())
    _add_metric(values, samples, "bat_important_strike_rate",
                _safe_divide(important_batting.groupby("striker_id")["runs"].sum() * 100.0, important_batting.groupby("striker_id")["balls"].sum()),
                important_batting.groupby("striker_id")["balls"].sum())
    _add_metric(values, samples, "bowl_important_wickets_per_innings",
                _safe_divide(important_bowling.groupby("bowler_id")["wickets"].sum(), important_bowling.groupby("bowler_id")["innings"].count()),
                important_bowling.groupby("bowler_id")["innings"].count())
    _add_metric(values, samples, "bowl_important_economy",
                _safe_divide(important_bowling.groupby("bowler_id")["runs_conceded"].sum() * 6.0, important_bowling.groupby("bowler_id")["legal_balls"].sum()),
                important_bowling.groupby("bowler_id")["legal_balls"].sum())
    importance_samples = pd.concat([
        important_batting.groupby("striker_id")["innings"].count(),
        important_bowling.groupby("bowler_id")["innings"].count(),
    ]).groupby(level=0).sum()
    _add_metric(values, samples, "important_match_contributions", importance_samples, importance_samples)

    # Last five/ten appearance samples and season-to-season repeatability.
    bat_recent = batter_match.sort_values(["match_date", "match_id"]).groupby("striker_id", sort=False)
    for n in (5, 10):
        recent = bat_recent.tail(n).groupby("striker_id").agg(runs=("runs", "sum"), matches=("match_id", "nunique"))
        _add_metric(values, samples, f"bat_last{n}_runs_per_innings", _safe_divide(recent["runs"], recent["matches"]), recent["matches"])
    bowl_recent = bowler_match.sort_values(["match_date", "match_id"]).groupby("bowler_id", sort=False)
    for n in (5, 10):
        recent = bowl_recent.tail(n).groupby("bowler_id").agg(wickets=("wickets", "sum"), matches=("match_id", "nunique"))
        _add_metric(values, samples, f"bowl_last{n}_wickets_per_match", _safe_divide(recent["wickets"], recent["matches"]), recent["matches"])

    bat_by_season = batter_match.groupby(["striker_id", "season"]).agg(runs=("runs", "sum"), matches=("match_id", "nunique")).reset_index()
    bat_by_season["runs_per_match"] = _safe_divide(bat_by_season["runs"], bat_by_season["matches"])
    bat_season_cv = bat_by_season.groupby("striker_id")["runs_per_match"].agg(lambda series: series.std(ddof=0) / max(series.mean(), 1e-9) if len(series) >= 2 else np.nan)
    _add_metric(values, samples, "bat_season_consistency", 1.0 / (1.0 + bat_season_cv), bat_by_season.groupby("striker_id")["season"].nunique())
    bowl_by_season = bowler_match.groupby(["bowler_id", "season"]).agg(wickets=("wickets", "sum"), matches=("match_id", "nunique")).reset_index()
    bowl_by_season["wickets_per_match"] = _safe_divide(bowl_by_season["wickets"], bowl_by_season["matches"])
    bowl_season_cv = bowl_by_season.groupby("bowler_id")["wickets_per_match"].agg(lambda series: series.std(ddof=0) / max(series.mean(), 1e-9) if len(series) >= 2 else np.nan)
    _add_metric(values, samples, "bowl_season_consistency", 1.0 / (1.0 + bowl_season_cv), bowl_by_season.groupby("bowler_id")["season"].nunique())

    # Home/away split: supplied team IDs establish a player's home-team status per match.
    bat_home = batter_match[batter_match["batting_team_id"].eq(batter_match["home_team_id"])].groupby("striker_id")["runs"].mean()
    bat_away = batter_match[~batter_match["batting_team_id"].eq(batter_match["home_team_id"])].groupby("striker_id")["runs"].mean()
    _add_metric(values, samples, "bat_home_away_balance", _balanced_home_away(bat_home, bat_away), batter_match.groupby("striker_id")["match_id"].nunique())
    bowl_home = bowler_match[bowler_match["bowling_team_id"].eq(bowler_match["home_team_id"])].groupby("bowler_id")["wickets"].mean()
    bowl_away = bowler_match[~bowler_match["bowling_team_id"].eq(bowler_match["home_team_id"])].groupby("bowler_id")["wickets"].mean()
    _add_metric(values, samples, "bowl_home_away_balance", _balanced_home_away(bowl_home, bowl_away), bowler_match.groupby("bowler_id")["match_id"].nunique())

    # "Strong opposition" means the four teams with the highest observed win rates
    # in the supplied IPL sample, a dataset-derived estimate rather than a source label.
    if "win_team_id" in data.columns:
        team_match = data[["match_id", "win_team_id", "batting_team_id", "bowling_team_id"]].drop_duplicates("match_id")
        all_teams = pd.concat([team_match["batting_team_id"], team_match["bowling_team_id"]]).dropna().unique()
        wins = team_match["win_team_id"].value_counts()
        games = pd.concat([team_match["batting_team_id"], team_match["bowling_team_id"]]).value_counts()
        team_strength = (wins / games).sort_values(ascending=False)
        strong_teams = set(team_strength.head(4).index)
        strong_bat_match = batter_match[batter_match["bowling_team_id"].isin(strong_teams)]
        _add_metric(values, samples, "bat_strong_opposition_runs_per_match",
                    _safe_divide(strong_bat_match.groupby("striker_id")["runs"].sum(), strong_bat_match.groupby("striker_id")["match_id"].nunique()),
                    strong_bat_match.groupby("striker_id")["match_id"].nunique())
        strong_bowl_match = bowler_match[bowler_match["batting_team_id"].isin(strong_teams)]
        _add_metric(values, samples, "bowl_strong_opposition_wickets_per_match",
                    _safe_divide(strong_bowl_match.groupby("bowler_id")["wickets"].sum(), strong_bowl_match.groupby("bowler_id")["match_id"].nunique()),
                    strong_bowl_match.groupby("bowler_id")["match_id"].nunique())

    position_counts = batting.groupby("striker_id")["batting_order"].nunique()
    _add_metric(values, samples, "bat_positions_played", position_counts, batter_match.groupby("striker_id")["match_id"].nunique())
    bowling["phase"] = pd.cut(pd.to_numeric(bowling["overid"], errors="coerce"), [0, 6, 16, np.inf], labels=["powerplay", "middle", "death"], include_lowest=True)
    phases_played = bowling.groupby("bowler_id")["phase"].nunique()
    _add_metric(values, samples, "bowl_phase_roles_played", phases_played, bowler_match.groupby("bowler_id")["match_id"].nunique())

    # Phase, bowling type, handedness, batting failures, and expensive spells.
    for name, mask in (
        ("powerplay", pd.to_numeric(data["overid"], errors="coerce") <= 6),
        ("middle_overs", pd.to_numeric(data["overid"], errors="coerce").between(7, 16)),
        ("death_overs", pd.to_numeric(data["overid"], errors="coerce") >= 17),
    ):
        add_delivery_rate(f"bat_{name}_sr", data, "striker_id", mask, "runs", "bat_balls", 100.0)
        add_delivery_rate(f"bowl_{name}_economy", data, "bowler_id", mask, "bowler_runs", "is_legal", 6.0)
    pace = data["pace_or_spin"].fillna("").astype(str).str.lower()
    add_delivery_rate("bat_vs_pace_sr", data, "striker_id", pace.eq("pace"), "runs", "bat_balls", 100.0)
    add_delivery_rate("bat_vs_spin_sr", data, "striker_id", pace.eq("spin"), "runs", "bat_balls", 100.0)
    hand = data["batting_type"].fillna("").astype(str).str.upper()
    add_delivery_rate("bowl_vs_left_handed_economy", data, "bowler_id", hand.eq("LHB"), "bowler_runs", "is_legal", 6.0)
    add_delivery_rate("bowl_vs_right_handed_economy", data, "bowler_id", hand.eq("RHB"), "bowler_runs", "is_legal", 6.0)

    bat_failure = batter_innings[batter_innings["balls"] > 0].groupby("striker_id").agg(
        failures=("runs", lambda series: (series < 10).sum()), innings=("runs", "count")
    )
    _add_metric(values, samples, "bat_failure_rate", _safe_divide(bat_failure["failures"] * 100.0, bat_failure["innings"]), bat_failure["innings"])
    bowler_spells = bowling.groupby(["match_id", "innings", "bowler_id"]).agg(
        runs=("bowler_runs", "sum"), balls=("is_legal", "sum")
    )
    bowler_spells["economy"] = _safe_divide(bowler_spells["runs"] * 6.0, bowler_spells["balls"])
    expensive = bowler_spells.groupby("bowler_id").agg(
        expensive=("economy", lambda series: (series > 12.0).sum()), spells=("economy", "count")
    )
    _add_metric(values, samples, "bowl_expensive_spell_rate", _safe_divide(expensive["expensive"] * 100.0, expensive["spells"]), expensive["spells"])

    # Compile raw metric values and samples. Missing metrics remain null.
    metric_values = pd.DataFrame(index=player_ids)
    metric_samples = pd.DataFrame(index=player_ids)
    for key, series in values.items():
        metric_values[key] = series.reindex(player_ids)
        metric_samples[key] = samples[key].reindex(player_ids)

    # Role-aware, outlier-controlled scaling. A role's metric is never compared
    # to the other role's scale. Small event samples are flagged and not scored.
    score_values: dict[str, pd.Series] = {}
    normalization_directions = {
        key: direction
        for specs in CATEGORY_METRICS.values()
        for key, _, direction in specs
    }
    normalization_directions.update({"bat_strike_rate": True, "bowl_economy": False})
    for key, direction in normalization_directions.items():
        if key not in metric_values:
            continue
        if key == "bat_average":
            sample_floor = 3
            prior_sample = 15
        elif key == "bowl_average":
            sample_floor = 2
            prior_sample = 10
        elif key == "bat_strike_rate":
            sample_floor = 30
            prior_sample = 100
        elif key == "bowl_economy":
            sample_floor = 12
            prior_sample = 60
        elif key.endswith("season_consistency"):
            sample_floor = 2
            prior_sample = 3
        elif key.endswith(("strike_rate", "economy", "dot_pct")):
            sample_floor = 5
            prior_sample = 30
        elif key.startswith(("bat_high_rrr", "bat_low_rrr", "bat_low_wickets", "bat_close_finish", "bat_last_over", "bat_recovery",
                             "bowl_high_rrr", "bowl_low_rrr", "bowl_low_wickets", "bowl_close_finish", "bowl_last_over")):
            sample_floor = 5
            prior_sample = 30
        elif key.endswith(("last5_runs_per_innings", "last5_wickets_per_match")):
            sample_floor = 3
            prior_sample = 5
        elif key.endswith(("last10_runs_per_innings", "last10_wickets_per_match")):
            sample_floor = 5
            prior_sample = 10
        else:
            sample_floor = 3
            prior_sample = 5
        eligible = metric_values[key].where(metric_samples[key] >= sample_floor)
        scaled = _clip_score(eligible, direction)
        shrinkage = metric_samples[key].div(metric_samples[key] + prior_sample).clip(0, 1)
        score_values[key] = (50.0 + (scaled - 50.0) * shrinkage).where(eligible.notna())
    normalized = pd.DataFrame(score_values, index=player_ids)

    # Match-level sample sizes and role summaries.
    bat_match_count = batter_match.groupby("striker_id")["match_id"].nunique()
    bowl_match_count = bowler_match.groupby("bowler_id")["match_id"].nunique()
    total_matches = pd.concat([bat_match_count, bowl_match_count]).groupby(level=0).max().reindex(player_ids).fillna(0).astype(int)

    categories: dict[str, dict[str, dict[str, pd.Series]]] = {}
    role_path_category_blends: dict[str, dict[str, dict[str, float]]] = {
        "batting": {},
        "bowling": {},
    }
    for category, specs in CATEGORY_METRICS.items():
        category_scores: dict[str, pd.Series] = {}
        category_coverages: dict[str, pd.Series] = {}
        component_roles = {role for _, role, _ in specs}
        component_scores: dict[str, pd.Series] = {}
        for component_role in component_roles:
            keys = [key for key, metric_role, _ in specs if metric_role == component_role]
            existing = [key for key in keys if key in normalized.columns]
            component_scores[component_role] = (
                _equal_metric_mean(normalized, existing)
                if existing else pd.Series(np.nan, index=player_ids, dtype=float)
            )

        for player_role, match_counts in (
            ("batting", bat_match_count),
            ("bowling", bowl_match_count),
        ):
            blend = _role_path_category_blend_percent(specs, player_role)
            role_path_category_blends[player_role][category] = blend
            if not blend:
                continue
            own_keys = [key for key, metric_role, _ in specs if metric_role == player_role]
            shared_keys = [key for key, metric_role, _ in specs if metric_role == "shared"]
            fielding_keys = [key for key, metric_role, _ in specs if metric_role == "fielding"]
            expected_keys = own_keys + shared_keys + fielding_keys
            usable_keys = [key for key in expected_keys if key in normalized.columns]
            coverage = (
                normalized[usable_keys].notna().sum(axis=1).div(max(1, len(expected_keys)))
                if usable_keys else pd.Series(0.0, index=player_ids)
            )
            has_component = pd.Series(False, index=player_ids)
            blended_score = pd.Series(0.0, index=player_ids)
            for component_role, role_weight in blend.items():
                score = component_scores[component_role]
                has_component |= score.notna()
                blended_score += score.fillna(0.0).mul(role_weight / 100.0)
            role_active = match_counts.reindex(player_ids).fillna(0).gt(0)
            category_scores[player_role] = blended_score.mul(coverage).where(role_active & has_component)
            category_coverages[player_role] = coverage.where(role_active, 0.0)

        categories[category] = {
            "score": category_scores,
            "coverage": category_coverages,
        }

    master_results: dict[str, dict[str, pd.Series]] = {}
    master_role_blends: dict[str, dict[str, float]] = {}
    for master_key, spec in MASTER_METRICS.items():
        role_blends = _category_role_blend_percent(spec)
        master_role_blends[master_key] = role_blends
        role_scores: dict[str, pd.Series] = {}
        for role in role_blends:
            role_keys = [key for key, metric_role, _ in spec if metric_role == role]
            existing = [key for key in role_keys if key in normalized.columns]
            if existing:
                role_scores[role] = _equal_metric_mean(normalized, existing)
        combined_score = pd.DataFrame(role_scores, index=player_ids).mean(axis=1, skipna=True)
        available_keys = [
            key for key, role, _ in spec
            if role != "fielding" and key in normalized.columns
        ]
        coverage = normalized[available_keys].notna().sum(axis=1).div(max(1, len(spec)))
        master_results[master_key] = {
            "score": combined_score.mul(coverage),
            "coverage": coverage,
        }

    bat_ball_count = batter_totals["balls"].reindex(player_ids).fillna(0).astype(int)
    bowl_ball_count = bowler_totals["legal_balls"].reindex(player_ids).fillna(0).astype(int)

    player_records: list[dict[str, Any]] = []
    for player_id in player_ids:
        role_path_scores: dict[str, dict[str, Any]] = {}
        role_metric_keys = {"batting": "bat_strike_rate", "bowling": "bowl_economy"}
        role_match_counts = {"batting": bat_match_count, "bowling": bowl_match_count}
        for role, rate_key in role_metric_keys.items():
            path_categories: dict[str, float | None] = {}
            path_coverage: dict[str, float] = {}
            for category, result in categories.items():
                score = result["score"][role].get(player_id, np.nan)
                coverage = result["coverage"][role].get(player_id, 0.0)
                path_categories[category] = float(score) if pd.notna(score) else None
                path_coverage[category] = float(coverage)
            rate_score = normalized.at[player_id, rate_key] if rate_key in normalized else np.nan
            category_component = sum(
                path_categories[category] * weight
                for category, weight in CATEGORY_WEIGHTS.items()
                if path_categories[category] is not None
            )
            has_category_score = any(score is not None for score in path_categories.values())
            has_rate_score = pd.notna(rate_score)
            path_has_score = has_category_score or has_rate_score
            active = role_match_counts[role].get(player_id, 0) > 0
            path_final = (
                category_component + (float(rate_score) * ROLE_RATE_COMPONENT_WEIGHT[role] if has_rate_score else 0.0)
                if active and path_has_score else None
            )
            category_coverage_percent = {
                category: round(coverage * 100.0, 1)
                for category, coverage in path_coverage.items()
            }
            coverage_total = sum(
                path_coverage[category] * weight * 100.0
                for category, weight in CATEGORY_WEIGHTS.items()
            ) + (ROLE_RATE_COMPONENT_WEIGHT[role] * 100.0 if has_rate_score else 0.0)
            role_path_scores[role] = {
                "final_score": float(np.clip(path_final, 0, 100)) if path_final is not None else None,
                "category_scores": path_categories,
                "category_coverage": category_coverage_percent,
                "rate_component_score": float(rate_score) if has_rate_score else None,
                "rate_component_key": rate_key,
                "rate_component_weight_percent": ROLE_RATE_COMPONENT_WEIGHT[role] * 100.0,
                "data_coverage": coverage_total if active else 0.0,
            }

        batting_path = role_path_scores["batting"]
        bowling_path = role_path_scores["bowling"]
        has_batting_path = batting_path["final_score"] is not None
        has_bowling_path = bowling_path["final_score"] is not None
        is_all_rounder = has_batting_path and has_bowling_path
        if is_all_rounder:
            all_rounder_categories = {
                category: (
                    (batting_path["category_scores"][category] or 0.0) * ALL_ROUNDER_ROLE_BLEND
                    + (bowling_path["category_scores"][category] or 0.0) * ALL_ROUNDER_ROLE_BLEND
                )
                if batting_path["category_scores"][category] is not None
                or bowling_path["category_scores"][category] is not None else None
                for category in CATEGORY_WEIGHTS
            }
            all_rounder_coverage = {
                category: round((
                    batting_path["category_coverage"][category]
                    + bowling_path["category_coverage"][category]
                ) * ALL_ROUNDER_ROLE_BLEND, 1)
                for category in CATEGORY_WEIGHTS
            }
            final_score_raw = (
                batting_path["final_score"] * ALL_ROUNDER_ROLE_BLEND
                + bowling_path["final_score"] * ALL_ROUNDER_ROLE_BLEND
            )
            overall_coverage_raw = (
                batting_path["data_coverage"] * ALL_ROUNDER_ROLE_BLEND
                + bowling_path["data_coverage"] * ALL_ROUNDER_ROLE_BLEND
            )
            overall_role = "all_rounder"
            row_categories = all_rounder_categories
            category_coverage = all_rounder_coverage
            role_path_scores["all_rounder"] = {
                "final_score": final_score_raw,
                "category_scores": all_rounder_categories,
                "category_coverage": all_rounder_coverage,
                "rate_component_score": None,
                "rate_component_key": None,
                "rate_component_weight_percent": 10.0,
                "data_coverage": overall_coverage_raw,
                "role_blend_percent": {"batting": 50.0, "bowling": 50.0},
            }
        elif has_batting_path:
            overall_role = "batter"
            row_categories = batting_path["category_scores"]
            category_coverage = batting_path["category_coverage"]
            final_score_raw = batting_path["final_score"]
            overall_coverage_raw = batting_path["data_coverage"]
        elif has_bowling_path:
            overall_role = "bowler"
            row_categories = bowling_path["category_scores"]
            category_coverage = bowling_path["category_coverage"]
            final_score_raw = bowling_path["final_score"]
            overall_coverage_raw = bowling_path["data_coverage"]
        else:
            continue
        final_score = round(float(np.clip(final_score_raw, 0, 100)), 2)
        row_categories = {
            key: round(float(value), 2) if value is not None else None
            for key, value in row_categories.items()
        }
        overall_coverage = round(float(overall_coverage_raw), 1)

        master_scores = {
            key: round(float(result["score"].get(player_id)), 2)
            if pd.notna(result["score"].get(player_id)) else None
            for key, result in master_results.items()
        }
        master_coverage = {
            key: round(float(result["coverage"].get(player_id, 0.0)) * 100.0, 1)
            for key, result in master_results.items()
        }

        metrics = {}
        for key in sorted(set(metric_values.columns).union(UNAVAILABLE_METRICS)):
            if key in UNAVAILABLE_METRICS:
                metrics[key] = {
                    "value": None, "sample_size": None, "available": False,
                    "source": "unavailable", "reason": UNAVAILABLE_METRICS[key],
                }
                continue
            value = metric_values.at[player_id, key] if key in metric_values else np.nan
            sample = metric_samples.at[player_id, key] if key in metric_samples else np.nan
            if pd.isna(value) or pd.isna(sample) or sample <= 0:
                metrics[key] = {"value": None, "sample_size": 0 if pd.notna(sample) else None, "available": False, "source": "unavailable"}
            else:
                estimate_keys = {
                    "bat_home_away_balance", "bowl_home_away_balance",
                    "bat_season_consistency", "bowl_season_consistency",
                    "bat_strong_opposition_runs_per_match",
                    "bowl_strong_opposition_wickets_per_match",
                }
                metrics[key] = {
                    "value": round(float(value), 3),
                    "sample_size": int(sample) if float(sample).is_integer() else round(float(sample), 2),
                    "available": True,
                    "source": "derived_estimate" if key in estimate_keys else "derived_statistic",
                }

        limited_sample = int(total_matches.at[player_id]) < minimum_matches
        serialized_role_scores = {}
        for role, path in role_path_scores.items():
            serialized_role_scores[role] = {
                **path,
                "final_score": round(float(path["final_score"]), 2) if path["final_score"] is not None else None,
                "category_scores": {
                    key: round(float(value), 2) if value is not None else None
                    for key, value in path["category_scores"].items()
                },
                "data_coverage": round(float(path["data_coverage"]), 1),
            }
        player_records.append({
            "player_id": int(player_id) if isinstance(player_id, (int, np.integer)) else str(player_id),
            "name": player_names[player_id],
            "rank": 0,
            "final_score": final_score,
            "overall_role": overall_role,
            "role_scores": serialized_role_scores,
            "category_scores": row_categories,
            "category_coverage": category_coverage,
            "master_scores": master_scores,
            "master_coverage": master_coverage,
            "data_coverage": overall_coverage,
            "incomplete_data": overall_coverage < 100.0,
            "limited_sample": limited_sample,
            "minimum_sample_matches": minimum_matches,
            "sample": {
                "matches": int(total_matches.at[player_id]),
                "batting_matches": int(bat_match_count.get(player_id, 0)),
                "bowling_matches": int(bowl_match_count.get(player_id, 0)),
                "balls_faced": int(bat_ball_count.at[player_id]),
                "legal_balls_bowled": int(bowl_ball_count.at[player_id]),
            },
            "batting": {
                "runs": int(batter_totals.at[player_id, "runs"]) if player_id in batter_totals.index else None,
                "average": round(float(batter_totals.at[player_id, "avg"]), 2) if player_id in batter_totals.index and pd.notna(batter_totals.at[player_id, "avg"]) else None,
                "strike_rate": round(float(batter_totals.at[player_id, "sr"]), 2) if player_id in batter_totals.index and pd.notna(batter_totals.at[player_id, "sr"]) else None,
                "boundary_run_percent": round(float(batter_totals.at[player_id, "boundary_pct"]), 2) if player_id in batter_totals.index and pd.notna(batter_totals.at[player_id, "boundary_pct"]) else None,
                "fifties": int(bat_half_centuries.get(player_id, 0)),
                "hundreds": int(bat_centuries.get(player_id, 0)),
                "fours": int(batter_totals.at[player_id, "fours"]) if player_id in batter_totals.index else None,
                "sixes": int(batter_totals.at[player_id, "sixes"]) if player_id in batter_totals.index else None,
                "innings": int(batter_totals.at[player_id, "innings"]) if player_id in batter_totals.index else 0,
            },
            "bowling": {
                "wickets": int(bowler_totals.at[player_id, "wickets"]) if player_id in bowler_totals.index else None,
                "average": round(float(bowler_totals.at[player_id, "average"]), 2) if player_id in bowler_totals.index and pd.notna(bowler_totals.at[player_id, "average"]) else None,
                "economy": round(float(bowler_totals.at[player_id, "economy"]), 2) if player_id in bowler_totals.index and pd.notna(bowler_totals.at[player_id, "economy"]) else None,
                "strike_rate": round(float(bowler_totals.at[player_id, "sr"]), 2) if player_id in bowler_totals.index and pd.notna(bowler_totals.at[player_id, "sr"]) else None,
                "wickets_per_innings": round(float(bowler_totals.at[player_id, "wickets"] / bowler_totals.at[player_id, "innings"]), 3) if player_id in bowler_totals.index and bowler_totals.at[player_id, "innings"] > 0 else None,
                "dot_ball_percent": round(float(bowler_totals.at[player_id, "dot_pct"]), 2) if player_id in bowler_totals.index and pd.notna(bowler_totals.at[player_id, "dot_pct"]) else None,
                "legal_balls": int(bowler_totals.at[player_id, "legal_balls"]) if player_id in bowler_totals.index else None,
                "innings": int(bowler_totals.at[player_id, "innings"]) if player_id in bowler_totals.index else 0,
            },
            "metrics": metrics,
        })

    player_records.sort(key=lambda player: (-player["final_score"], player["name"].casefold()))
    for rank, record in enumerate(player_records, 1):
        record["rank"] = rank
    metric_leaders = []
    for metric_key in ("bat_strike_rate", "bowl_economy"):
        if metric_key not in normalized:
            continue
        metric_scores = normalized[metric_key].dropna()
        if metric_scores.empty:
            continue
        player_id = metric_scores.idxmax()
        sample_size = float(metric_samples.at[player_id, metric_key])
        metric_leaders.append({
            "key": metric_key,
            "player_id": int(player_id) if isinstance(player_id, (int, np.integer)) else str(player_id),
            "name": player_names[player_id],
            "value": round(float(metric_values.at[player_id, metric_key]), 2),
            "normalized_score": round(float(metric_scores.at[player_id]), 2),
            "sample_size": int(sample_size) if sample_size.is_integer() else round(sample_size, 2),
        })

    observed_categories = {}
    category_role_blends: dict[str, dict[str, float]] = {}
    category_parameter_weights: dict[str, dict[str, dict[str, float]]] = {}
    role_path_category_blends["all_rounder"] = {}
    for category, specs in CATEGORY_METRICS.items():
        available = [key for key, role, _ in specs if key in metric_values and metric_values[key].notna().any()]
        role_blend = _category_role_blend_percent(specs)
        role_path_category_blends["all_rounder"][category] = role_blend
        parameter_weights = {
            role: _role_parameter_weights_percent(
                [key for key, metric_role, _ in specs if metric_role == role], role
            )
            for role in role_blend
        }
        category_role_blends[category] = role_blend
        category_parameter_weights[category] = parameter_weights
        observed_categories[category] = {
            "name": CATEGORY_NAMES[category],
            "weight": CATEGORY_WEIGHTS[category],
            "weight_percent": CATEGORY_WEIGHTS[category] * 100,
            "role_blend_percent": role_blend,
            "role_path_blends_percent": {
                "batter": role_path_category_blends["batting"][category],
                "bowler": role_path_category_blends["bowling"][category],
                "all_rounder": role_blend,
            },
            "parameter_weights_percent": parameter_weights,
            "supported_metrics": available,
            "unavailable_metrics": [
                {"key": key, "reason": UNAVAILABLE_METRICS[key]}
                for key, _, _ in specs if key in UNAVAILABLE_METRICS
            ],
        }
    master_parameter_weights = {
        master_key: {
            role: _role_parameter_weights_percent(
                [key for key, metric_role, _ in spec if metric_role == role], role
            )
            for role in master_role_blends[master_key]
        }
        for master_key, spec in MASTER_METRICS.items()
    }
    total_deliveries = int(len(data))
    metadata = {
        "system_name": "IPL Six-Category Player Ranking",
        "methodology_version": 3,
        "competition_scope": "Indian Premier League only",
        "seasons": sorted(data["season"].dropna().astype(str).unique().tolist()),
        "total_deliveries": total_deliveries,
        "total_matches": int(data["match_id"].nunique()),
        "players_ranked": len(player_records),
        "minimum_sample_matches": minimum_matches,
        "role_leaderboard_minimums": ROLE_LEADERBOARD_MINIMUMS,
        "weights_total_percent": round(sum(CATEGORY_WEIGHTS.values()) * 100, 8),
        "overall_framework_total_percent": 100.0,
        "role_framework_totals_percent": ROLE_FRAMEWORK_TOTALS_PERCENT,
        "role_final_score_components_percent": {
            "batter": {"six_categories": 90.0, "overall_strike_rate": 10.0, "total": 100.0},
            "bowler": {"six_categories": 90.0, "overall_economy": 10.0, "total": 100.0},
            "all_rounder": {"batter_final_score": 50.0, "bowler_final_score": 50.0, "total": 100.0},
        },
        "role_framework_policy": "Batter and bowler scores each combine the six official categories (90% total) with one standalone overall rate component (10%). The dataset has no official player-role label, so the all-rounder pool is operationally inferred when both role scores are available and uses the equally weighted mean of those two independently normalized scores.",
        "metric_weight_policy": "The standalone batting strike-rate and bowling economy components are excluded from all six categories to prevent double counting. Metrics within each role/category group share that group's role blend equally. Contextual and phase-specific rate metrics remain separate category metrics. Missing values are not imputed or reweighted; category coverage is reduced and unsupported fielding metrics remain coverage-only.",
        "category_role_blends_percent": category_role_blends,
        "role_path_category_role_blends_percent": role_path_category_blends,
        "category_parameter_weights_percent": category_parameter_weights,
        "master_score_role_blends_percent": master_role_blends,
        "master_score_parameter_weights_percent": master_parameter_weights,
        "normalization": "Role-aware 5th-95th percentile clipping; lower-is-better metrics inverted; missing metrics omitted without imputation.",
        "missing_category_policy": "Official category and standalone component weights remain fixed. Unavailable category/component values contribute no points and weights are not reassigned; coverage and incomplete-data flags are shown.",
        "master_scores_are_supplementary": True,
        "master_scores_added_to_final": False,
        "category_definitions": observed_categories,
        "unavailable_metrics": UNAVAILABLE_METRICS,
        "score_formula": "Batter = sum(category score × category weight) + normalized overall batting strike rate × 0.10. Bowler = sum(category score × category weight) + normalized overall bowling economy × 0.10. All-rounder = 0.50 × batter score + 0.50 × bowler score. Missing contributions are not reweighted.",
    }
    def role_leaderboard(role: str, stat: str, minimum: int) -> list[dict[str, Any]]:
        candidates = [
            {
                "player_id": player["player_id"],
                "name": player["name"],
                "final_score": player["role_scores"][role]["final_score"],
                "overall_final_score": player["final_score"],
                "category_scores": player["role_scores"][role]["category_scores"],
                "category_coverage": player["role_scores"][role]["category_coverage"],
                "master_scores": player["master_scores"],
                "data_coverage": player["role_scores"][role]["data_coverage"],
                "sample": player["sample"],
                role: player[role],
            }
            for player in player_records
            if player[role][stat] is not None
            and player[role][stat] >= minimum
            and player["role_scores"][role]["final_score"] is not None
        ]
        candidates.sort(key=lambda player: (-player["final_score"], player["name"].casefold()))
        for rank, player in enumerate(candidates, 1):
            player["rank"] = rank
        return candidates

    all_rounders = [
        player for player in player_records
        if player["role_scores"].get("all_rounder", {}).get("final_score") is not None
    ]
    all_rounders.sort(
        key=lambda player: (
            -player["role_scores"]["all_rounder"]["final_score"],
            player["name"].casefold(),
        )
    )
    all_rounder_leaderboard = [
        {
            "player_id": player["player_id"],
            "rank": rank,
            "name": player["name"],
            "final_score": player["role_scores"]["all_rounder"]["final_score"],
            "role_scores": player["role_scores"],
            "category_scores": player["role_scores"]["all_rounder"]["category_scores"],
            "category_coverage": player["role_scores"]["all_rounder"]["category_coverage"],
            "data_coverage": player["role_scores"]["all_rounder"]["data_coverage"],
            "sample": player["sample"],
            "batting": player["batting"],
            "bowling": player["bowling"],
        }
        for rank, player in enumerate(all_rounders, 1)
    ]
    top_batters = role_leaderboard(
        "batting", "runs", ROLE_LEADERBOARD_MINIMUMS["batting_runs"]
    )[:10]
    top_bowlers = role_leaderboard(
        "bowling", "wickets", ROLE_LEADERBOARD_MINIMUMS["bowling_wickets"]
    )[:10]
    return {
        "metadata": metadata,
        "weights": {key: value * 100 for key, value in CATEGORY_WEIGHTS.items()},
        "categories": observed_categories,
        "players": player_records,
        "top_metric_performers": metric_leaders,
        "top_batters": top_batters,
        "top_bowlers": top_bowlers,
        "all_rounders": all_rounder_leaderboard,
        "top_all_rounders": all_rounder_leaderboard[:10],
    }


def run_ipl_ranking_pipeline(
    data_path: str | Path | None = None,
    matches_path: str | Path | None = None,
    output_path: str | Path | None = None,
    minimum_matches: int = 5,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    balls, matches = _load_ipl_data(data_path, matches_path)
    rankings = calculate_ipl_player_rankings(balls, matches, minimum_matches)
    output_path = Path(output_path) if output_path else root / "dashboard_data.json"
    output_path.write_text(json.dumps(rankings, indent=2, allow_nan=False), encoding="utf-8")
    return rankings

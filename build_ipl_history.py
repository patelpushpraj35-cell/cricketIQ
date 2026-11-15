import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_pack" / "Data Pack"
OUTPUT = ROOT / "ipl_history_data.json"
WICKET_DISMISSALS = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def _records(frame):
    return json.loads(frame.to_json(orient="records"))


def build_ipl_history():
    competitions = pd.read_csv(DATA_DIR / "competitions.csv")
    ipl = competitions[
        competitions["comp_name"].astype(str).str.fullmatch(r"IPL \d{4}", case=False, na=False)
    ].copy()
    if ipl.empty:
        raise ValueError("No IPL season competitions were found in competitions.csv")

    ipl["season"] = pd.to_numeric(ipl["season"], errors="raise").astype(int)
    comp_seasons = dict(zip(ipl["comp_id"], ipl["season"]))
    matches = pd.read_csv(DATA_DIR / "matches.csv")
    matches = matches[matches["comp_id"].isin(comp_seasons)].copy()
    matches["season"] = matches["comp_id"].map(comp_seasons).astype(int)
    if matches.empty:
        raise ValueError("No IPL matches were found for the listed IPL competitions")

    teams = pd.read_csv(DATA_DIR / "teams.csv")
    teams = teams[teams["comp_id"].isin(comp_seasons)].copy()
    team_names = {}
    season_teams = {}
    for comp_id, season in comp_seasons.items():
        comp_teams = teams[teams["comp_id"] == comp_id]
        season_teams[season] = {
            int(row.team_id): str(row.team_name)
            for row in comp_teams.itertuples(index=False)
        }
        team_names.update(season_teams[season])

    player_map = pd.read_csv(DATA_DIR / "competition_team_player_mapping.csv")
    player_map = player_map[player_map["comp_id"].isin(comp_seasons)].copy()
    player_map["season"] = player_map["comp_id"].map(comp_seasons).astype(int)
    player_names = {}
    for row in player_map.itertuples(index=False):
        player_names[int(row.player_id)] = str(row.player_name)

    appearances = pd.read_csv(DATA_DIR / "playing_eleven.csv")
    appearances = appearances[
        appearances["match_id"].isin(matches["match_id"])
        & appearances["is_played"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    appearances = appearances.drop_duplicates(["match_id", "team_id", "player_id"])
    match_seasons = matches.set_index("match_id")["season"].to_dict()
    appearances["season"] = appearances["match_id"].map(match_seasons).astype(int)
    appearances["player_id"] = pd.to_numeric(appearances["player_id"], errors="coerce")
    appearances = appearances.dropna(subset=["player_id"])
    appearances["player_id"] = appearances["player_id"].astype(int)

    balls = pd.read_csv(DATA_DIR / "ball_by_ball.csv")
    balls = balls[balls["match_id"].isin(matches["match_id"])].copy()
    balls["season"] = balls["match_id"].map(match_seasons).astype(int)
    for column in ["runs", "extras1_runs", "extras_2_runs"]:
        balls[column] = pd.to_numeric(balls[column], errors="coerce").fillna(0)
    balls["is_four"] = balls["is_four"].fillna(False).astype(bool)
    balls["is_six"] = balls["is_six"].fillna(False).astype(bool)
    balls["is_wicket"] = balls["is_wicket"].fillna(False).astype(bool)
    balls["dismissal_type"] = balls["dismissal_type"].fillna("")
    extras = balls["extras_type"].fillna("")
    balls["bat_balls"] = (extras != "wide").astype(int)
    balls["legal_balls"] = (~extras.isin(["wide", "Overstep"])).astype(int)
    balls["bowler_runs"] = balls["runs"] + balls["extras1_runs"] + np.where(
        ~extras.isin(["byes", "legbyes"]), balls["extras_2_runs"], 0
    )
    balls["bowler_wickets"] = balls["dismissal_type"].isin(WICKET_DISMISSALS).astype(int)
    balls["batter_out"] = (
        balls["is_wicket"]
        & (balls["striker_id"] == balls["dismissed_player_id"])
        & (balls["dismissal_type"] != "Retired Hurt")
    ).astype(int)

    batting = balls.groupby(["season", "striker_id"], as_index=False).agg(
        runs=("runs", "sum"),
        balls_faced=("bat_balls", "sum"),
        dismissals=("batter_out", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    ).rename(columns={"striker_id": "player_id"})
    bowling = balls.groupby(["season", "bowler_id"], as_index=False).agg(
        runs_conceded=("bowler_runs", "sum"),
        balls_bowled=("legal_balls", "sum"),
        wickets=("bowler_wickets", "sum"),
    ).rename(columns={"bowler_id": "player_id"})
    for frame in [batting, bowling]:
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
        frame.dropna(subset=["player_id"], inplace=True)
        frame["player_id"] = frame["player_id"].astype(int)

    player_seasons = appearances.groupby(["season", "player_id"], as_index=False).agg(
        matches=("match_id", "nunique")
    )
    player_teams = appearances.groupby(["season", "player_id"])["team_id"].agg(
        lambda values: sorted({int(value) for value in values if pd.notna(value)})
    ).rename("team_ids").reset_index()
    season_stats = player_seasons.merge(player_teams, on=["season", "player_id"], how="left")
    season_stats = season_stats.merge(batting, on=["season", "player_id"], how="left")
    season_stats = season_stats.merge(bowling, on=["season", "player_id"], how="left")
    for column in ["runs", "balls_faced", "dismissals", "fours", "sixes", "runs_conceded", "balls_bowled", "wickets"]:
        season_stats[column] = season_stats[column].fillna(0)
    season_stats["average"] = np.where(
        season_stats["dismissals"] > 0, season_stats["runs"] / season_stats["dismissals"], np.nan
    )
    season_stats["strike_rate"] = np.where(
        season_stats["balls_faced"] > 0, season_stats["runs"] * 100 / season_stats["balls_faced"], np.nan
    )
    season_stats["bowling_average"] = np.where(
        season_stats["wickets"] > 0, season_stats["runs_conceded"] / season_stats["wickets"], np.nan
    )
    season_stats["economy"] = np.where(
        season_stats["balls_bowled"] > 0,
        season_stats["runs_conceded"] * 6 / season_stats["balls_bowled"],
        np.nan,
    )
    season_stats["bowling_strike_rate"] = np.where(
        season_stats["wickets"] > 0, season_stats["balls_bowled"] / season_stats["wickets"], np.nan
    )

    motm = matches[pd.to_numeric(matches["man_of_the_match"], errors="coerce").fillna(0) > 0].copy()
    motm["player_id"] = pd.to_numeric(motm["man_of_the_match"], errors="coerce").astype(int)
    motm["season"] = motm["season"].astype(int)
    award_counts = motm.groupby("player_id").size().to_dict()
    season_award_counts = motm.groupby(["season", "player_id"]).size().to_dict()

    winner_team_by_season = {}
    final_by_season = {}
    final_match_by_season = {}
    for season in sorted(comp_seasons.values()):
        season_matches = matches[matches["season"] == season]
        final_rows = season_matches[season_matches["stage"].astype(str).str.lower() == "final"]
        final = final_rows.iloc[-1] if not final_rows.empty else None
        final_info = None
        if final is not None:
            winner_id = int(final["win_team_id"]) if pd.notna(final["win_team_id"]) and final["win_team_id"] else None
            home_id = int(final["home_team_id"]) if pd.notna(final["home_team_id"]) else None
            away_id = int(final["away_team_id"]) if pd.notna(final["away_team_id"]) else None
            runner_id = next((team_id for team_id in [home_id, away_id] if team_id and team_id != winner_id), None)
            if winner_id:
                winner_team_by_season[season] = winner_id
            final_match_by_season[season] = {
                "matchId": int(final["match_id"]),
                "winnerId": winner_id,
            }
            final_info = {
                "matchNumber": int(final["match_number"]) if pd.notna(final["match_number"]) else None,
                "date": str(final["match_date"]) if pd.notna(final["match_date"]) else None,
                "winner": team_names.get(winner_id),
                "runnerUp": team_names.get(runner_id),
                "result": str(final["result"]) if pd.notna(final["result"]) else None,
                "venueId": int(final["venue_id"]) if pd.notna(final["venue_id"]) else None,
                "venueName": None,
            }
        final_by_season[season] = final_info

    player_data = {}
    for row in season_stats.itertuples(index=False):
        player_id = int(row.player_id)
        player = player_data.setdefault(player_id, {
            "id": player_id,
            "name": player_names.get(player_id, str(player_id)),
            "seasonStats": [],
            "teams": set(),
            "championshipSeasons": [],
            "finalAppearanceSeasons": [],
        })
        team_ids = row.team_ids or []
        team_names_for_player = sorted({team_names[team_id] for team_id in team_ids if team_id in team_names})
        player["teams"].update(team_names_for_player)
        player["seasonStats"].append({
            "season": int(row.season),
            "teams": team_names_for_player,
            "matches": int(row.matches),
            "runs": int(row.runs),
            "ballsFaced": int(row.balls_faced),
            "average": row.average,
            "dismissals": int(row.dismissals),
            "strikeRate": row.strike_rate,
            "fours": int(row.fours),
            "sixes": int(row.sixes),
            "wickets": int(row.wickets),
            "runsConceded": int(row.runs_conceded),
            "ballsBowled": int(row.balls_bowled),
            "bowlingAverage": row.bowling_average,
            "economy": row.economy,
            "bowlingStrikeRate": row.bowling_strike_rate,
            "playerOfMatchAwards": int(season_award_counts.get((int(row.season), player_id), 0)),
        })

    for row in appearances.itertuples(index=False):
        final = final_match_by_season.get(int(row.season))
        player = player_data.get(int(row.player_id))
        if player and final and int(row.match_id) == final["matchId"]:
            player["finalAppearanceSeasons"].append(int(row.season))
            if final["winnerId"] == int(row.team_id):
                player["championshipSeasons"].append(int(row.season))

    match_stats = defaultdict(dict)
    batting_match = balls.groupby(["match_id", "striker_id"], as_index=False).agg(
        runs=("runs", "sum"), balls=("bat_balls", "sum"), fours=("is_four", "sum"), sixes=("is_six", "sum")
    )
    bowling_match = balls.groupby(["match_id", "bowler_id"], as_index=False).agg(
        wickets=("bowler_wickets", "sum"), runs=("bowler_runs", "sum"), balls=("legal_balls", "sum")
    )
    for row in batting_match.itertuples(index=False):
        match_stats[int(row.match_id)].setdefault(int(row.striker_id), {})["batting"] = {
            "runs": int(row.runs), "balls": int(row.balls), "fours": int(row.fours), "sixes": int(row.sixes)
        }
    for row in bowling_match.itertuples(index=False):
        match_stats[int(row.match_id)].setdefault(int(row.bowler_id), {})["bowling"] = {
            "wickets": int(row.wickets), "runs": int(row.runs), "balls": int(row.balls)
        }

    appearances_by_match_player = appearances.drop_duplicates(["match_id", "player_id"]).set_index(
        ["match_id", "player_id"]
    )["team_id"].to_dict()
    award_history = []
    for award in motm.sort_values(["season", "match_date", "match_number"]).itertuples(index=False):
        player_id = int(award.player_id)
        match_id = int(award.match_id)
        home_id = int(award.home_team_id) if pd.notna(award.home_team_id) else None
        away_id = int(award.away_team_id) if pd.notna(award.away_team_id) else None
        player_team_id = appearances_by_match_player.get((match_id, player_id))
        opponent_id = away_id if player_team_id == home_id else home_id if player_team_id == away_id else None
        if player_id not in player_data:
            player_data[player_id] = {
                "id": player_id, "name": player_names.get(player_id, str(player_id)),
                "seasonStats": [], "teams": set(), "championshipSeasons": [],
                "finalAppearanceSeasons": [],
            }
        player_data[player_id]["teams"].update(
            name for name in [team_names.get(player_team_id)] if name
        )
        perf = match_stats.get(match_id, {}).get(player_id, {})
        award_history.append({
            "playerId": player_id,
            "playerName": player_names.get(player_id, str(player_id)),
            "team": team_names.get(player_team_id),
            "season": int(award.season),
            "matchNumber": int(award.match_number) if pd.notna(award.match_number) else None,
            "matchId": match_id,
            "opponent": team_names.get(opponent_id),
            "result": str(award.result) if pd.notna(award.result) else None,
            "performance": perf or None,
        })

    for player in player_data.values():
        player["teams"] = sorted(player["teams"])
        player["seasonStats"].sort(key=lambda row: row["season"])
        player["seasonsPlayed"] = len(player["seasonStats"])
        player["debutSeason"] = player["seasonStats"][0]["season"] if player["seasonStats"] else None
        player["matches"] = sum(row["matches"] for row in player["seasonStats"])
        player["runs"] = sum(row["runs"] for row in player["seasonStats"])
        player["wickets"] = sum(row["wickets"] for row in player["seasonStats"])
        player["ballsFaced"] = sum(row["ballsFaced"] for row in player["seasonStats"])
        player["dismissals"] = sum(row["dismissals"] for row in player["seasonStats"])
        player["runsConceded"] = sum(row["runsConceded"] for row in player["seasonStats"])
        player["ballsBowled"] = sum(row["ballsBowled"] for row in player["seasonStats"])
        player["fours"] = sum(row["fours"] for row in player["seasonStats"])
        player["sixes"] = sum(row["sixes"] for row in player["seasonStats"])
        player["playerOfMatchAwards"] = int(award_counts.get(player["id"], 0))
        player["orangeCapAwards"] = None
        player["purpleCapAwards"] = None
        player["championshipSeasons"].sort()
        player["finalAppearanceSeasons"].sort()
        player["seasonStats"] = [
            {key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
             for key, value in row.items()}
            for row in player["seasonStats"]
        ]

    players = sorted(player_data.values(), key=lambda row: row["name"].casefold())
    season_data = []
    for season in sorted(comp_seasons.values()):
        comp_id = next(comp_id for comp_id, comp_season in comp_seasons.items() if comp_season == season)
        season_matches = matches[matches["season"] == season]
        season_roster = teams[teams["comp_id"] == comp_id]
        stats = season_stats[season_stats["season"] == season]
        run_leaders = stats.sort_values(["runs", "strike_rate"], ascending=[False, False]).head(25)
        wicket_leaders = stats.sort_values(["wickets", "economy"], ascending=[False, True]).head(25)
        season_data.append({
            "season": season,
            "matchCount": int(season_matches["match_id"].nunique()),
            "teams": sorted(season_roster["team_name"].dropna().astype(str).unique().tolist()),
            "final": final_by_season[season],
            "officialAwards": {"orangeCapWinner": None, "purpleCapWinner": None},
            "orangeCapWinnerVerified": False,
            "purpleCapWinnerVerified": False,
            "runLeaders": [
                {"playerId": int(row.player_id), "playerName": player_names.get(int(row.player_id), str(row.player_id)),
                 "team": ", ".join(sorted({team_names[t] for t in (row.team_ids or []) if t in team_names})),
                 "runs": int(row.runs), "matches": int(row.matches),
                 "average": None if not np.isfinite(row.average) else round(float(row.average), 2),
                 "strikeRate": None if not np.isfinite(row.strike_rate) else round(float(row.strike_rate), 2),
                 "fours": int(row.fours), "sixes": int(row.sixes)}
                for row in run_leaders.itertuples(index=False)
            ],
            "wicketLeaders": [
                {"playerId": int(row.player_id), "playerName": player_names.get(int(row.player_id), str(row.player_id)),
                 "team": ", ".join(sorted({team_names[t] for t in (row.team_ids or []) if t in team_names})),
                 "wickets": int(row.wickets), "matches": int(row.matches),
                 "average": None if not np.isfinite(row.bowling_average) else round(float(row.bowling_average), 2),
                 "economy": None if not np.isfinite(row.economy) else round(float(row.economy), 2),
                 "strikeRate": None if not np.isfinite(row.bowling_strike_rate) else round(float(row.bowling_strike_rate), 2)}
                for row in wicket_leaders.itertuples(index=False)
            ],
            "playerOfMatchAwardsRecorded": sum(1 for row in award_history if row["season"] == season),
            "playerOfMatchMatchesMissing": int(season_matches["match_id"].nunique()) -
                sum(1 for row in award_history if row["season"] == season),
        })

    records = {
        "runs": sorted(players, key=lambda row: (-row["runs"], row["name"]))[:20],
        "wickets": sorted(players, key=lambda row: (-row["wickets"], row["name"]))[:20],
        "playerOfMatch": sorted(players, key=lambda row: (-row["playerOfMatchAwards"], row["name"]))[:20],
        "matches": sorted(players, key=lambda row: (-row["matches"], row["name"]))[:20],
        "seasons": sorted(players, key=lambda row: (-row["seasonsPlayed"], row["name"]))[:20],
        "championshipSeasons": sorted(
            players, key=lambda row: (-len(row["championshipSeasons"]), row["name"])
        )[:20],
        "finalAppearances": sorted(
            players, key=lambda row: (-len(row["finalAppearanceSeasons"]), row["name"])
        )[:20],
    }

    payload = {
        "metadata": {
            "source": "Provided IPL Data Pack CSVs",
            "seasons": sorted(comp_seasons.values()),
            "sourceFiles": [
                "competitions.csv", "matches.csv", "teams.csv",
                "competition_team_player_mapping.csv", "playing_eleven.csv", "ball_by_ball.csv",
            ],
            "scope": "IPL only; T20Is and other competitions are excluded.",
            "limits": [
                "The source pack covers IPL 2022–2026 only.",
                "Official Orange Cap and Purple Cap winner records are not included; season leaders are calculated from ball-by-ball totals and are not represented as award winners.",
                "Venue names are not included; final venue IDs are shown only when present.",
                "Player of the Match is missing for some source matches; recorded award counts are lower bounds.",
                "Player images and franchise logos are not included in the source pack.",
            ],
            "playerOfMatchRecorded": len(award_history),
            "playerOfMatchMatchesMissing": int(matches["match_id"].nunique()) - len(award_history),
        },
        "seasons": season_data,
        "players": players,
        "playerOfMatchAwards": award_history,
        "records": records,
        "officialCapRecords": {
            "orange": [],
            "purple": [],
            "status": "Unavailable: official award winner data is absent from the provided data pack.",
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(
        f"Wrote {OUTPUT.name}: {len(players)} players, {len(season_data)} seasons, "
        f"{len(award_history)} recorded Player of the Match awards."
    )


if __name__ == "__main__":
    build_ipl_history()

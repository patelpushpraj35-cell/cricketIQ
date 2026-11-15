"""Generate all public IPL ranking data from the shared six-category engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.ipl_player_ranking import (
    CATEGORY_NAMES,
    CATEGORY_WEIGHTS,
    ROLE_LEADERBOARD_MINIMUMS,
    run_ipl_ranking_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


def run_full_pipeline():
    rankings = run_ipl_ranking_pipeline()
    players = pd.json_normalize(rankings["players"], sep="_")
    players.to_csv(ROOT / "overall_ipl_player_leaderboard.csv", index=False)

    # Keep the existing batter/bowler exports available, but populate them only
    # from the same official leaderboard and its role-specific source statistics.
    batters = players[
        (players["batting_runs"] >= ROLE_LEADERBOARD_MINIMUMS["batting_runs"])
        & players["role_scores_batting_final_score"].notna()
    ].sort_values(["role_scores_batting_final_score", "name"], ascending=[False, True])
    bowlers = players[
        (players["bowling_wickets"] >= ROLE_LEADERBOARD_MINIMUMS["bowling_wickets"])
        & players["role_scores_bowling_final_score"].notna()
    ].sort_values(["role_scores_bowling_final_score", "name"], ascending=[False, True])
    all_rounders = players[
        players["role_scores_all_rounder_final_score"].notna()
    ].sort_values(["role_scores_all_rounder_final_score", "name"], ascending=[False, True])
    batters.to_csv(ROOT / "full_batting_ratings.csv", index=False)
    bowlers.to_csv(ROOT / "full_bowling_ratings.csv", index=False)
    all_rounders.to_csv(ROOT / "full_all_rounder_ratings.csv", index=False)
    for filename, frame in (
        ("top_10_batters_final.csv", batters.head(10)),
        ("top_10_batters_clutch.csv", batters.head(10)),
        ("top_10_batters_advanced.csv", batters.head(10)),
        ("top_batters_analysis.csv", batters),
        ("top_10_bowlers_final.csv", bowlers.head(10)),
        ("top_10_bowlers_clutch.csv", bowlers.head(10)),
        ("top_10_bowlers_advanced.csv", bowlers.head(10)),
        ("top_bowlers_analysis.csv", bowlers),
        ("top_10_all_rounders_final.csv", all_rounders.head(10)),
        ("top_all_rounders_analysis.csv", all_rounders),
    ):
        frame.to_csv(ROOT / filename, index=False)

    for category, weight in CATEGORY_WEIGHTS.items():
        rows = [
            {
                "rank": rank,
                "player_id": player["player_id"],
                "player_name": player["name"],
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "weight_percent": weight * 100,
                "category_score": player["category_scores"][category],
                "category_coverage_percent": player["category_coverage"][category],
                "final_score": player["final_score"],
                "data_coverage_percent": player["data_coverage"],
                "sample_matches": player["sample"]["matches"],
            }
            for rank, player in enumerate(
                sorted(
                    (p for p in rankings["players"] if p["category_scores"][category] is not None),
                    key=lambda p: (-p["category_scores"][category], p["name"].casefold()),
                ),
                1,
            )
        ]
        pd.DataFrame(rows).to_csv(ROOT / f"top_{category}_players.csv", index=False)

    with (ROOT / "dashboard_data.json").open("w", encoding="utf-8") as output:
        json.dump(rankings, output, indent=2, allow_nan=False)

    print(
        "Generated official IPL leaderboard:",
        f"{rankings['metadata']['players_ranked']} players,",
        f"{rankings['metadata']['total_matches']} matches,",
        f"{rankings['metadata']['total_deliveries']} deliveries.",
    )
    return rankings


if __name__ == "__main__":
    run_full_pipeline()

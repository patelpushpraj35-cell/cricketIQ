import unittest

import numpy as np
import pandas as pd

from engine.ipl_player_ranking import (
    CATEGORY_METRICS,
    CATEGORY_WEIGHTS,
    _clip_score,
    calculate_ipl_player_rankings,
)


def sample_data(match_count=8):
    balls = []
    matches = []
    for match_id in range(1, match_count + 1):
        home_team = 10
        winner = 10 if match_id % 2 else 20
        matches.append({
            "match_id": match_id,
            "home_team_id": home_team,
            "win_team_id": winner,
            "man_of_the_match": 101 if match_id == 1 else 0,
            "result": "Team Won by 5 Runs",
            "stage": "Final" if match_id == match_count else "League",
            "knock_out": int(match_id == match_count),
            "match_date": f"2022-04-{match_id:02d}",
            "season": 2022,
        })
        for innings in (1, 2):
            batting_team = 10 if innings == 1 else 20
            bowling_team = 20 if innings == 1 else 10
            for ball in range(1, 19):
                over = (ball - 1) // 6 + 1
                striker_id = 101 if batting_team == 10 else 202
                bowler_id = 202 if batting_team == 10 else 101
                runs = 2 if striker_id == 101 else (1 if ball % 2 else 0)
                balls.append({
                    "match_id": match_id,
                    "innings": innings,
                    "overid": over,
                    "over_ball_num": (ball - 1) % 6 + 1,
                    "striker_id": striker_id,
                    "striker_name": "Player One" if striker_id == 101 else "Player Two",
                    "bowler_id": bowler_id,
                    "bowler_name": "Player Two" if bowler_id == 202 else "Player One",
                    "batting_team_id": batting_team,
                    "bowling_team_id": bowling_team,
                    "runs": runs,
                    "ball_runs": runs,
                    "bowler_runs": runs,
                    "extras_type": None,
                    "is_legal": 1,
                    "is_four": int(runs == 4),
                    "is_six": 0,
                    "is_wicket": False,
                    "is_bowler_wicket": 0,
                    "striker_dismissed": 0,
                    "dismissed_player_id": None,
                    "is_dot_bat": int(runs == 0),
                    "is_dot_bowl": int(runs == 0),
                    "batting_order": 1,
                    "pace_or_spin": "Pace",
                    "batting_type": "RHB",
                    "rrr_before": 8 if innings == 2 else 0,
                    "wickets_in_hand_before": 10,
                    "cum_wickets_before": 0,
                    "is_any_knockout": int(match_id == match_count),
                    "knock_out": int(match_id == match_count),
                    "stage": "Final" if match_id == match_count else "League",
                    "comp_name": "IPL 2022",
                    "season": "2022",
                })
    return pd.DataFrame(balls), pd.DataFrame(matches)


class TestIplPlayerRanking(unittest.TestCase):
    def test_weights_scores_order_and_supplemental_masters(self):
        balls, matches = sample_data()
        result = calculate_ipl_player_rankings(balls, matches, minimum_matches=10)

        self.assertEqual(CATEGORY_WEIGHTS, {
            "traditional": 0.15,
            "pressure": 0.25,
            "impact": 0.17,
            "important_matches": 0.15,
            "consistency": 0.10,
            "game_intelligence": 0.08,
        })
        metric_categories = {}
        for category, specs in CATEGORY_METRICS.items():
            for key, _, _ in specs:
                metric_categories.setdefault(key, []).append(category)
        self.assertTrue(all(len(categories) == 1 for categories in metric_categories.values()))
        self.assertAlmostEqual(sum(CATEGORY_WEIGHTS.values()), 0.90)
        self.assertEqual(result["metadata"]["weights_total_percent"], 90.0)
        self.assertEqual(result["metadata"]["overall_framework_total_percent"], 100.0)
        metadata = result["metadata"]
        self.assertEqual(metadata["role_final_score_components_percent"], {
            "batter": {"six_categories": 90.0, "overall_strike_rate": 10.0, "total": 100.0},
            "bowler": {"six_categories": 90.0, "overall_economy": 10.0, "total": 100.0},
            "all_rounder": {"batter_final_score": 50.0, "bowler_final_score": 50.0, "total": 100.0},
        })
        self.assertEqual(metadata["role_framework_totals_percent"], {
            "batter": 100.0,
            "bowler": 100.0,
            "all_rounder": 100.0,
        })
        for category, role_blends in metadata["category_role_blends_percent"].items():
            self.assertAlmostEqual(sum(role_blends.values()), 100.0)
            self.assertTrue(all(weight >= 0 for weight in role_blends.values()))
            parameter_weights = metadata["category_parameter_weights_percent"][category]
            for role, weights in parameter_weights.items():
                self.assertAlmostEqual(sum(weights.values()), 100.0)
                self.assertTrue(all(weight >= 0 for weight in weights.values()))
            for framework, blends in metadata["role_path_category_role_blends_percent"].items():
                framework_blend = blends[category]
                if framework_blend:
                    self.assertAlmostEqual(sum(framework_blend.values()), 100.0)
                    effective_weight = sum(
                        parameter_weights[role][metric] * role_weight * result["weights"][category] / 10000
                        for role, role_weight in framework_blend.items()
                        for metric in parameter_weights[role]
                    )
                    self.assertAlmostEqual(effective_weight, result["weights"][category])
        for master_blends in metadata["master_score_role_blends_percent"].values():
            self.assertAlmostEqual(sum(master_blends.values()), 100.0)
        for master_weights in metadata["master_score_parameter_weights_percent"].values():
            for weights in master_weights.values():
                self.assertAlmostEqual(sum(weights.values()), 100.0)
                self.assertTrue(all(weight >= 0 for weight in weights.values()))

        category_metric_keys = {key for specs in CATEGORY_METRICS.values() for key, _, _ in specs}
        self.assertNotIn("bat_strike_rate", category_metric_keys)
        self.assertNotIn("bowl_economy", category_metric_keys)
        scores = [player["final_score"] for player in result["players"]]
        self.assertTrue(all(0 <= score <= 100 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        for key in ("top_batters", "top_bowlers", "top_all_rounders"):
            self.assertLessEqual(len(result[key]), 10)
            role_scores = [player["final_score"] for player in result[key]]
            self.assertEqual(role_scores, sorted(role_scores, reverse=True))
        for player in result["players"]:
            for role, rate_weight in (("batting", 0.10), ("bowling", 0.10)):
                path = player["role_scores"][role]
                if path["final_score"] is None:
                    continue
                weighted = sum(
                    path["category_scores"][category] * weight
                    for category, weight in CATEGORY_WEIGHTS.items()
                    if path["category_scores"][category] is not None
                )
                if path["rate_component_score"] is not None:
                    weighted += path["rate_component_score"] * rate_weight
                self.assertAlmostEqual(path["final_score"], weighted, delta=0.01)
            if player["overall_role"] == "all_rounder":
                self.assertAlmostEqual(
                    player["final_score"],
                    (player["role_scores"]["batting"]["final_score"]
                     + player["role_scores"]["bowling"]["final_score"]) / 2,
                    places=2,
                )
            self.assertTrue(all(
                score is None or 0 <= score <= 100
                for score in player["category_scores"].values()
            ))
            self.assertTrue(all(
                score is None or 0 <= score <= 100
                for score in player["master_scores"].values()
            ))
            self.assertTrue(player["limited_sample"])
            self.assertIn("consistency_score", player["master_scores"])
            self.assertNotIn("master_scores", result["metadata"]["score_formula"])

    def test_unavailable_fielding_is_not_imputed_or_awarded(self):
        balls, matches = sample_data()
        result = calculate_ipl_player_rankings(balls, matches)
        player = result["players"][0]
        self.assertIsNone(player["metrics"]["field_catches"]["value"])
        self.assertFalse(player["metrics"]["field_catches"]["available"])
        self.assertLess(player["data_coverage"], 100)
        self.assertTrue(player["incomplete_data"])

    def test_robust_normalization_clips_outliers_and_inverts_lower_is_better(self):
        values = pd.Series([1, 2, 3, 4, 1000], dtype=float)
        higher = _clip_score(values, True)
        lower = _clip_score(values, False)
        self.assertEqual(float(higher.max()), 100.0)
        self.assertEqual(float(lower.min()), 0.0)
        self.assertEqual(float(higher.iloc[-1]), 100.0)
        self.assertTrue(np.allclose(higher + lower, 100.0))

    def test_standalone_rates_are_normalized_and_economy_is_inverted(self):
        balls, matches = sample_data()
        result = calculate_ipl_player_rankings(balls, matches)

        for player in result["players"]:
            batting = player["role_scores"]["batting"]
            bowling = player["role_scores"]["bowling"]
            self.assertEqual(batting["rate_component_weight_percent"], 10.0)
            self.assertEqual(bowling["rate_component_weight_percent"], 10.0)
            self.assertIn(batting["rate_component_key"], {"bat_strike_rate"})
            self.assertIn(bowling["rate_component_key"], {"bowl_economy"})
            for path in (batting, bowling):
                if path["rate_component_score"] is not None:
                    self.assertGreaterEqual(path["rate_component_score"], 0)
                    self.assertLessEqual(path["rate_component_score"], 100)

    def test_missing_standalone_rate_does_not_get_imputed_or_reweighted(self):
        balls, matches = sample_data()
        balls["is_legal"] = 0
        result = calculate_ipl_player_rankings(balls, matches)

        for player in result["players"]:
            bowling = player["role_scores"]["bowling"]
            self.assertIsNone(bowling["rate_component_score"])
            self.assertEqual(bowling["rate_component_weight_percent"], 10.0)
            category_total = sum(
                score * CATEGORY_WEIGHTS[key]
                for key, score in bowling["category_scores"].items()
                if score is not None
            )
            self.assertAlmostEqual(bowling["final_score"], category_total, delta=0.01)

    def test_strike_rate_and_economy_leaders_include_normalized_scores_and_samples(self):
        balls, matches = sample_data()
        result = calculate_ipl_player_rankings(balls, matches)

        leaders = {leader["key"]: leader for leader in result["top_metric_performers"]}
        self.assertEqual(set(leaders), {"bat_strike_rate", "bowl_economy"})
        for key, leader in leaders.items():
            player = next(row for row in result["players"] if row["player_id"] == leader["player_id"])
            self.assertEqual(leader["value"], player["metrics"][key]["value"])
            self.assertEqual(leader["sample_size"], player["metrics"][key]["sample_size"])
            self.assertGreaterEqual(leader["normalized_score"], 0)
            self.assertLessEqual(leader["normalized_score"], 100)

    def test_role_leaderboards_enforce_minimum_runs_and_wickets(self):
        balls, matches = sample_data(match_count=20)
        fifty_wicket_rows = balls.index[balls["bowler_id"].eq(202)][:50]
        forty_nine_wicket_rows = balls.index[balls["bowler_id"].eq(101)][:49]
        balls.loc[balls["striker_id"].eq(101), "runs"] = 4
        balls.loc[fifty_wicket_rows, "runs"] = 0
        balls.loc[fifty_wicket_rows, "bowler_runs"] = 0
        balls.loc[fifty_wicket_rows, "is_bowler_wicket"] = 1
        balls.loc[forty_nine_wicket_rows, "is_bowler_wicket"] = 1

        result = calculate_ipl_player_rankings(balls, matches)

        self.assertEqual([player["name"] for player in result["top_batters"]], ["Player One"])
        self.assertEqual([player["name"] for player in result["top_bowlers"]], ["Player Two"])
        self.assertTrue(all(player["batting"]["runs"] >= 1000 for player in result["top_batters"]))
        self.assertTrue(all(player["bowling"]["wickets"] >= 50 for player in result["top_bowlers"]))


if __name__ == "__main__":
    unittest.main()

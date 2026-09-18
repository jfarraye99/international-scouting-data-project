import json
import sqlite3
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ProjectOutputTests(unittest.TestCase):
    def test_summary_has_expected_scale(self):
        summary = json.loads((ROOT / "data" / "processed" / "summary.json").read_text())
        self.assertGreaterEqual(summary["players"], 300)
        self.assertGreater(summary["source_rows"], 1_000)
        self.assertGreater(summary["match_rate"], 0.95)

    def test_quality_queue_contains_errors_and_warnings(self):
        issues = pd.read_csv(ROOT / "data" / "processed" / "data_quality_issues.csv")
        self.assertIn("error", set(issues["severity"]))
        self.assertIn("warning", set(issues["severity"]))
        self.assertIn("duplicate_player_id", set(issues["rule"]))

    def test_archetypes_respect_minimum_sample(self):
        outcomes = pd.read_csv(ROOT / "data" / "processed" / "archetype_outcomes.csv")
        self.assertTrue((outcomes["n"] >= 8).all())
        self.assertTrue(outcomes["outcome_rate"].between(0, 1).all())
        self.assertTrue((outcomes["ci_low"] <= outcomes["outcome_rate"]).all())
        self.assertTrue((outcomes["ci_high"] >= outcomes["outcome_rate"]).all())

    def test_sql_views_exist(self):
        database = ROOT / "outputs" / "international_scouting_poc.sqlite"
        with sqlite3.connect(database) as conn:
            views = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        self.assertIn("v_archetype_track_record", views)
        self.assertIn("v_data_quality_queue", views)
        self.assertIn("v_player_projection_board", views)


if __name__ == "__main__":
    unittest.main()


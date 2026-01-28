import unittest


class TestStatsMetricsSmoke(unittest.TestCase):
    def test_compute_stats_metrics(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.validation.stats import compute_stats_metrics

        synthetic = pd.DataFrame(
            {
                "puma": ["1", "1", "2", "2"],
                "AGEP": [10, 30, 40, 12],
                "PINCP": [0, 20000, 50000, 1000],
                "SEX": ["1", "2", "1", "2"],
                "ESR": ["1", "6", "2", "3"],  # includes a child-labor violation (AGEP=12, ESR=3)
            }
        )
        reference = pd.DataFrame(
            {
                "puma": ["1", "1", "2", "2"],
                "AGEP": [11, 31, 41, 13],
                "PINCP": [0, 18000, 52000, 800],
                "SEX": ["1", "2", "1", "2"],
                "ESR": ["6", "6", "2", "6"],
            }
        )

        metrics = compute_stats_metrics(synthetic=synthetic, reference=reference, group_col="puma")
        self.assertIn("marginal_tvd", metrics)
        self.assertIn("association", metrics)
        self.assertIn("hard_rule_violations", metrics)
        self.assertIn("meta", metrics)

        self.assertIn("AGEP_bin", metrics["marginal_tvd"])
        self.assertIn("PINCP_bin", metrics["marginal_tvd"])
        self.assertIn("SEX", metrics["marginal_tvd"])
        self.assertIn("ESR", metrics["marginal_tvd"])

        v = metrics["hard_rule_violations"]["child_labor"]
        self.assertEqual(v["count"], 2)
        self.assertGreaterEqual(v["rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

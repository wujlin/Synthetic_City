import unittest


class TestStatsMetricsTargetsSmoke(unittest.TestCase):
    def test_compute_stats_metrics_against_targets_long(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.validation.stats import compute_stats_metrics_against_targets_long

        synthetic = pd.DataFrame(
            {
                "puma": ["1", "1", "2", "2"],
                "AGEP": [10, 30, 40, 12],
                "PINCP": [0, 20000, 50000, 1000],
                "SEX": ["1", "2", "1", "2"],
                "ESR": ["1", "6", "2", "3"],
            }
        )

        # Build a targets_long table from the synthetic itself (so TVD should be 0).
        edges_age = [0.0, 5.0, 18.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 1000.0]
        synthetic["AGEP_bin"] = pd.cut(synthetic["AGEP"], bins=edges_age, include_lowest=True, right=False).astype(str)

        rows = []
        for puma, g in synthetic.groupby("puma", sort=False):
            for var in ["SEX", "AGEP_bin"]:
                counts = g[var].astype(str).value_counts(dropna=False)
                for cat, cnt in counts.items():
                    rows.append({"puma": str(puma), "variable": var, "category": str(cat), "target": int(cnt)})
        targets_long = pd.DataFrame(rows)

        metrics = compute_stats_metrics_against_targets_long(synthetic=synthetic, targets_long=targets_long, group_col="puma")
        self.assertIn("marginal_tvd", metrics)
        self.assertIn("meta", metrics)
        self.assertIn("SEX", metrics["marginal_tvd"])
        self.assertIn("AGEP_bin", metrics["marginal_tvd"])

        self.assertEqual(metrics["marginal_tvd"]["SEX"]["mean"], 0.0)
        self.assertEqual(metrics["marginal_tvd"]["AGEP_bin"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()


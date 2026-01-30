import unittest


class TestVerasetSmoke(unittest.TestCase):
    def test_device_features_and_center(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.data.veraset import DeviceFeatureSpec, compute_activity_center, compute_device_features

        visits = pd.DataFrame(
            {
                "CAID": ["a", "a", "b", "b", "b"],
                "TOP_CATEGORY": ["Food", "Food", "Shop", "Shop", "Gym"],
                "LOCAL_TIMESTAMP": [
                    "2026-01-01 19:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-03 20:00:00",
                    "2026-01-03 21:00:00",
                    "2026-01-03 09:00:00",
                ],
                "CENSUS_BLOCK_GROUP": ["1", "1", "2", "2", "2"],
                "GEOHASH_5": ["abcde", "abcde", "fghij", "fghij", "fghij"],
                "MINIMUM_DWELL": [5, 10, 1, 2, 3],
            }
        )

        feats = compute_device_features(visits, spec=DeviceFeatureSpec(max_top_categories=2))
        self.assertIn("CAID", feats.columns)
        self.assertIn("n_visits", feats.columns)
        self.assertIn("weekend_ratio", feats.columns)
        self.assertTrue(any(c.startswith("cat__") for c in feats.columns))

        center = compute_activity_center(visits)
        self.assertEqual(set(center.columns), {"CAID", "GEOHASH_5", "CENSUS_BLOCK_GROUP"})


if __name__ == "__main__":
    unittest.main()


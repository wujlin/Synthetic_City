import unittest


class TestBuildingAllocationSmoke(unittest.TestCase):
    def test_allocate_to_buildings(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.building_allocation import allocate_to_buildings

        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5],
                "puma": ["1", "1", "1", "2", "missing"],
                "PINCP": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        buildings = pd.DataFrame(
            {
                "bldg_id": ["b1", "b2", "b3"],
                "puma": ["1", "1", "2"],
                "cap_proxy": [1.0, 3.0, 2.0],
                "price_tier": [1, 2, 1],
            }
        )

        out = allocate_to_buildings(persons=persons, buildings=buildings, method="random", seed=0)
        self.assertEqual(len(out), len(persons))
        self.assertIn("bldg_id", out.columns)
        # missing group gets None
        self.assertTrue(out.loc[out["puma"] == "missing", "bldg_id"].isna().all())

        out2 = allocate_to_buildings(persons=persons, buildings=buildings, method="capacity_only", seed=0)
        self.assertTrue(out2.loc[out2["puma"] == "missing", "bldg_id"].isna().all())

        out3 = allocate_to_buildings(persons=persons, buildings=buildings, method="income_price_match", n_tiers=2, seed=0)
        self.assertTrue(out3.loc[out3["puma"] == "missing", "bldg_id"].isna().all())


if __name__ == "__main__":
    unittest.main()


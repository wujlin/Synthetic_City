import unittest


class TestAssignBuildingsSmoke(unittest.TestCase):
    def test_assign_buildings_within_bg(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.assign_buildings import assign_buildings_within_bg

        persons = pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "bg_geoid": ["bg1", "bg1", "bg2", "bg_missing"],
            }
        )
        buildings = pd.DataFrame(
            {
                "bldg_id": ["b1", "b2", "b3"],
                "bg_geoid": ["bg1", "bg1", "bg2"],
                "footprint_area_m2": [100.0, 50.0, 10.0],
            }
        )

        out = assign_buildings_within_bg(persons=persons, buildings=buildings, seed=0)
        self.assertEqual(len(out), len(persons))
        self.assertIn("bldg_id", out.columns)

        # Within bg1/bg2 should assign a known building; missing bg gets None.
        assigned_bg1 = out.loc[out["bg_geoid"] == "bg1", "bldg_id"].tolist()
        self.assertTrue(all(x in {"b1", "b2"} for x in assigned_bg1))
        assigned_bg2 = out.loc[out["bg_geoid"] == "bg2", "bldg_id"].tolist()
        self.assertTrue(all(x in {"b3"} for x in assigned_bg2))
        self.assertTrue(out.loc[out["bg_geoid"] == "bg_missing", "bldg_id"].isna().all())


if __name__ == "__main__":
    unittest.main()


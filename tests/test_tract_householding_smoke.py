import unittest


class TestTractHouseholdingSmoke(unittest.TestCase):
    def test_householding_matches_person_counts(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.tract_householding import synthesize_households_from_persons

        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(1, 9)],
                "tract_geoid": ["g1"] * 5 + ["g2"] * 3,
                "AGEP_bin": [
                    "[35.0, 45.0)",
                    "[30.0, 35.0)",
                    "[5.0, 18.0)",
                    "[0.0, 5.0)",
                    "[65.0, 75.0)",
                    "[25.0, 35.0)",
                    "[25.0, 35.0)",
                    "[5.0, 18.0)",
                ],
                "EARN_16p_bin": [
                    "[50000.0, 75000.0)",
                    "[25000.0, 50000.0)",
                    "not_16p",
                    "not_16p",
                    "[10000.0, 25000.0)",
                    "[75000.0, 100000.0)",
                    "[150000.0, 200000.0)",
                    "not_16p",
                ],
            }
        )

        shell_targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g1", "g2", "g2"],
                "household_type": ["family", "nonfamily", "nonfamily", "family", "nonfamily"],
                "household_size": [3, 1, 1, 2, 1],
                "n_target": [1, 1, 1, 1, 1],
            }
        )
        income_targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2"],
                "HHINCP_bin": ["[0.0, 10000.0)", "[50000.0, 60000.0)", "[0.0, 10000.0)", "[125000.0, 150000.0)"],
                "n_target": [2, 1, 1, 1],
            }
        )

        persons_hh, households, meta = synthesize_households_from_persons(
            persons=persons,
            shell_targets=shell_targets,
            income_targets=income_targets,
            group_col="tract_geoid",
            person_id_col="person_id",
            age_col="AGEP_bin",
            earn_col="EARN_16p_bin",
            household_id_prefix="toyhh",
            seed=0,
        )

        self.assertEqual(int(meta["n_persons"]), len(persons))
        self.assertEqual(int(meta["persons_with_household_id"]), len(persons))
        self.assertTrue(persons_hh["household_id"].notna().all())
        self.assertTrue(persons_hh["household_role"].notna().all())
        self.assertTrue(persons_hh["household_type"].isin(["family", "nonfamily"]).all())

        hh_counts = (
            persons_hh.groupby(["tract_geoid", "household_id"], as_index=False)
            .size()
            .rename(columns={"size": "n_members"})
        )
        chk = hh_counts.merge(
            households[["tract_geoid", "household_id", "household_size", "HHINCP_bin"]],
            on=["tract_geoid", "household_id"],
            how="left",
        )
        self.assertTrue((chk["n_members"] == chk["household_size"]).all())
        self.assertTrue(chk["HHINCP_bin"].notna().all())
        self.assertEqual(int(households["household_id"].nunique()), len(households))
        self.assertEqual(int(households.shape[0]), 5)


if __name__ == "__main__":
    unittest.main()

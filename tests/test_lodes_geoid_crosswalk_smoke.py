import unittest

import pandas as pd

from synthpop.data.lodes import (
    prepare_internal_study_tract_od,
    remap_tract_od_geoids,
    remap_tract_wac_geoids,
)


class LodesGeoidCrosswalkSmokeTest(unittest.TestCase):
    def test_remap_recovers_connecticut_planning_region_study_tracts(self):
        study_tracts = {"09110400101", "09110400102"}
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["09001010101", "09001010102"],
                "work_tract_geoid": ["09001010102", "09001010101"],
                "S000": [80, 20],
                "SA01": [30, 5],
            }
        )
        before, _, before_summary = prepare_internal_study_tract_od(
            tract_od=tract_od,
            study_tracts=study_tracts,
        )
        self.assertEqual(before.shape[0], 0)
        self.assertEqual(before_summary["n_origin_tracts_with_any_jobs"], 0)

        crosswalk = pd.DataFrame(
            {
                "legacy_tract_geoid": ["09001010101", "09001010102"],
                "tract_geoid": ["09110400101", "09110400102"],
                "weight": [1.0, 1.0],
            }
        )
        remapped = remap_tract_od_geoids(tract_od, crosswalk)
        after, origin_stats, after_summary = prepare_internal_study_tract_od(
            tract_od=remapped,
            study_tracts=study_tracts,
        )

        self.assertEqual(after.shape[0], 2)
        self.assertAlmostEqual(float(after["S000"].sum()), 100.0)
        self.assertEqual(after_summary["n_origin_tracts_with_any_jobs"], 2)
        self.assertEqual(after_summary["n_origin_tracts_with_internal_dest"], 2)
        self.assertTrue((origin_stats["total_jobs_from_origin"] > 0).all())
        self.assertEqual(set(after["home_tract_geoid"]), study_tracts)
        self.assertEqual(set(after["work_tract_geoid"]), study_tracts)

    def test_od_remap_splits_weighted_legacy_tracts_and_preserves_total(self):
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["09001010101"],
                "work_tract_geoid": ["09001010200"],
                "S000": [100.0],
                "SE01": [40.0],
            }
        )
        crosswalk = pd.DataFrame(
            {
                "legacy_tract_geoid": ["09001010101", "09001010101", "09001010200"],
                "tract_geoid": ["09110400101", "09110400102", "09110400300"],
                "weight": [0.25, 0.75, 1.0],
            }
        )
        remapped = remap_tract_od_geoids(tract_od, crosswalk)

        self.assertEqual(remapped.shape[0], 2)
        self.assertAlmostEqual(float(remapped["S000"].sum()), 100.0)
        self.assertAlmostEqual(float(remapped["SE01"].sum()), 40.0)
        counts = dict(zip(remapped["home_tract_geoid"], remapped["S000"]))
        self.assertAlmostEqual(counts["09110400101"], 25.0)
        self.assertAlmostEqual(counts["09110400102"], 75.0)
        self.assertEqual(set(remapped["work_tract_geoid"]), {"09110400300"})

    def test_wac_remap_recomputes_shares_after_weighted_grouping(self):
        tract_wac = pd.DataFrame(
            {
                "tract_geoid": ["09001010101", "09001010200"],
                "C000": [100.0, 50.0],
                "CA01": [40.0, 10.0],
                "CA02": [60.0, 40.0],
                "CE01": [25.0, 5.0],
                "CE02": [75.0, 45.0],
            }
        )
        crosswalk = pd.DataFrame(
            {
                "legacy_tract_geoid": ["09001010101", "09001010101", "09001010200"],
                "tract_geoid": ["09110400101", "09110400102", "09110400102"],
                "weight": [0.5, 0.5, 1.0],
            }
        )
        remapped = remap_tract_wac_geoids(tract_wac, crosswalk)

        self.assertEqual(set(remapped["tract_geoid"]), {"09110400101", "09110400102"})
        totals = dict(zip(remapped["tract_geoid"], remapped["C000"]))
        self.assertAlmostEqual(totals["09110400101"], 50.0)
        self.assertAlmostEqual(totals["09110400102"], 100.0)
        row = remapped.set_index("tract_geoid").loc["09110400102"]
        self.assertAlmostEqual(float(row["share_CA01"]), 30.0 / 100.0)
        self.assertAlmostEqual(float(row["share_CA02"]), 70.0 / 100.0)


if __name__ == "__main__":
    unittest.main()

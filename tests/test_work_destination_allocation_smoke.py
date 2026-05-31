from __future__ import annotations

import unittest

import pandas as pd

from synthpop.spatial.work_destination_allocation import assign_work_destination_tract


class TestWorkDestinationAllocationSmoke(unittest.TestCase):
    def test_assigns_destination_tract_from_od(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": ["p1", "p2", "p3"],
                "tract_geoid": ["t1", "t1", "t2"],
                "is_worker": [True, True, False],
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1", "t2"],
                "work_tract_geoid": ["t2", "t3", "t2"],
                "S000": [8, 2, 5],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            home_group_col="tract_geoid",
            out_col="work_tract_geoid",
            work_eligible_col="is_worker",
            seed=0,
        )
        self.assertEqual(meta["work_eligible"], 2)
        self.assertEqual(meta["work_destination_assigned"], 2)
        self.assertEqual(meta["destination_mode_counts"]["ineligible"], 1)
        self.assertTrue(set(out.loc[out["is_worker"], "work_tract_geoid"].dropna().tolist()).issubset({"t2", "t3"}))
        self.assertTrue((out.loc[~out["is_worker"], "work_destination_mode"] == "ineligible").all())

    def test_marks_unassigned_when_origin_has_no_internal_destination(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": ["p1"],
                "tract_geoid": ["t9"],
                "is_worker": [True],
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1"],
                "work_tract_geoid": ["t2"],
                "S000": [5],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            seed=0,
        )
        self.assertIsNone(out.loc[0, "work_tract_geoid"])
        self.assertEqual(out.loc[0, "work_destination_mode"], "unassigned_no_destination")
        self.assertEqual(meta["work_destination_unassigned"], 1)

    def test_distance_beta_reweights_destination_choices(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(200)],
                "tract_geoid": ["t1"] * 200,
                "is_worker": [True] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1"],
                "work_tract_geoid": ["near", "far"],
                "S000": [100, 100],
                "distance_km": [1.0, 40.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.2,
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertGreater(int(counts.get("near", 0)), int(counts.get("far", 0)))
        self.assertTrue(meta["use_distance"])

    def test_segment_weight_reweights_destination_choices(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(200)],
                "tract_geoid": ["t1"] * 200,
                "is_worker": [True] * 200,
                "EARN_16p_bin": ["ge_100k"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1"],
                "work_tract_geoid": ["lowseg", "highseg"],
                "S000": [100, 100],
                "work_share_CE01": [0.9, 0.1],
                "work_share_CE02": [0.1, 0.1],
                "work_share_CE03": [0.0, 0.8],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            earn_col="EARN_16p_bin",
            destination_segment_weight=2.0,
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertGreater(int(counts.get("highseg", 0)), int(counts.get("lowseg", 0)))
        self.assertTrue(meta["use_destination_earn_prior"])

    def test_od_age_and_earn_segments_reweight_destination_choices(self) -> None:
        low_earn_young = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(200)],
                "tract_geoid": ["t1"] * 200,
                "is_worker": [True] * 200,
                "EARN_16p_bin": ["lt_25k"] * 200,
                "AGEP_bin": ["[18.0, 25.0)"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1"],
                "work_tract_geoid": ["young_low", "older_high"],
                "S000": [100, 100],
                "SE01": [90, 10],
                "SE02": [10, 10],
                "SE03": [5, 95],
                "SA01": [80, 20],
                "SA02": [20, 80],
                "SA03": [10, 90],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=low_earn_young,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            earn_col="EARN_16p_bin",
            age_col="AGEP_bin",
            od_earn_segment_weight=1.0,
            od_age_segment_weight=1.0,
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertGreater(int(counts.get("young_low", 0)), int(counts.get("older_high", 0)))
        self.assertTrue(meta["use_od_earn_prior"])
        self.assertTrue(meta["use_od_age_prior"])

    def test_destination_accessibility_reweights_destination_choices(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(200)],
                "tract_geoid": ["t1"] * 200,
                "is_worker": [True] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1"],
                "work_tract_geoid": ["low_access", "high_access"],
                "S000": [100, 100],
                "work_access_jobs_gravity": [10.0, 1000.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=1.0,
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertGreater(int(counts.get("high_access", 0)), int(counts.get("low_access", 0)))
        self.assertTrue(meta["use_destination_access_prior"])

    def test_balanced_mode_preserves_origin_destination_totals(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(10)],
                "tract_geoid": ["t1"] * 10,
                "is_worker": [True] * 10,
                "EARN_16p_bin": ["lt_25k"] * 6 + ["ge_100k"] * 4,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["t1", "t1"],
                "work_tract_geoid": ["d1", "d2"],
                "S000": [6, 4],
                "SE01": [6, 0],
                "SE02": [0, 0],
                "SE03": [0, 4],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            earn_col="EARN_16p_bin",
            od_earn_segment_weight=1.0,
            assignment_mode="balanced",
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertEqual(int(counts.get("d1", 0)), 6)
        self.assertEqual(int(counts.get("d2", 0)), 4)
        self.assertEqual(meta["assignment_mode"], "balanced")

    def test_center_access_and_same_county_can_reweight_choices(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(200)],
                "tract_geoid": ["26001000100"] * 200,
                "is_worker": [True] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100", "26001000100"],
                "work_tract_geoid": ["26001000200", "26003000100"],
                "S000": [100, 100],
                "work_access_job_centers_gravity": [1000.0, 10.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=1.0,
            same_county_weight=0.5,
            seed=0,
        )
        counts = out["work_tract_geoid"].value_counts(dropna=False).to_dict()
        self.assertGreater(int(counts.get("26001000200", 0)), int(counts.get("26003000100", 0)))
        self.assertTrue(meta["use_destination_center_prior"])

    def test_type_specific_utility_coefficients_change_choice_by_earn_segment(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(400)],
                "tract_geoid": ["26001000100"] * 400,
                "is_worker": [True] * 400,
                "EARN_16p_bin": ["lt_25k"] * 200 + ["ge_100k"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100", "26001000100"],
                "work_tract_geoid": ["26001000200", "26003000100"],
                "S000": [100, 100],
                "distance_km": [2.0, 20.0],
                "work_access_jobs_gravity": [20.0, 200.0],
                "work_access_job_centers_gravity": [20.0, 500.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.12,
            earn_col="EARN_16p_bin",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.8,
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=1.0,
            same_county_weight=0.8,
            distance_earn_multiplier_map={"CE01": 1.6, "CE03": 0.5},
            destination_access_earn_multiplier_map={"CE01": 0.7, "CE03": 1.4},
            destination_center_earn_multiplier_map={"CE01": 0.6, "CE03": 1.6},
            same_county_earn_multiplier_map={"CE01": 1.5, "CE03": 0.6},
            seed=0,
        )
        chosen = out.loc[out["is_worker"], ["EARN_16p_bin", "work_tract_geoid"]].copy()
        low_far = int(((chosen["EARN_16p_bin"] == "lt_25k") & (chosen["work_tract_geoid"] == "26003000100")).sum())
        high_far = int(((chosen["EARN_16p_bin"] == "ge_100k") & (chosen["work_tract_geoid"] == "26003000100")).sum())
        self.assertGreater(high_far, low_far)
        self.assertTrue(meta["use_earn_type_coefficients"])
        self.assertEqual(meta["distance_earn_multiplier_map"]["CE01"], 1.6)
        self.assertEqual(meta["destination_center_earn_multiplier_map"]["CE03"], 1.6)

    def test_hierarchical_county_mode_assigns_valid_destinations(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(100)],
                "tract_geoid": ["26001000100"] * 100,
                "is_worker": [True] * 100,
                "EARN_16p_bin": ["lt_25k"] * 50 + ["ge_100k"] * 50,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100"] * 4,
                "work_tract_geoid": ["26001000200", "26001000300", "26003000100", "26003000200"],
                "S000": [60, 40, 50, 50],
                "distance_km": [2.0, 4.0, 20.0, 25.0],
                "work_access_jobs_gravity": [20.0, 25.0, 200.0, 180.0],
                "work_access_job_centers_gravity": [30.0, 35.0, 400.0, 300.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.08,
            earn_col="EARN_16p_bin",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.5,
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=0.5,
            same_county_weight=0.15,
            assignment_mode="hierarchical_county",
            seed=0,
        )
        self.assertEqual(meta["assignment_mode"], "hierarchical_county")
        self.assertEqual(meta["work_destination_assigned"], 100)
        self.assertTrue(set(out["work_tract_geoid"].dropna().tolist()).issubset(set(tract_od["work_tract_geoid"].tolist())))

    def test_hierarchical_county_center_mode_assigns_valid_destinations(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(120)],
                "tract_geoid": ["26001000100"] * 120,
                "is_worker": [True] * 120,
                "EARN_16p_bin": ["lt_25k"] * 60 + ["ge_100k"] * 60,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100"] * 6,
                "work_tract_geoid": [
                    "26001000200",
                    "26001000300",
                    "26003000100",
                    "26003000200",
                    "26003000300",
                    "26003000400",
                ],
                "work_center_geoid": [
                    "26001000200",
                    "26001000200",
                    "26003000100",
                    "26003000100",
                    "26003000400",
                    "26003000400",
                ],
                "S000": [40, 20, 15, 15, 15, 15],
                "distance_km": [2.0, 3.0, 18.0, 20.0, 22.0, 24.0],
                "work_access_jobs_gravity": [30.0, 28.0, 180.0, 170.0, 220.0, 210.0],
                "work_access_job_centers_gravity": [35.0, 35.0, 260.0, 260.0, 400.0, 400.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.08,
            earn_col="EARN_16p_bin",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.5,
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=0.5,
            same_county_weight=0.15,
            assignment_mode="hierarchical_county_center",
            seed=0,
        )
        self.assertEqual(meta["assignment_mode"], "hierarchical_county_center")
        self.assertEqual(meta["work_destination_assigned"], 120)
        self.assertTrue(set(out["work_tract_geoid"].dropna().tolist()).issubset(set(tract_od["work_tract_geoid"].tolist())))

    def test_same_home_center_weight_reweights_center_choice(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(400)],
                "tract_geoid": ["26001000100"] * 400,
                "is_worker": [True] * 400,
                "EARN_16p_bin": ["lt_25k"] * 200 + ["ge_100k"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100"] * 4,
                "work_tract_geoid": ["26001000200", "26001000300", "26001000400", "26001000500"],
                "work_center_geoid": ["26001000200", "26001000200", "26001000400", "26001000400"],
                "home_center_geoid": ["26001000200"] * 4,
                "S000": [50, 50, 50, 50],
                "distance_km": [5.0, 6.0, 5.0, 6.0],
                "work_access_jobs_gravity": [100.0, 100.0, 100.0, 100.0],
                "work_access_job_centers_gravity": [100.0, 100.0, 100.0, 100.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.01,
            earn_col="EARN_16p_bin",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.1,
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=0.1,
            same_home_center_weight=1.2,
            same_home_center_earn_multiplier_map={"CE01": 1.6, "CE03": 0.6},
            assignment_mode="hierarchical_county_center",
            seed=0,
        )
        chosen = out.loc[out["is_worker"], ["EARN_16p_bin", "work_tract_geoid"]].copy()
        home_center_dests = {"26001000200", "26001000300"}
        low_same_center = int(
            ((chosen["EARN_16p_bin"] == "lt_25k") & (chosen["work_tract_geoid"].isin(home_center_dests))).sum()
        )
        high_same_center = int(
            ((chosen["EARN_16p_bin"] == "ge_100k") & (chosen["work_tract_geoid"].isin(home_center_dests))).sum()
        )
        self.assertGreater(low_same_center, high_same_center)
        self.assertTrue(meta["use_same_home_center_prior"])
        self.assertEqual(meta["same_home_center_earn_multiplier_map"]["CE01"], 1.6)

    def test_hierarchical_regime_mode_changes_regime_choice_by_type(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(400)],
                "tract_geoid": ["26001000100"] * 400,
                "is_worker": [True] * 400,
                "EARN_16p_bin": ["lt_25k"] * 200 + ["ge_100k"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100"] * 4,
                "work_tract_geoid": ["26001000100", "26001000200", "26001000300", "26003000100"],
                "work_center_geoid": ["26001000100", "26001000100", "26001000300", "26003000100"],
                "home_center_geoid": ["26001000100"] * 4,
                "S000": [40, 40, 40, 40],
                "distance_km": [1.0, 3.0, 6.0, 20.0],
                "work_access_jobs_gravity": [40.0, 45.0, 70.0, 120.0],
                "work_access_job_centers_gravity": [40.0, 40.0, 80.0, 160.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            distance_col="distance_km",
            distance_beta=0.08,
            earn_col="EARN_16p_bin",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.4,
            destination_center_col="work_access_job_centers_gravity",
            destination_center_weight=0.6,
            same_tract_weight=1.0,
            same_home_center_weight=0.6,
            same_county_weight=0.2,
            same_tract_earn_multiplier_map={"CE01": 1.4, "CE03": 0.7},
            assignment_mode="hierarchical_regime",
            seed=0,
        )
        chosen = out.loc[out["is_worker"], ["EARN_16p_bin", "work_tract_geoid"]].copy()
        low_same = int(((chosen["EARN_16p_bin"] == "lt_25k") & (chosen["work_tract_geoid"] == "26001000100")).sum())
        high_same = int(((chosen["EARN_16p_bin"] == "ge_100k") & (chosen["work_tract_geoid"] == "26001000100")).sum())
        self.assertGreater(low_same, high_same)
        self.assertEqual(meta["assignment_mode"], "hierarchical_regime")
        self.assertTrue(meta["use_same_tract_prior"])

    def test_latent_job_family_prior_reweights_destination_by_inferred_family(self) -> None:
        persons = pd.DataFrame(
            {
                "person_id": [f"p{i}" for i in range(400)],
                "tract_geoid": ["26001000100"] * 400,
                "is_worker": [True] * 400,
                "EARN_16p_bin": ["ge_100k"] * 200 + ["lt_25k"] * 200,
                "AGEP_bin": ["[35.0, 45.0)"] * 200 + ["[18.0, 25.0)"] * 200,
                "SCHL_allpop": ["bachelor_plus"] * 200 + ["high_school_or_ged"] * 200,
            }
        )
        tract_od = pd.DataFrame(
            {
                "home_tract_geoid": ["26001000100", "26001000100"],
                "work_tract_geoid": ["26001000200", "26001000300"],
                "S000": [100, 100],
                "work_share_JF_SERVICE": [0.1, 0.8],
                "work_share_JF_INDUSTRIAL": [0.2, 0.15],
                "work_share_JF_PROFESSIONAL": [0.7, 0.05],
                "work_access_jobs_gravity": [100.0, 100.0],
            }
        )
        out, meta = assign_work_destination_tract(
            persons=persons,
            tract_od=tract_od,
            work_eligible_col="is_worker",
            earn_col="EARN_16p_bin",
            age_col="AGEP_bin",
            schl_col="SCHL_allpop",
            destination_access_col="work_access_jobs_gravity",
            destination_access_weight=0.1,
            job_family_weight=1.8,
            seed=0,
        )
        chosen = out.loc[out["is_worker"], ["SCHL_allpop", "work_tract_geoid"]].copy()
        bach_prof = int(((chosen["SCHL_allpop"] == "bachelor_plus") & (chosen["work_tract_geoid"] == "26001000200")).sum())
        hs_prof = int(((chosen["SCHL_allpop"] == "high_school_or_ged") & (chosen["work_tract_geoid"] == "26001000200")).sum())
        self.assertGreater(bach_prof, hs_prof)
        self.assertTrue(meta["use_job_family_prior"])
        self.assertIn("JF_PROFESSIONAL", meta["job_family_counts_among_eligible"])


if __name__ == "__main__":
    unittest.main()

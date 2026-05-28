import unittest


class TestPumaToSmallAreaSmoke(unittest.TestCase):
    def test_b01001_records_can_emit_age_sex_cross(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from tools.build_acs_targets_long_michigan import _b01001_records

        row = {"tract_geoid": "26163500100"}
        for i in range(1, 50):
            row[f"B01001_{i:03d}E"] = 0
        row.update(
            {
                "B01001_002E": 30,
                "B01001_026E": 40,
                "B01001_003E": 4,
                "B01001_027E": 6,
                "B01001_004E": 8,
                "B01001_005E": 9,
                "B01001_006E": 9,
                "B01001_028E": 10,
                "B01001_029E": 12,
                "B01001_030E": 12,
            }
        )
        records = _b01001_records(
            pd.DataFrame([row]),
            group_col="tract_geoid",
            include_age_sex_cross=True,
        )
        cross = [r for r in records if r["variable"] == "AGEP_SEX_cross"]
        self.assertEqual(len(cross), 20)
        by_cat = {r["category"]: r["target"] for r in cross}
        self.assertAlmostEqual(by_cat["[0.0, 5.0)__1"], 4.0)
        self.assertAlmostEqual(by_cat["[0.0, 5.0)__2"], 6.0)
        self.assertAlmostEqual(by_cat["[5.0, 18.0)__1"], 26.0)
        self.assertAlmostEqual(by_cat["[5.0, 18.0)__2"], 34.0)

    def test_allocate_region_type_counts_matches_age_sex_cross(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import (
            allocate_region_type_counts,
            build_type_catalog,
            joint_wide_to_type_counts,
            summarize_type_allocation_against_targets,
        )

        schema = {
            "variable_order": ["AGEP_bin", "SEX"],
            "categories": {
                "AGEP_bin": ["young", "old"],
                "SEX": ["1", "2"],
            },
        }
        joint_wide = pd.DataFrame(
            {
                "puma_uid": ["2600100"],
                "total_person_weight": [100.0],
                "p_joint_000": [0.25],
                "p_joint_001": [0.25],
                "p_joint_002": [0.25],
                "p_joint_003": [0.25],
            }
        )
        type_counts = joint_wide_to_type_counts(joint_wide=joint_wide, schema=schema).drop(columns=["puma_uid"])
        type_catalog = build_type_catalog(schema=schema)
        self.assertIn("AGEP_SEX_cross", type_catalog.columns)

        targets = pd.DataFrame(
            {
                "tract_geoid": ["t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2"],
                "variable": ["AGEP_SEX_cross"] * 8,
                "category": [
                    "young__1",
                    "young__2",
                    "old__1",
                    "old__2",
                    "young__1",
                    "young__2",
                    "old__1",
                    "old__2",
                ],
                "target": [10.0, 5.0, 20.0, 15.0, 15.0, 20.0, 5.0, 10.0],
            }
        )
        alloc, meta = allocate_region_type_counts(
            type_counts=type_counts,
            hard_targets_long=targets,
            type_catalog=type_catalog,
            group_col="tract_geoid",
            hard_variables=["AGEP_SEX_cross"],
            tol=1e-8,
            max_iters=200,
        )
        self.assertTrue(meta["converged"])
        summary = summarize_type_allocation_against_targets(
            allocation_long=alloc,
            targets_long=targets,
            type_catalog=type_catalog,
            group_col="tract_geoid",
        )
        self.assertLessEqual(summary["variables"]["AGEP_SEX_cross"]["max_abs_err"], 1e-5)

    def test_allocate_region_type_counts_matches_hard_marginals(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import (
            allocate_region_type_counts,
            build_type_catalog,
            build_type_to_group_prior,
            joint_wide_to_type_counts,
            summarize_type_allocation_against_targets,
        )

        schema = {
            "variable_order": ["AGEP_bin", "SEX"],
            "categories": {
                "AGEP_bin": ["young", "old"],
                "SEX": ["1", "2"],
            },
        }
        joint_wide = pd.DataFrame(
            {
                "puma_uid": ["2600100"],
                "total_person_weight": [100.0],
                "p_joint_000": [0.20],
                "p_joint_001": [0.10],
                "p_joint_002": [0.30],
                "p_joint_003": [0.40],
            }
        )
        type_counts = joint_wide_to_type_counts(joint_wide=joint_wide, schema=schema)
        type_counts = type_counts.drop(columns=["puma_uid"])
        type_catalog = build_type_catalog(schema=schema)

        targets = pd.DataFrame(
            {
                "tract_geoid": [
                    "t1",
                    "t1",
                    "t2",
                    "t2",
                    "t1",
                    "t1",
                    "t2",
                    "t2",
                ],
                "variable": [
                    "AGEP_bin",
                    "AGEP_bin",
                    "AGEP_bin",
                    "AGEP_bin",
                    "SEX",
                    "SEX",
                    "SEX",
                    "SEX",
                ],
                "category": ["young", "old", "young", "old", "1", "2", "1", "2"],
                "target": [15.0, 25.0, 15.0, 45.0, 20.0, 20.0, 30.0, 30.0],
            }
        )

        alloc, meta = allocate_region_type_counts(
            type_counts=type_counts,
            hard_targets_long=targets,
            type_catalog=type_catalog,
            group_col="tract_geoid",
            hard_variables=["AGEP_bin", "SEX"],
            tol=1e-8,
            max_iters=200,
        )
        self.assertTrue(meta["converged"])

        by_type = alloc.groupby("type_idx", as_index=False)["count"].sum().sort_values("type_idx")
        ref_type = type_counts[["type_idx", "count"]].sort_values("type_idx")
        self.assertEqual(by_type["type_idx"].tolist(), ref_type["type_idx"].tolist())
        for got, want in zip(by_type["count"].tolist(), ref_type["count"].tolist()):
            self.assertAlmostEqual(got, want, places=6)

        summary = summarize_type_allocation_against_targets(
            allocation_long=alloc,
            targets_long=targets,
            type_catalog=type_catalog,
            group_col="tract_geoid",
        )
        self.assertLessEqual(summary["variables"]["AGEP_bin"]["max_abs_err"], 1e-5)
        self.assertLessEqual(summary["variables"]["SEX"]["max_abs_err"], 1e-5)

    def test_aggregate_home_origin_profiles_from_safegraph(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.data.poi_safegraph import aggregate_home_origin_profiles

        poi = pd.DataFrame(
            {
                "region": ["MI", "MI", "TX"],
                "top_category": ["School", "Office", "School"],
                "visitor_home_cbgs": [
                    '{"261635001001": 2, "261635002002": 3}',
                    '{"261635001002": 5}',
                    '{"482012411011": 7}',
                ],
            }
        )

        out = aggregate_home_origin_profiles(
            merged_poi=poi,
            group_level="tract",
            region_filter="MI",
            top_n_categories=4,
        ).sort_values("tract_geoid")

        self.assertEqual(out["tract_geoid"].tolist(), ["26163500100", "26163500200"])
        counts = dict(zip(out["tract_geoid"], out["home_origin_count"]))
        self.assertAlmostEqual(counts["26163500100"], 7.0, places=6)
        self.assertAlmostEqual(counts["26163500200"], 3.0, places=6)

        school_col = "cat__school"
        office_col = "cat__office"
        self.assertIn(school_col, out.columns)
        self.assertIn(office_col, out.columns)
        row1 = out[out["tract_geoid"] == "26163500100"].iloc[0]
        row2 = out[out["tract_geoid"] == "26163500200"].iloc[0]
        self.assertAlmostEqual(float(row1[school_col]), 2.0 / 7.0, places=6)
        self.assertAlmostEqual(float(row1[office_col]), 5.0 / 7.0, places=6)
        self.assertAlmostEqual(float(row2[school_col]), 1.0, places=6)

    def test_predict_targets_from_group_features_and_blend_prior(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import (
            blend_prior_targets_long,
            compare_targets_long,
            predict_targets_from_group_features,
        )

        group_features = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g2", "g3", "g4"],
                "cat__school": [0.9, 0.2, 0.8, 0.1],
                "cat__office": [0.1, 0.8, 0.2, 0.9],
                "home_origin_share": [0.25, 0.25, 0.25, 0.25],
            }
        )
        reference = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"],
                "variable": ["EARN_16p_bin"] * 8,
                "category": ["low", "high"] * 4,
                "target": [0.9, 0.1, 0.2, 0.8, 0.8, 0.2, 0.1, 0.9],
            }
        )
        predicted, meta = predict_targets_from_group_features(
            group_features=group_features,
            reference_targets_long=reference,
            group_col="tract_geoid",
            region_col=None,
            variables=["EARN_16p_bin"],
            ridge_alpha=1e-4,
            min_train_groups=1,
        )
        self.assertIn("fit_against_reference", meta)
        fit = compare_targets_long(
            predicted_targets_long=predicted,
            reference_targets_long=reference,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
        )
        self.assertLess(fit["variables"]["EARN_16p_bin"]["mean_tvd"], 1e-3)

        base = reference.copy()
        base["target"] = 0.5
        blended = blend_prior_targets_long(
            base_targets_long=base,
            extra_targets_long=predicted,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
            variables=["EARN_16p_bin"],
            base_weight=1.0,
            extra_weight=3.0,
        )
        base_cmp = compare_targets_long(
            predicted_targets_long=base,
            reference_targets_long=predicted,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
        )
        blend_cmp = compare_targets_long(
            predicted_targets_long=blended,
            reference_targets_long=predicted,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
        )
        self.assertLess(
            blend_cmp["variables"]["EARN_16p_bin"]["mean_tvd"],
            base_cmp["variables"]["EARN_16p_bin"]["mean_tvd"],
        )

    def test_low_rank_project_targets_long_preserves_normalization(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import low_rank_project_targets_long

        targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2", "g3", "g3", "g1", "g1", "g2", "g2", "g3", "g3"],
                "variable": ["SCHL_allpop"] * 6 + ["EARN_16p_bin"] * 6,
                "category": ["low", "high"] * 3 + ["low", "high"] * 3,
                "target": [0.8, 0.2, 0.5, 0.5, 0.2, 0.8, 0.7, 0.3, 0.5, 0.5, 0.3, 0.7],
            }
        )
        out, meta = low_rank_project_targets_long(
            targets_long=targets,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
            variables=["SCHL_allpop", "EARN_16p_bin"],
            rank=1,
        )
        self.assertTrue(meta["applied"])
        self.assertGreater(meta["explained_frob_ratio"], 0.0)
        denom = (
            out.groupby(["tract_geoid", "variable"], as_index=False)["target"]
            .sum()
            .sort_values(["tract_geoid", "variable"])
        )
        for got in denom["target"].tolist():
            self.assertAlmostEqual(float(got), 1.0, places=6)

    def test_low_rank_plus_sparse_project_targets_long_can_recover_input(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import (
            compare_targets_long,
            low_rank_plus_sparse_project_targets_long,
        )

        targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2", "g3", "g3", "g1", "g1", "g2", "g2", "g3", "g3"],
                "variable": ["SCHL_allpop"] * 6 + ["EARN_16p_bin"] * 6,
                "category": ["low", "high"] * 3 + ["low", "high"] * 3,
                "target": [0.82, 0.18, 0.48, 0.52, 0.16, 0.84, 0.72, 0.28, 0.58, 0.42, 0.21, 0.79],
            }
        )
        out, meta = low_rank_plus_sparse_project_targets_long(
            targets_long=targets,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
            variables=["SCHL_allpop", "EARN_16p_bin"],
            rank=1,
            sparse_weight=1.0,
            sparse_threshold=0.0,
        )
        self.assertTrue(meta["applied"])
        self.assertGreater(meta["sparse_retained_fraction"], 0.0)
        fit = compare_targets_long(
            predicted_targets_long=out,
            reference_targets_long=targets,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
        )
        self.assertLess(fit["variables"]["SCHL_allpop"]["mean_tvd"], 1e-6)
        self.assertLess(fit["variables"]["EARN_16p_bin"]["mean_tvd"], 1e-6)

    def test_low_rank_plus_smooth_project_targets_long_preserves_normalization(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import low_rank_plus_smooth_project_targets_long

        targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"] * 2,
                "variable": ["SCHL_allpop"] * 8 + ["EARN_16p_bin"] * 8,
                "category": ["low", "high"] * 4 + ["low", "high"] * 4,
                "target": [
                    0.80,
                    0.20,
                    0.72,
                    0.28,
                    0.25,
                    0.75,
                    0.30,
                    0.70,
                    0.78,
                    0.22,
                    0.68,
                    0.32,
                    0.35,
                    0.65,
                    0.40,
                    0.60,
                ],
            }
        )
        group_features = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g2", "g3", "g4"],
                "puma_uid": ["p1", "p1", "p1", "p1"],
                "cat__school": [0.9, 0.85, 0.2, 0.25],
                "cat__office": [0.1, 0.15, 0.8, 0.75],
                "home_origin_share": [0.25, 0.25, 0.25, 0.25],
            }
        )
        out, meta = low_rank_plus_smooth_project_targets_long(
            targets_long=targets,
            group_features=group_features,
            group_col="tract_geoid",
            region_col="puma_uid",
            variable_col="variable",
            category_col="category",
            target_col="target",
            variables=["SCHL_allpop", "EARN_16p_bin"],
            rank=1,
            smooth_weight=0.5,
            smooth_knn=2,
        )
        self.assertTrue(meta["applied"])
        self.assertEqual(meta["smooth_knn"], 2)
        denom = (
            out.groupby(["tract_geoid", "variable"], as_index=False)["target"]
            .sum()
            .sort_values(["tract_geoid", "variable"])
        )
        for got in denom["target"].tolist():
            self.assertAlmostEqual(float(got), 1.0, places=6)

    def test_build_type_to_group_prior_with_residual_energy(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.puma_to_small_area import build_type_catalog, build_type_to_group_prior

        schema = {
            "variable_order": ["EARN_16p_bin"],
            "categories": {
                "EARN_16p_bin": ["low", "high"],
            },
        }
        type_catalog = build_type_catalog(schema=schema)
        groups = ["g1", "g2"]
        base_targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2"],
                "variable": ["EARN_16p_bin"] * 4,
                "category": ["low", "high", "low", "high"],
                "target": [0.5, 0.5, 0.5, 0.5],
            }
        )
        residual_targets = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g1", "g2", "g2"],
                "variable": ["EARN_16p_bin"] * 4,
                "category": ["low", "high", "low", "high"],
                "target": [0.8, 0.2, 0.2, 0.8],
            }
        )
        prior, meta = build_type_to_group_prior(
            type_catalog=type_catalog,
            groups=groups,
            prior_targets_long=base_targets,
            residual_targets_long=residual_targets,
            group_col="tract_geoid",
            variable_col="variable",
            category_col="category",
            target_col="target",
            prior_variables=[],
            residual_variables=["EARN_16p_bin"],
            residual_ratio_clip=10.0,
        )
        self.assertIn("EARN_16p_bin", meta["residual_variables"])
        low_row = type_catalog[type_catalog["EARN_16p_bin"] == "low"].index[0]
        high_row = type_catalog[type_catalog["EARN_16p_bin"] == "high"].index[0]
        self.assertGreater(float(prior[low_row, 0]), float(prior[low_row, 1]))
        self.assertGreater(float(prior[high_row, 1]), float(prior[high_row, 0]))
        self.assertAlmostEqual(float(prior[low_row].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(prior[high_row].sum()), 1.0, places=6)

    def test_shuffle_feature_rows_within_region_preserves_region_membership(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from tools.exp_phase2_puma_to_small_area import _shuffle_feature_rows_within_region

        df = pd.DataFrame(
            {
                "tract_geoid": ["g1", "g2", "g3", "g4"],
                "puma_uid": ["p1", "p1", "p2", "p2"],
                "cat__a": [1.0, 2.0, 3.0, 4.0],
                "cat__b": [10.0, 20.0, 30.0, 40.0],
            }
        )
        out = _shuffle_feature_rows_within_region(df, group_col="tract_geoid", region_col="puma_uid", seed=7)
        self.assertEqual(out["tract_geoid"].tolist(), df["tract_geoid"].tolist())
        self.assertEqual(out["puma_uid"].tolist(), df["puma_uid"].tolist())
        p1_vals = set(map(tuple, out[out["puma_uid"] == "p1"][["cat__a", "cat__b"]].to_numpy().tolist()))
        p2_vals = set(map(tuple, out[out["puma_uid"] == "p2"][["cat__a", "cat__b"]].to_numpy().tolist()))
        self.assertEqual(p1_vals, {(1.0, 10.0), (2.0, 20.0)})
        self.assertEqual(p2_vals, {(3.0, 30.0), (4.0, 40.0)})


if __name__ == "__main__":
    unittest.main()

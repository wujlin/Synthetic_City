import unittest

import pandas as pd


class TestRoadLocationAllocationSmoke(unittest.TestCase):
    def _build_toy_inputs(self):
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import LineString, Polygon
        except Exception:
            self.skipTest("geopandas/shapely not installed")

        areas = gpd.GeoDataFrame(
            {
                "tract_geoid": ["t1", "t2"],
                "geometry": [
                    Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
                    Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "MTFCC": ["S1400", "S1200", "S1740"],
                "component": [1, 2, 3],
                "geometry": [
                    LineString([(0.1, 0.2), (0.9, 0.2)]),
                    LineString([(0.1, 0.8), (0.9, 0.8)]),
                    LineString([(2.1, 0.2), (2.9, 0.2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        persons = pd.DataFrame(
            {
                "person_id": ["p1", "p2", "p3", "p4"],
                "tract_geoid": ["t1", "t1", "t2", "t2"],
                "household_id": ["hh1", "hh1", "hh2", None],
                "is_worker": [True, False, True, False],
            }
        )
        return areas, roads, persons

    def test_build_candidates_and_assign_locations_no_fallback(self) -> None:
        from src.synthpop.spatial.road_location_allocation import (
            assign_home_work_locations,
            build_road_location_candidates,
        )

        areas, roads, persons = self._build_toy_inputs()
        home_candidates, work_candidates, meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="conservative",
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
        )
        self.assertEqual(meta["home_stage_counts"]["primary"], 1)
        self.assertEqual(meta["home_stage_counts"]["no_candidates"], 1)
        self.assertEqual(meta["work_stage_counts"]["primary"], 1)
        self.assertEqual(meta["work_stage_counts"]["no_candidates"], 1)

        t2_home_stages = home_candidates.loc[home_candidates["tract_geoid"] == "t2", "source_stage"].unique().tolist()
        t2_work_stages = work_candidates.loc[work_candidates["tract_geoid"] == "t2", "source_stage"].unique().tolist()
        self.assertEqual(t2_home_stages, [])
        self.assertEqual(t2_work_stages, [])

        assigned, assign_meta = assign_home_work_locations(
            persons=persons,
            home_candidates=home_candidates,
            work_candidates=work_candidates,
            group_col="tract_geoid",
            person_id_col="person_id",
            household_col="household_id",
            work_eligible_col="is_worker",
            seed=0,
        )

        hh1 = assigned.loc[assigned["household_id"] == "hh1", "home_candidate_id"].tolist()
        self.assertEqual(len(set(hh1)), 1)
        self.assertEqual(assign_meta["home_assignment_mode"], "household")

        p3 = assigned.loc[assigned["person_id"] == "p3"].iloc[0]
        self.assertIsNone(p3["home_candidate_id"])
        self.assertEqual(p3["home_source_stage"], "no_candidates")
        self.assertEqual(p3["home_assignment_mode"], "unassigned_no_candidates")
        self.assertIsNone(p3["work_candidate_id"])
        self.assertEqual(p3["work_source_stage"], "no_candidates")
        self.assertEqual(p3["work_assignment_mode"], "unassigned_no_candidates")

        p4 = assigned.loc[assigned["person_id"] == "p4"].iloc[0]
        self.assertIsNone(p4["home_candidate_id"])
        self.assertEqual(p4["home_assignment_mode"], "unassigned_no_candidates")
        self.assertIsNone(p4["work_candidate_id"])
        self.assertEqual(p4["work_assignment_mode"], "ineligible")

        self.assertEqual(int(assign_meta["home_unassigned"]), 2)
        self.assertEqual(int(assign_meta["home_fallback_assignments"]), 0)
        self.assertEqual(int(assign_meta["work_assigned"]), 1)
        self.assertEqual(int(assign_meta["work_unassigned"]), 1)
        self.assertEqual(int(assign_meta["work_fallback_assignments"]), 0)

    def test_build_candidates_and_assign_locations_with_optional_fallback(self) -> None:
        from src.synthpop.spatial.road_location_allocation import (
            assign_home_work_locations,
            build_road_location_candidates,
        )

        areas, roads, persons = self._build_toy_inputs()
        home_candidates, work_candidates, meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="conservative",
            allow_home_fallback=True,
            allow_work_fallback=True,
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
        )
        self.assertEqual(meta["home_stage_counts"]["primary"], 1)
        self.assertEqual(meta["home_stage_counts"]["compatibility_fallback"], 1)
        self.assertEqual(meta["work_stage_counts"]["primary"], 1)
        self.assertEqual(meta["work_stage_counts"]["home_intersection_fallback"], 1)

        t2_home_stages = home_candidates.loc[home_candidates["tract_geoid"] == "t2", "source_stage"].unique().tolist()
        t2_work_stages = work_candidates.loc[work_candidates["tract_geoid"] == "t2", "source_stage"].unique().tolist()
        self.assertEqual(t2_home_stages, ["compatibility_fallback"])
        self.assertEqual(t2_work_stages, ["home_intersection_fallback"])

        assigned, assign_meta = assign_home_work_locations(
            persons=persons,
            home_candidates=home_candidates,
            work_candidates=work_candidates,
            group_col="tract_geoid",
            person_id_col="person_id",
            household_col="household_id",
            work_eligible_col="is_worker",
            seed=0,
        )

        p3 = assigned.loc[assigned["person_id"] == "p3"].iloc[0]
        self.assertEqual(p3["home_source_stage"], "compatibility_fallback")
        self.assertEqual(p3["work_source_stage"], "home_intersection_fallback")
        self.assertTrue(bool(p3["work_fallback_flag"]))
        self.assertEqual(int(assign_meta["home_unassigned"]), 0)
        self.assertEqual(int(assign_meta["work_unassigned"]), 0)
        self.assertGreater(int(assign_meta["work_fallback_assignments"]), 0)

    def test_boundary_points_are_legalized_into_target_area(self) -> None:
        try:
            import geopandas as gpd
            from shapely.geometry import LineString, Polygon
        except Exception:
            self.skipTest("geopandas/shapely not installed")

        from src.synthpop.spatial.road_location_allocation import build_road_location_candidates

        areas = gpd.GeoDataFrame(
            {
                "tract_geoid": ["t1"],
                "geometry": [Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "MTFCC": ["S1200"],
                "component": [1],
                "geometry": [LineString([(1.0, 0.0), (1.0, 1.0)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        work_candidates = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="compatibility",
            work_mtfcc_values=["S1200"],
            work_interpolation_density=0.4,
            legalization_fraction=1e-3,
        )[1]

        self.assertGreater(int(work_candidates.shape[0]), 0)
        area = areas.iloc[0].geometry
        self.assertTrue(all(bool(area.covers(pt)) for pt in work_candidates.geometry.tolist()))

    def test_build_candidates_with_explicit_work_gap_exception(self) -> None:
        try:
            import geopandas as gpd
            import pandas as pd
            from shapely.geometry import LineString, Polygon
        except Exception:
            self.skipTest("geopandas/shapely not installed")

        from src.synthpop.spatial.road_location_allocation import (
            assign_home_work_locations,
            build_road_location_candidates,
        )

        areas = gpd.GeoDataFrame(
            {
                "tract_geoid": ["t1", "t2"],
                "geometry": [
                    Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
                    Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "MTFCC": ["S1400", "S1200", "S1400"],
                "component": [1, 2, 3],
                "geometry": [
                    LineString([(0.1, 0.2), (0.9, 0.2)]),
                    LineString([(0.1, 0.8), (0.9, 0.8)]),
                    LineString([(2.1, 0.2), (2.9, 0.2)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        persons = pd.DataFrame(
            {
                "person_id": ["p1", "p2"],
                "tract_geoid": ["t1", "t2"],
                "household_id": ["hh1", "hh2"],
                "is_worker": [True, True],
            }
        )

        home_candidates, work_candidates, meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="conservative",
            work_mtfcc_values=["S1100", "S1200"],
            work_gap_exception_mtfcc_values=["S1400"],
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
        )

        self.assertEqual(meta["work_allowed_non_primary_stages"], ["arterial_missing_exception"])
        self.assertEqual(meta["work_stage_counts"]["primary"], 1)
        self.assertEqual(meta["work_stage_counts"]["arterial_missing_exception"], 1)
        t2_work_stages = work_candidates.loc[work_candidates["tract_geoid"] == "t2", "source_stage"].unique().tolist()
        self.assertEqual(t2_work_stages, ["arterial_missing_exception"])

        assigned, assign_meta = assign_home_work_locations(
            persons=persons,
            home_candidates=home_candidates,
            work_candidates=work_candidates,
            group_col="tract_geoid",
            person_id_col="person_id",
            household_col="household_id",
            work_eligible_col="is_worker",
            seed=0,
        )

        p2 = assigned.loc[assigned["person_id"] == "p2"].iloc[0]
        self.assertEqual(p2["work_source_stage"], "arterial_missing_exception")
        self.assertFalse(bool(p2["work_fallback_flag"]))
        self.assertEqual(int(assign_meta["work_unassigned"]), 0)
        self.assertEqual(int(assign_meta["work_fallback_assignments"]), 0)

    def test_parallel_candidate_generation_matches_serial(self) -> None:
        from src.synthpop.spatial.road_location_allocation import build_road_location_candidates

        areas, roads, _ = self._build_toy_inputs()
        serial_home, serial_work, serial_meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="compatibility",
            work_mtfcc_values=["S1200"],
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
            n_jobs=1,
        )
        parallel_home, parallel_work, parallel_meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="compatibility",
            work_mtfcc_values=["S1200"],
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
            n_jobs=2,
            parallel_chunksize=1,
        )

        self.assertFalse(bool(serial_meta["parallel_used"]))
        self.assertTrue(bool(parallel_meta["parallel_used"]))
        self.assertEqual(int(serial_home.shape[0]), int(parallel_home.shape[0]))
        self.assertEqual(int(serial_work.shape[0]), int(parallel_work.shape[0]))
        self.assertEqual(
            serial_home.sort_values("candidate_id", kind="stable")["candidate_id"].tolist(),
            parallel_home.sort_values("candidate_id", kind="stable")["candidate_id"].tolist(),
        )
        self.assertEqual(
            serial_work.sort_values("candidate_id", kind="stable")["candidate_id"].tolist(),
            parallel_work.sort_values("candidate_id", kind="stable")["candidate_id"].tolist(),
        )

    def test_work_gap_exception_triggers_when_primary_lines_exist_but_yield_no_points(self) -> None:
        try:
            import geopandas as gpd
            from shapely.geometry import LineString, Polygon
        except Exception:
            self.skipTest("geopandas/shapely not installed")

        from src.synthpop.spatial.road_location_allocation import build_road_location_candidates

        areas = gpd.GeoDataFrame(
            {
                "tract_geoid": ["t1"],
                "geometry": [Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        roads = gpd.GeoDataFrame(
            {
                "MTFCC": ["S1200", "S1400"],
                "component": [1, 2],
                "geometry": [
                    LineString([(0.10, 0.10), (0.10005, 0.10)]),
                    LineString([(0.20, 0.20), (0.80, 0.20)]),
                ],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

        _, work_candidates, meta = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="conservative",
            work_mtfcc_values=["S1100", "S1200"],
            work_gap_exception_mtfcc_values=["S1400"],
            work_interpolation_density=0.2,
        )

        self.assertEqual(meta["work_stage_counts"]["arterial_missing_exception"], 1)
        self.assertNotIn("no_candidates", meta["work_stage_counts"])
        self.assertEqual(
            work_candidates.loc[work_candidates["tract_geoid"] == "t1", "source_stage"].unique().tolist(),
            ["arterial_missing_exception"],
        )
        self.assertGreater(int(work_candidates.shape[0]), 0)

    def test_assign_locations_respects_separate_work_group_col(self) -> None:
        from src.synthpop.spatial.road_location_allocation import (
            assign_home_work_locations,
            build_road_location_candidates,
        )

        areas, roads, persons = self._build_toy_inputs()
        roads = roads.copy()
        roads = pd.concat(
            [
                roads,
                roads.iloc[[1]].assign(
                    geometry=lambda d: d.geometry.translate(xoff=2.0, yoff=0.0),
                    MTFCC="S1200",
                    component=4,
                ),
            ],
            ignore_index=True,
        )
        persons = persons.copy()
        persons["work_tract_geoid"] = ["t2", "t1", "t1", "t2"]

        home_candidates, work_candidates, _ = build_road_location_candidates(
            areas=areas,
            roads=roads,
            group_col="tract_geoid",
            home_mode="compatibility",
            work_mtfcc_values=["S1200"],
            home_interpolation_density=0.2,
            work_interpolation_density=0.2,
        )

        assigned, meta = assign_home_work_locations(
            persons=persons,
            home_candidates=home_candidates,
            work_candidates=work_candidates,
            group_col="tract_geoid",
            work_group_col="work_tract_geoid",
            person_id_col="person_id",
            household_col="household_id",
            work_eligible_col="is_worker",
            seed=0,
        )

        p1 = assigned.loc[assigned["person_id"] == "p1"].iloc[0]
        p3 = assigned.loc[assigned["person_id"] == "p3"].iloc[0]
        self.assertEqual(meta["work_group_col"], "work_tract_geoid")
        self.assertEqual(str(p1["work_candidate_id"]).split(":")[0], "t2")
        self.assertEqual(str(p3["work_candidate_id"]).split(":")[0], "t1")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import pandas as pd

from src.synthpop.validation.mobility_anchor import (
    AnchorSpec,
    compare_share_frames,
    select_device_anchors,
    summarize_distance_distribution,
    within_tract_bg_spearman,
)


class MobilityAnchorValidationSmokeTest(unittest.TestCase):
    def test_select_device_anchors_prefers_long_night_and_far_day(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "ad_id": "d1",
                    "latitude": 42.30,
                    "longitude": -83.10,
                    "time_spent": 8 * 3600,
                    "start_time_local": "2019-02-05 21:00:00-05:00",
                    "end_time_local": "2019-02-06 05:00:00-05:00",
                },
                {
                    "ad_id": "d1",
                    "latitude": 42.35,
                    "longitude": -83.00,
                    "time_spent": 5 * 3600,
                    "start_time_local": "2019-02-06 10:00:00-05:00",
                    "end_time_local": "2019-02-06 15:00:00-05:00",
                },
                {
                    "ad_id": "d2",
                    "latitude": 42.31,
                    "longitude": -83.11,
                    "time_spent": 7 * 3600,
                    "start_time_local": "2019-02-05 22:00:00-05:00",
                    "end_time_local": "2019-02-06 05:00:00-05:00",
                },
                {
                    "ad_id": "d2",
                    "latitude": 42.3101,
                    "longitude": -83.1101,
                    "time_spent": 4 * 3600,
                    "start_time_local": "2019-02-06 11:00:00-05:00",
                    "end_time_local": "2019-02-06 15:00:00-05:00",
                },
            ]
        )
        home, work, summary = select_device_anchors(
            events,
            spec=AnchorSpec(min_home_secs=6 * 3600, min_work_secs=3 * 3600, min_home_work_distance_m=500.0),
        )
        self.assertEqual(len(home), 2)
        self.assertEqual(len(work), 1)
        self.assertEqual(summary["n_home_anchor_devices"], 2)
        self.assertEqual(summary["n_home_work_devices"], 1)
        self.assertEqual(work.iloc[0]["ad_id"], "d1")

    def test_compare_share_frames_and_bg_spearman(self) -> None:
        left = pd.DataFrame(
            {
                "tract_geoid": ["t1", "t2"],
                "synthetic_count": [80, 20],
            }
        )
        right = pd.DataFrame(
            {
                "tract_geoid": ["t1", "t2"],
                "mobility_count": [8, 2],
            }
        )
        _, summary = compare_share_frames(
            left=left,
            right=right,
            key_cols=["tract_geoid"],
            left_value_col="synthetic_count",
            right_value_col="mobility_count",
        )
        self.assertAlmostEqual(summary["tvd_share"], 0.0, places=8)
        self.assertAlmostEqual(summary["cosine_share"], 1.0, places=8)

        syn_bg = pd.DataFrame(
            {
                "tract_geoid": ["t1", "t1", "t2", "t2"],
                "bg_geoid": ["b1", "b2", "b3", "b4"],
                "count": [30, 10, 20, 5],
            }
        )
        mob_bg = pd.DataFrame(
            {
                "tract_geoid": ["t1", "t1", "t2", "t2"],
                "bg_geoid": ["b1", "b2", "b3", "b4"],
                "count": [3, 1, 2, 1],
            }
        )
        tract_df, tract_summary = within_tract_bg_spearman(
            synthetic_bg_counts=syn_bg,
            mobility_bg_counts=mob_bg,
            min_mobility_total=1,
        )
        self.assertEqual(len(tract_df), 2)
        self.assertEqual(tract_summary["n_tracts_with_valid_spearman"], 2)
        self.assertTrue(tract_df["spearman_bg"].dropna().ge(0.99).all())

    def test_distance_distribution_summary(self) -> None:
        table, summary = summarize_distance_distribution(
            synthetic_distance_m=pd.Series([1000.0, 3000.0, 10000.0]),
            mobility_distance_m=pd.Series([1200.0, 2800.0, 9000.0]),
        )
        self.assertEqual(int(table["synthetic_count"].sum()), 3)
        self.assertEqual(int(table["mobility_count"].sum()), 3)
        self.assertTrue(math.isfinite(summary["cosine_share"]))
        self.assertLessEqual(summary["tvd_share"], 1.0)


if __name__ == "__main__":
    unittest.main()

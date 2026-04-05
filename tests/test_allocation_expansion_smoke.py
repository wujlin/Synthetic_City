import unittest


class TestAllocationExpansionSmoke(unittest.TestCase):
    def test_integerize_and_expand_allocation(self) -> None:
        try:
            import pandas as pd
        except Exception:
            self.skipTest("pandas not installed")

        from src.synthpop.spatial.allocation_expansion import (
            expand_integer_allocation_to_persons,
            integerize_type_allocation_long,
        )

        alloc = pd.DataFrame(
            {
                "puma_uid": ["r1", "r1", "r1", "r1"],
                "tract_geoid": ["g1", "g2", "g1", "g2"],
                "type_idx": [0, 0, 1, 1],
                "AGEP_bin": ["young", "young", "old", "old"],
                "SEX": ["1", "1", "2", "2"],
                "ESR_allpop": ["employed", "employed", "not_16p", "not_16p"],
                "count": [1.4, 2.6, 1.6, 0.4],
            }
        )

        alloc_int, meta = integerize_type_allocation_long(
            allocation_long=alloc,
            region_col="puma_uid",
            group_col="tract_geoid",
            type_idx_col="type_idx",
            count_col="count",
            out_count_col="count_int",
        )
        self.assertEqual(meta["n_regions"], 1)
        self.assertEqual(int(meta["total_after"]), 6)

        by_type = alloc_int.groupby("type_idx", as_index=False)["count_int"].sum().sort_values("type_idx")
        self.assertEqual(by_type["count_int"].tolist(), [4, 2])
        by_group = alloc_int.groupby("tract_geoid", as_index=False)["count_int"].sum().sort_values("tract_geoid")
        self.assertEqual(by_group["count_int"].tolist(), [3, 3])

        persons, person_meta = expand_integer_allocation_to_persons(
            integer_allocation_long=alloc_int,
            count_col="count_int",
            person_id_col="person_id",
            person_id_prefix="toy",
            esr_col="ESR_allpop",
        )
        self.assertEqual(int(person_meta["n_persons"]), 6)
        self.assertEqual(int(person_meta["n_unique_person_ids"]), 6)
        self.assertEqual(len(persons), 6)
        self.assertTrue(persons["person_id"].str.startswith("toy_").all())
        self.assertEqual(int(persons["is_worker"].sum()), 4)


if __name__ == "__main__":
    unittest.main()

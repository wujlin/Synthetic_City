import unittest


class TestTrainingPairsSmoke(unittest.TestCase):
    def test_build_training_pairs_shapes(self) -> None:
        try:
            import numpy as np
            import pandas as pd
        except Exception:
            self.skipTest("numpy/pandas not installed")

        from src.synthpop.alignment.training_pairs import build_training_pairs

        persons = pd.DataFrame({"person_id": [1, 2, 3], "cbg_geoid": ["a", "a", "b"]})
        devices = pd.DataFrame({"CAID": ["d1", "d2"], "CENSUS_BLOCK_GROUP": ["a", "b"]})
        buildings = pd.DataFrame({"bldg_id": ["x", "y", "z"], "cbg_geoid": ["a", "a", "b"]})

        z_persons = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0]], dtype=float)
        z_devices = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=float)
        z_buildings = np.array([[0.0, 0.0], [0.2, 0.2], [10.0, 10.0]], dtype=float)

        zpo, zbo, gids = build_training_pairs(
            persons=persons,
            devices=devices,
            buildings=buildings,
            z_persons=z_persons,
            z_devices=z_devices,
            z_buildings=z_buildings,
            k_soft_labels=3,
            seed=0,
        )

        self.assertEqual(zpo.shape, (len(persons) * 3, 2))
        self.assertEqual(zbo.shape, (len(persons) * 3, 2))
        self.assertEqual(gids.shape, (len(persons) * 3,))


if __name__ == "__main__":
    unittest.main()


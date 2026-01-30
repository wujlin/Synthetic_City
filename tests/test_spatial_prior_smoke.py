import unittest


class TestSpatialPriorSmoke(unittest.TestCase):
    def test_activity_center_loss(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        from src.synthpop.alignment.spatial_prior import activity_center_loss

        a = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        b = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
        loss = activity_center_loss(activity_centers=a, building_locations=b, reduction="mean")
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreaterEqual(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()


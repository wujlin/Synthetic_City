import unittest


class TestAlignmentLossesSmoke(unittest.TestCase):
    def test_alignment_loss_finite(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        from src.synthpop.encoders.shared_latent import SharedLatentSpace, SharedLatentSpaceSpec

        torch.manual_seed(0)
        sls = SharedLatentSpace(person_input_dim=2, device_input_dim=2, building_input_dim=2, spec=SharedLatentSpaceSpec(latent_dim=2, hidden_dims=(8,)))

        # Fake embeddings already in latent space (we are testing loss composition, not encoder outputs).
        z_persons = torch.randn(20, 2)
        z_devices = torch.randn(10, 2)
        z_buildings = torch.randn(15, 2)

        device_groups = ["a"] * 5 + ["b"] * 5
        building_groups = ["a"] * 7 + ["b"] * 8
        person_groups = ["a"] * 10 + ["b"] * 10

        loss = sls.alignment_loss(
            z_persons=z_persons,
            z_devices=z_devices,
            z_buildings=z_buildings,
            device_cbg_ids=device_groups,
            building_cbg_ids=building_groups,
            person_cbg_ids=person_groups,
            weights={"contrastive": 1.0, "mmd": 0.1, "spatial": 0.0},
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()


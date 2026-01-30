import unittest


class TestJointDiffusionSmoke(unittest.TestCase):
    def test_fit_and_sample(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        from src.synthpop.model.diffusion_tabular import TabDDPMConfig
        from src.synthpop.model.joint_diffusion import JointDiffusionConfig, JointDiffusionModel

        torch.manual_seed(0)
        z_person = torch.randn(64, 4)
        z_building = torch.randn(64, 4)

        cfg = JointDiffusionConfig(latent_dim=4, cond_dim=0, seed=0, tabddpm=TabDDPMConfig(timesteps=5, hidden_dims=(32, 32)))
        model = JointDiffusionModel(config=cfg)
        metrics = model.fit(z_person=z_person, z_building=z_building, epochs=1, batch_size=16, device="cpu", log_every=0)
        self.assertIn("loss", metrics)

        zp, zb = model.sample(n=10, device="cpu")
        self.assertEqual(tuple(zp.shape), (10, 4))
        self.assertEqual(tuple(zb.shape), (10, 4))


if __name__ == "__main__":
    unittest.main()


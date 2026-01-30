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

        # Guidance (smoke): simple continuous histogram on the first dimension.
        target_marginals = {
            "x0_dim0": {"type": "continuous_hist", "index": 0, "bins": [-1.0, 0.0, 1.0], "target": [0.2, 0.6, 0.2], "sigma": 1.0}
        }
        zp2, zb2 = model.sample(
            n=10,
            device="cpu",
            target_marginals=target_marginals,
            guidance_scale=0.05,
            guidance_schedule="linear",
        )
        self.assertEqual(tuple(zp2.shape), (10, 4))
        self.assertEqual(tuple(zb2.shape), (10, 4))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest


class TestDiffusionTabularSmoke(unittest.TestCase):
    def test_fit_sample_save_load(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig

        torch.manual_seed(0)
        x = torch.randn(64, 8)
        cond = torch.randn(64, 4)

        cfg = TabDDPMConfig(timesteps=10, hidden_dims=(32, 32), time_embed_dim=32, lr=1e-3)
        model = DiffusionTabularModel(input_dim=8, cond_dim=4, seed=0, config=cfg)
        metrics = model.fit(x=x, cond=cond, epochs=1, batch_size=16, device="cpu", log_every=0)
        self.assertIn("loss", metrics)

        out = model.sample(n=10, cond=cond[:10], device="cpu")
        self.assertEqual(tuple(out.shape), (10, 8))
        self.assertTrue(torch.isfinite(out).all().item())

        with tempfile.TemporaryDirectory() as td:
            ckpt = tempfile.NamedTemporaryFile(dir=td, suffix=".pt", delete=False).name
            from pathlib import Path

            model.save(Path(ckpt))

            model2 = DiffusionTabularModel(input_dim=1, cond_dim=0, seed=123)
            model2.load(Path(ckpt))
            out2 = model2.sample(n=10, cond=cond[:10], device="cpu")
            self.assertEqual(tuple(out2.shape), (10, 8))


if __name__ == "__main__":
    unittest.main()


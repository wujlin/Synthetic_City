import unittest


class TestSharedLatentSmoke(unittest.TestCase):
    def test_encode_shapes(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("torch not installed")

        from src.synthpop.encoders.shared_latent import SharedLatentSpace, SharedLatentSpaceSpec

        spec = SharedLatentSpaceSpec(latent_dim=8, hidden_dims=(16,))
        sls = SharedLatentSpace(person_input_dim=5, device_input_dim=7, building_input_dim=6, spec=spec).to("cpu")

        zp = sls.encode_person(torch.randn(4, 5))
        zd = sls.encode_device(torch.randn(4, 7))
        zb = sls.encode_building(torch.randn(4, 6))

        self.assertEqual(tuple(zp.shape), (4, 8))
        self.assertEqual(tuple(zd.shape), (4, 8))
        self.assertEqual(tuple(zb.shape), (4, 8))


if __name__ == "__main__":
    unittest.main()


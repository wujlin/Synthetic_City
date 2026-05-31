from __future__ import annotations

"""
Tabular diffusion model (v0): TabDDPM-style Gaussian DDPM on mixed-type features.

Design choices (KISS):
- Operate in a continuous vector space.
- Continuous columns are standardized (handled outside this module).
- Categorical columns can be one-hot (or embedded) and concatenated with continuous columns.
- Optional conditioning is supported via concatenating a condition vector.

This is intended for a minimal proof-of-concept on PUMS subsets first; more advanced
discrete diffusion / guidance methods can be layered later without changing the pipeline contracts.
"""

import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This module requires PyTorch. Install with conda/pip (CUDA if available).") from e
    return torch


@dataclass(frozen=True)
class TabDDPMConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dims: tuple[int, ...] = (256, 256)
    time_embed_dim: int = 128
    condition_injection: str = "concat"  # concat | film
    film_hidden_dim: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0


def _torch_module_base() -> type:
    try:
        import torch  # type: ignore
    except Exception:
        return object
    return torch.nn.Module


def _sinusoidal_time_embedding(t: Any, *, dim: int) -> Any:
    torch = _require_torch()
    if dim % 2 != 0:
        raise ValueError("time_embed_dim must be even")
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1)))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class _DenoiserMLP(_torch_module_base()):
    def __init__(self, *, input_dim: int, cond_dim: int, hidden_dims: tuple[int, ...], time_embed_dim: int) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]

        layers: list[Any] = []
        dim_in = input_dim + cond_dim + time_embed_dim
        for dim_out in hidden_dims:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.SiLU())
            dim_in = dim_out
        layers.append(nn.Linear(dim_in, input_dim))

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Any, t: Any, cond: Any | None) -> Any:  # type: ignore[override]
        torch = _require_torch()
        t_emb = _sinusoidal_time_embedding(t, dim=self.time_embed_dim)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            inp = torch.cat([x_t, cond, t_emb], dim=1)
        else:
            inp = torch.cat([x_t, t_emb], dim=1)
        return self.net(inp)


class _FiLMDenoiserMLP(_torch_module_base()):
    """
    FiLM conditioning:
    - t embedding is concatenated with x_t at input.
    - cond is mapped to per-layer (gamma, beta), modulating each hidden layer:
        h_l <- (1 + gamma_l) * h_l + beta_l
    """

    def __init__(
        self,
        *,
        input_dim: int,
        cond_dim: int,
        hidden_dims: tuple[int, ...],
        time_embed_dim: int,
        film_hidden_dim: int = 128,
    ) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be non-empty for FiLM denoiser")

        self.input_dim = int(input_dim)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)

        self.hidden_layers = nn.ModuleList()
        dim_in = self.input_dim + self.time_embed_dim
        for dim_out in self.hidden_dims:
            self.hidden_layers.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out
        self.out = nn.Linear(dim_in, self.input_dim)
        self.act = nn.SiLU()

        self._film_slices: list[tuple[int, int]] = []
        if self.cond_dim > 0:
            total = int(sum(self.hidden_dims))
            self.cond_net = nn.Sequential(
                nn.Linear(self.cond_dim, int(film_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(film_hidden_dim), int(film_hidden_dim)),
                nn.SiLU(),
                nn.Linear(int(film_hidden_dim), 2 * total),
            )
            start = 0
            for d in self.hidden_dims:
                self._film_slices.append((start, start + int(d)))
                start += int(d)
        else:
            self.cond_net = None

    def forward(self, x_t: Any, t: Any, cond: Any | None) -> Any:  # type: ignore[override]
        torch = _require_torch()
        t_emb = _sinusoidal_time_embedding(t, dim=self.time_embed_dim)
        h = torch.cat([x_t, t_emb], dim=1)

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            assert self.cond_net is not None
            film = self.cond_net(cond)
            split = film.shape[1] // 2
            gamma_all = film[:, :split]
            beta_all = film[:, split:]
        else:
            gamma_all = None
            beta_all = None

        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            if gamma_all is not None and beta_all is not None:
                s0, s1 = self._film_slices[i]
                gamma = gamma_all[:, s0:s1]
                beta = beta_all[:, s0:s1]
                # Keep modulation numerically stable around identity.
                h = (1.0 + 0.1 * gamma) * h + 0.1 * beta
            h = self.act(h)
        return self.out(h)


class DiffusionTabularModel:
    def __init__(self, *, input_dim: int, cond_dim: int = 0, seed: int = 0, config: TabDDPMConfig | None = None) -> None:
        self.input_dim = int(input_dim)
        self.cond_dim = int(cond_dim)
        self.seed = int(seed)
        self.config = config or TabDDPMConfig()

        self._net: Any | None = None
        self._schedule: dict[str, Any] | None = None

    def _init_model(self, *, device: Any) -> None:
        torch = _require_torch()
        if self._net is None:
            mode = str(getattr(self.config, "condition_injection", "concat")).lower().strip()
            if mode == "concat":
                self._net = _DenoiserMLP(
                    input_dim=self.input_dim,
                    cond_dim=self.cond_dim,
                    hidden_dims=self.config.hidden_dims,
                    time_embed_dim=self.config.time_embed_dim,
                )
            elif mode == "film":
                self._net = _FiLMDenoiserMLP(
                    input_dim=self.input_dim,
                    cond_dim=self.cond_dim,
                    hidden_dims=self.config.hidden_dims,
                    time_embed_dim=self.config.time_embed_dim,
                    film_hidden_dim=int(getattr(self.config, "film_hidden_dim", 128)),
                )
            else:
                raise ValueError("TabDDPMConfig.condition_injection must be one of: concat, film")
        self._net.to(device)

        if self._schedule is None:
            betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps, device=device)
            alphas = 1.0 - betas
            alpha_cumprod = torch.cumprod(alphas, dim=0)
            alpha_cumprod_prev = torch.cat([torch.ones(1, device=device), alpha_cumprod[:-1]], dim=0)
            posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            self._schedule = {
                "betas": betas,
                "alphas": alphas,
                "alpha_cumprod": alpha_cumprod,
                "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
                "sqrt_one_minus_alpha_cumprod": torch.sqrt(1.0 - alpha_cumprod),
                "alpha_cumprod_prev": alpha_cumprod_prev,
                "posterior_variance": posterior_variance,
            }

    def save(self, path: pathlib.Path) -> None:
        """
        Save model weights + config to disk.
        """
        torch = _require_torch()
        if self._net is None:
            raise RuntimeError("Model is not initialized. Train (fit) or load a checkpoint first.")
        p = pathlib.Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "synthpop.tabddpm.v0",
            "input_dim": self.input_dim,
            "cond_dim": self.cond_dim,
            "seed": self.seed,
            "config": asdict(self.config),
            "state_dict": self._net.state_dict(),
        }
        torch.save(payload, p)

    def load(self, path: pathlib.Path) -> None:
        """
        Load model weights from disk.
        """
        torch = _require_torch()
        p = pathlib.Path(path).expanduser().resolve()
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != "synthpop.tabddpm.v0":
            raise ValueError(f"Unsupported checkpoint format: {p}")

        self.input_dim = int(payload["input_dim"])
        self.cond_dim = int(payload.get("cond_dim", 0))
        self.seed = int(payload.get("seed", 0))
        self.config = TabDDPMConfig(**dict(payload.get("config", {})))

        mode = str(getattr(self.config, "condition_injection", "concat")).lower().strip()
        if mode == "concat":
            self._net = _DenoiserMLP(
                input_dim=self.input_dim,
                cond_dim=self.cond_dim,
                hidden_dims=self.config.hidden_dims,
                time_embed_dim=self.config.time_embed_dim,
            )
        elif mode == "film":
            self._net = _FiLMDenoiserMLP(
                input_dim=self.input_dim,
                cond_dim=self.cond_dim,
                hidden_dims=self.config.hidden_dims,
                time_embed_dim=self.config.time_embed_dim,
                film_hidden_dim=int(getattr(self.config, "film_hidden_dim", 128)),
            )
        else:
            raise ValueError("TabDDPMConfig.condition_injection must be one of: concat, film")
        self._net.load_state_dict(payload["state_dict"])
        self._schedule = None

    def fit(
        self,
        *,
        x: Any,
        cond: Any | None = None,
        epochs: int = 5,
        batch_size: int = 2048,
        device: str | None = None,
        log_every: int = 200,
        epoch_callback: Any | None = None,
    ) -> dict[str, float]:
        torch = _require_torch()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)

        x = x.to(device=device, dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be (N,{self.input_dim}), got {tuple(x.shape)}")

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim or cond.shape[0] != x.shape[0]:
                raise ValueError(f"cond must be (N,{self.cond_dim}), got {tuple(cond.shape)}")
        else:
            cond = None

        self._init_model(device=device)
        assert self._net is not None
        assert self._schedule is not None
        self._net.train()

        optim = torch.optim.AdamW(self._net.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        mse = torch.nn.MSELoss()

        num_rows = x.shape[0]
        num_steps = 0
        last_loss = float("nan")

        for _epoch in range(int(epochs)):
            indices = torch.randperm(num_rows, device=device)
            for start in range(0, num_rows, int(batch_size)):
                batch_idx = indices[start : start + int(batch_size)]
                batch_x0 = x[batch_idx]
                batch_cond = cond[batch_idx] if cond is not None else None

                t = torch.randint(0, self.config.timesteps, (batch_x0.shape[0],), device=device)
                noise = torch.randn_like(batch_x0)
                sqrt_acp = self._schedule["sqrt_alpha_cumprod"][t].unsqueeze(1)
                sqrt_om = self._schedule["sqrt_one_minus_alpha_cumprod"][t].unsqueeze(1)
                x_t = sqrt_acp * batch_x0 + sqrt_om * noise

                eps_pred = self._net(x_t, t, batch_cond)
                loss = mse(eps_pred, noise)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.config.grad_clip)
                optim.step()

                last_loss = float(loss.detach().cpu().item())
                num_steps += 1
                if log_every > 0 and num_steps % int(log_every) == 0:
                    print(f"[train] step={num_steps} loss={last_loss:.6f}")

            if epoch_callback is not None:
                epoch_callback(
                    int(_epoch + 1),
                    {
                        "loss": float(last_loss),
                        "num_steps": int(num_steps),
                    },
                )

        return {"loss": last_loss}

    def sample(
        self,
        *,
        n: int,
        cond: Any | None = None,
        device: str | None = None,
        sampler: str = "ddpm",
        num_steps: int | None = None,
        eta: float = 0.0,
    ) -> Any:
        torch = _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        sampler = str(sampler).lower().strip()
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError("sampler must be one of: ddpm, ddim")
        if num_steps is not None and int(num_steps) <= 0:
            raise ValueError("num_steps must be > 0 when provided")
        if float(eta) < 0:
            raise ValueError("eta must be >= 0")

        self._init_model(device=device)
        assert self._net is not None
        assert self._schedule is not None
        self._net.eval()

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim != 2 or cond.shape[1] != self.cond_dim or cond.shape[0] != int(n):
                raise ValueError(f"cond must be (N,{self.cond_dim}) where N==n, got {tuple(cond.shape)}")
        else:
            cond = None

        # Sampling does not require gradients. Without no_grad/inference_mode, the graph across
        # timesteps will accumulate and can easily OOM on GPU.
        with torch.inference_mode():
            x_t = torch.randn((int(n), self.input_dim), device=device)
            betas = self._schedule["betas"]
            alphas = self._schedule["alphas"]
            alpha_cumprod = self._schedule["alpha_cumprod"]
            posterior_variance = self._schedule["posterior_variance"]
            sqrt_one_minus_alpha_cumprod = self._schedule["sqrt_one_minus_alpha_cumprod"]
            timesteps = int(self.config.timesteps)
            if num_steps is None:
                num_steps = timesteps
            num_steps = int(num_steps)
            if num_steps > timesteps:
                raise ValueError(f"num_steps ({num_steps}) cannot exceed trained timesteps ({timesteps})")
            if num_steps == timesteps:
                step_schedule = list(reversed(range(timesteps)))
            else:
                # Uniformly subsample denoising steps from [T-1 .. 0].
                idx = torch.linspace(float(timesteps - 1), 0.0, steps=int(num_steps), device=device)
                step_schedule = idx.round().to(dtype=torch.long).tolist()
                # Remove accidental duplicates while preserving order.
                seen: set[int] = set()
                tmp: list[int] = []
                for s in step_schedule:
                    si = int(s)
                    if si in seen:
                        continue
                    seen.add(si)
                    tmp.append(si)
                step_schedule = tmp
            for i, step in enumerate(step_schedule):
                step = int(step)
                t = torch.full((int(n),), step, device=device, dtype=torch.long)
                eps_pred = self._net(x_t, t, cond)

                # Standard DDPM update for the full-step schedule.
                if sampler == "ddpm" and len(step_schedule) == timesteps:
                    beta_t = betas[step]
                    alpha_t = alphas[step]
                    sqrt_om = sqrt_one_minus_alpha_cumprod[step]
                    model_mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_om) * eps_pred)
                    if step == 0:
                        x_t = model_mean
                        continue
                    var_t = posterior_variance[step]
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + torch.sqrt(var_t) * noise
                    continue

                # Generalized DDIM-style update (supports skip-step and deterministic sampling).
                prev_step = int(step_schedule[i + 1]) if i + 1 < len(step_schedule) else -1
                alpha_bar_t = alpha_cumprod[step]
                if prev_step >= 0:
                    alpha_bar_prev = alpha_cumprod[prev_step]
                else:
                    alpha_bar_prev = torch.tensor(1.0, device=device, dtype=alpha_bar_t.dtype)

                x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / (torch.sqrt(alpha_bar_t) + 1e-12)
                sigma_t = float(eta) * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(
                    1.0 - alpha_bar_t / (alpha_bar_prev + 1e-12)
                )
                c_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t * sigma_t, min=0.0))
                if prev_step < 0:
                    noise = torch.zeros_like(x_t)
                else:
                    noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_prev) * x0_pred + c_t * eps_pred + sigma_t * noise

            return x_t.detach().cpu()

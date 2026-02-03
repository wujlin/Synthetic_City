from __future__ import annotations

"""
Categorical diffusion model (v0): multinomial / discrete-state diffusion for a single categorical variable.

Why this exists:
- Gaussian DDPM on continuous surrogates (e.g. age_u) can break discrete semantics and distort marginals.
- For purely categorical targets (like 46-way age×sex joint bins), a discrete diffusion is the correct tool.

Noise process (KISS, D3PM-style uniform corruption):
  q(x_t | x_{t-1}) = (1 - beta_t) * I[x_t = x_{t-1}] + beta_t * Uniform(K)

Training objective:
- Sample t ~ Uniform({1..T})
- Sample x_t from q(x_t | x_0)
- Predict p_theta(x_0 | x_t, t, cond) with a network outputting logits over K classes
- Cross-entropy loss on x_0

Sampling:
- Start from x_T ~ Uniform(K)
- For t = T..1:
    p_theta(x_0 | x_t)  ->  p_theta(x_{t-1} | x_t) = sum_{x0} q(x_{t-1} | x_t, x0) p_theta(x0 | x_t)
    sample x_{t-1} from that distribution
"""

import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Literal


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This module requires PyTorch. Install with conda/pip (CUDA if available).") from e
    return torch


def _sinusoidal_time_embedding(t: Any, *, dim: int) -> Any:
    torch = _require_torch()
    if dim % 2 != 0:
        raise ValueError("time_embed_dim must be even")
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1)))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


@dataclass(frozen=True)
class CategoricalDiffusionConfig:
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 256
    hidden_mlp: tuple[int, ...] = (256, 256)
    time_embed_dim: int = 128
    n_heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0


def _torch_module_base() -> type:
    try:
        import torch  # type: ignore
    except Exception:
        return object
    return torch.nn.Module


class _CondTokenEmbedder(_torch_module_base()):
    """
    Convert a cond vector (B,D) into D tokens (B,D,H) via per-dimension affine maps:
      token_j = cond_j * W_j + b_j
    """

    def __init__(self, *, cond_dim: int, hidden_dim: int) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.weight = nn.Parameter(torch.randn(self.cond_dim, self.hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.cond_dim, self.hidden_dim))

    def forward(self, cond: Any) -> Any:  # type: ignore[override]
        torch = _require_torch()
        c = cond.to(dtype=torch.float32)
        if c.ndim != 2 or c.shape[1] != int(self.cond_dim):
            raise ValueError(f"cond must be (B,{self.cond_dim}), got {tuple(c.shape)}")
        # (B,D,1) * (D,H) -> (B,D,H)
        return c.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _CatDenoiserConcat(_torch_module_base()):
    def __init__(self, *, n_classes: int, cond_dim: int, config: CategoricalDiffusionConfig) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]

        self.n_classes = int(n_classes)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(config.time_embed_dim)

        dim_in = self.n_classes + self.cond_dim + self.time_embed_dim
        layers: list[Any] = []
        for h in config.hidden_mlp:
            layers.append(nn.Linear(dim_in, int(h)))
            layers.append(nn.SiLU())
            dim_in = int(h)
        layers.append(nn.Linear(dim_in, self.n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, *, x_t: Any, t: Any, cond: Any | None) -> Any:  # type: ignore[override]
        torch = _require_torch()
        if self.cond_dim > 0 and cond is None:
            raise ValueError("cond is required when cond_dim>0")
        t_emb = _sinusoidal_time_embedding(t, dim=self.time_embed_dim)
        x_oh = torch.nn.functional.one_hot(x_t.to(dtype=torch.long), num_classes=self.n_classes).to(dtype=torch.float32)
        if self.cond_dim > 0:
            inp = torch.cat([x_oh, cond.to(dtype=torch.float32), t_emb], dim=1)
        else:
            inp = torch.cat([x_oh, t_emb], dim=1)
        return self.net(inp)


class _CatDenoiserCrossAttn(_torch_module_base()):
    def __init__(self, *, n_classes: int, cond_dim: int, config: CategoricalDiffusionConfig) -> None:
        torch = _require_torch()
        nn = torch.nn
        if hasattr(super(), "__init__"):
            super().__init__()  # type: ignore[misc]

        if int(cond_dim) <= 0:
            raise ValueError("cross-attn fusion requires cond_dim>0")

        self.n_classes = int(n_classes)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.time_embed_dim = int(config.time_embed_dim)

        self.q_proj = nn.Linear(self.n_classes + self.time_embed_dim, self.hidden_dim)
        self.cond_embed = _CondTokenEmbedder(cond_dim=self.cond_dim, hidden_dim=self.hidden_dim)
        self.attn = nn.MultiheadAttention(self.hidden_dim, int(config.n_heads), batch_first=True)

        layers: list[Any] = []
        dim_in = self.hidden_dim
        for h in config.hidden_mlp:
            layers.append(nn.Linear(dim_in, int(h)))
            layers.append(nn.SiLU())
            dim_in = int(h)
        layers.append(nn.Linear(dim_in, self.n_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, *, x_t: Any, t: Any, cond: Any | None) -> Any:  # type: ignore[override]
        torch = _require_torch()
        if cond is None:
            raise ValueError("cond is required for cross-attn fusion")
        t_emb = _sinusoidal_time_embedding(t, dim=self.time_embed_dim)
        x_oh = torch.nn.functional.one_hot(x_t.to(dtype=torch.long), num_classes=self.n_classes).to(dtype=torch.float32)
        q = self.q_proj(torch.cat([x_oh, t_emb], dim=1)).unsqueeze(1)  # (B,1,H)
        kv = self.cond_embed(cond).to(dtype=torch.float32)  # (B,D,H)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = (q + attn_out).squeeze(1)
        return self.mlp(h)


class CategoricalDiffusionModel:
    def __init__(
        self,
        *,
        n_classes: int,
        cond_dim: int = 0,
        cond_fusion: Literal["concat", "cross_attn"] = "concat",
        seed: int = 0,
        config: CategoricalDiffusionConfig | None = None,
    ) -> None:
        self.n_classes = int(n_classes)
        self.cond_dim = int(cond_dim)
        self.cond_fusion = str(cond_fusion)
        self.seed = int(seed)
        self.config = config or CategoricalDiffusionConfig()

        self._net: Any | None = None
        self._schedule: dict[str, Any] | None = None

    def _init_model(self, *, device: Any) -> None:
        torch = _require_torch()
        if self._net is None:
            if self.cond_fusion == "cross_attn":
                self._net = _CatDenoiserCrossAttn(n_classes=self.n_classes, cond_dim=self.cond_dim, config=self.config)
            else:
                self._net = _CatDenoiserConcat(n_classes=self.n_classes, cond_dim=self.cond_dim, config=self.config)
        self._net.to(device)

        if self._schedule is None:
            # timesteps are indexed 1..T; we store betas for steps 1..T in a length-T tensor.
            betas = torch.linspace(self.config.beta_start, self.config.beta_end, self.config.timesteps, device=device).clamp(min=0.0, max=0.999)
            alphas = 1.0 - betas
            alpha_bar = torch.cat([torch.ones(1, device=device), torch.cumprod(alphas, dim=0)], dim=0)  # (T+1,)
            self._schedule = {"betas": betas, "alphas": alphas, "alpha_bar": alpha_bar}

    def save(self, path: pathlib.Path) -> None:
        torch = _require_torch()
        if self._net is None:
            raise RuntimeError("Model is not initialized. Train (fit) or load a checkpoint first.")
        p = pathlib.Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "synthpop.catdiff.v0",
            "n_classes": self.n_classes,
            "cond_dim": self.cond_dim,
            "cond_fusion": self.cond_fusion,
            "seed": self.seed,
            "config": asdict(self.config),
            "state_dict": self._net.state_dict(),
        }
        torch.save(payload, p)

    def load(self, path: pathlib.Path) -> None:
        torch = _require_torch()
        p = pathlib.Path(path).expanduser().resolve()
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != "synthpop.catdiff.v0":
            raise ValueError(f"Unsupported checkpoint format: {p}")
        self.n_classes = int(payload["n_classes"])
        self.cond_dim = int(payload.get("cond_dim", 0))
        self.cond_fusion = str(payload.get("cond_fusion", "concat"))
        self.seed = int(payload.get("seed", 0))
        self.config = CategoricalDiffusionConfig(**dict(payload.get("config", {})))

        self._net = None
        self._schedule = None
        self._init_model(device="cpu")
        assert self._net is not None
        self._net.load_state_dict(payload["state_dict"])

    def _noise_x(self, *, x0: Any, t: Any) -> Any:
        """
        Sample x_t from q(x_t | x_0), using:
          q(x_t | x0) = alpha_bar[t] * one_hot(x0) + (1 - alpha_bar[t]) * Uniform(K)
        """
        torch = _require_torch()
        assert self._schedule is not None
        ab = self._schedule["alpha_bar"]  # (T+1,)
        a = ab.gather(0, t.to(dtype=torch.long).clamp(min=0, max=ab.shape[0] - 1))
        # a: (B,)
        keep = torch.rand_like(a) < a
        x_rand = torch.randint(0, self.n_classes, x0.shape, device=x0.device, dtype=torch.long)
        return torch.where(keep, x0.to(dtype=torch.long), x_rand)

    def fit(
        self,
        *,
        x0: Any,
        cond: Any | None = None,
        epochs: int = 5,
        batch_size: int = 2048,
        device: str | None = None,
        log_every: int = 200,
    ) -> dict[str, float]:
        torch = _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self._init_model(device=device)
        assert self._net is not None
        assert self._schedule is not None
        self._net.train()

        x0 = x0.to(device=device, dtype=torch.long)
        if x0.ndim != 1:
            raise ValueError(f"x0 must be (N,), got {tuple(x0.shape)}")
        n = int(x0.shape[0])

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim != 2 or cond.shape[0] != n or cond.shape[1] != self.cond_dim:
                raise ValueError(f"cond must be (N,{self.cond_dim}), got {tuple(cond.shape)}")
        else:
            cond = None

        optim = torch.optim.Adam(self._net.parameters(), lr=float(self.config.lr), weight_decay=float(self.config.weight_decay))
        loss_fn = torch.nn.CrossEntropyLoss()

        num_steps = 0
        last_loss = float("nan")
        T = int(self.config.timesteps)

        for _ in range(int(epochs)):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, int(batch_size)):
                batch = idx[start : start + int(batch_size)]
                y0 = x0[batch]
                # Sample t in {1..T}.
                t = torch.randint(1, T + 1, (int(y0.shape[0]),), device=device, dtype=torch.long)
                y_t = self._noise_x(x0=y0, t=t)

                logits = self._net(x_t=y_t, t=t, cond=cond[batch] if cond is not None else None)
                loss = loss_fn(logits, y0)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), float(self.config.grad_clip))
                optim.step()

                last_loss = float(loss.detach().cpu().item())
                num_steps += 1
                if log_every > 0 and num_steps % int(log_every) == 0:
                    print(f"[train] step={num_steps} loss={last_loss:.6f}")

        return {"loss": float(last_loss)}

    def sample(self, *, n: int, cond: Any | None = None, device: str | None = None) -> Any:
        torch = _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._init_model(device=device)
        assert self._net is not None
        assert self._schedule is not None
        self._net.eval()

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when cond_dim>0")
            cond = cond.to(device=device, dtype=torch.float32)
            if cond.ndim != 2 or cond.shape[0] != int(n) or cond.shape[1] != self.cond_dim:
                raise ValueError(f"cond must be (N,{self.cond_dim}) where N==n, got {tuple(cond.shape)}")
        else:
            cond = None

        K = int(self.n_classes)
        T = int(self.config.timesteps)
        betas = self._schedule["betas"]  # (T,)
        alpha_bar = self._schedule["alpha_bar"]  # (T+1,)

        with torch.inference_mode():
            # x_T ~ Uniform(K)
            x_t = torch.randint(0, K, (int(n),), device=device, dtype=torch.long)
            for t in range(T, 0, -1):
                t_vec = torch.full((int(n),), t, device=device, dtype=torch.long)
                logits = self._net(x_t=x_t, t=t_vec, cond=cond)
                p_theta = torch.softmax(logits, dim=1)  # (B,K), over x0

                beta_t = float(betas[t - 1].item())
                a_prev = float(alpha_bar[t - 1].item())

                # q(x_t | x_{t-1}=i)
                q1 = torch.full((int(n), K), beta_t / K, device=device, dtype=torch.float32)
                q1.scatter_(1, x_t.view(-1, 1), (1.0 - beta_t) + (beta_t / K))

                # q(x_{t-1}=i | x0=k0)
                base = (1.0 - a_prev) / K
                q2 = torch.full((K, K), base, device=device, dtype=torch.float32)
                q2.view(-1)[:: K + 1] += a_prev  # add to diagonal

                u = q2.unsqueeze(0) * q1.unsqueeze(1)  # (B,K0,Kprev)
                z = u.sum(dim=2, keepdim=True).clamp(min=1e-12)
                q_post = u / z

                p_prev = torch.einsum("bk,bki->bi", p_theta, q_post).clamp(min=0.0)
                p_prev = p_prev / p_prev.sum(dim=1, keepdim=True).clamp(min=1e-12)
                x_t = torch.multinomial(p_prev, num_samples=1).view(-1)

            return x_t.detach().cpu()


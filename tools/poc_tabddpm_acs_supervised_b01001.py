#!/usr/bin/env python3
from __future__ import annotations

"""
PoC (Scheme C idea): ACS-supervised diffusion on tract-level age×sex, with PUMS as external validation.

Core principle (PI-aligned):
- Training uses ONLY ACS tract-level distributions (B01001), plus tract_context (geo + built).
- PUMS microdata is used ONLY for external validation at the PUMA level (never used in training).

Why "pseudo-individuals":
- Diffusion models are trained on samples x0.
- ACS provides distribution-level supervision; we convert it into sample-level supervision by
  sampling pseudo-individuals from tract-level B01001 age×sex distributions.

This script implements:
1) Build tract_context (geo-only / built-only / geo+built) and a "none" ablation.
2) 4-fold CV by PUMA blocks (pairs of adjacent PUMAs; greedy pairing).
3) Internal evaluation: per-tract TVD vs ACS on held-out tracts.
4) External evaluation: aggregate tract predictions to PUMA and compare vs PUMS (TVD),
   plus a baseline gap (ACS->PUMA vs PUMS) as a method-independent lower bound.
"""

import argparse
import json
import math
import pathlib
import sys
from typing import Any


# Allow running as a plain script without installing the repo.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require(pkg: str) -> Any:
    try:
        return __import__(pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency: {pkg}. Install it in your conda env.\n"
            "Recommended: conda install -c conda-forge pandas numpy geopandas pyproj shapely\n"
            "and install torch (CUDA if available)."
        ) from e


class _CategoricalMLPModel:
    """
    Minimal categorical model for discrete x0 (K classes) conditioned on a vector c.

    This is a KISS alternative to Gaussian DDPM when the target is purely discrete.
    """

    def __init__(self, *, cond_dim: int, n_classes: int = 46, hidden_dims: tuple[int, ...] = (256, 256), seed: int = 0) -> None:
        self.cond_dim = int(cond_dim)
        self.n_classes = int(n_classes)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.seed = int(seed)
        self._net = None

    def _init_model(self, *, device: str) -> None:
        torch = _require("torch")
        if self._net is not None:
            return

        class Net(torch.nn.Module):
            def __init__(self, *, cond_dim: int, n_classes: int, hidden_dims: tuple[int, ...]) -> None:
                super().__init__()
                layers: list[torch.nn.Module] = []
                in_dim = int(cond_dim)
                for h in hidden_dims:
                    layers.append(torch.nn.Linear(in_dim, int(h)))
                    layers.append(torch.nn.ReLU())
                    in_dim = int(h)
                layers.append(torch.nn.Linear(in_dim, int(n_classes)))
                self.net = torch.nn.Sequential(*layers)

            def forward(self, c: Any) -> Any:
                return self.net(c)

        torch.manual_seed(int(self.seed))
        self._net = Net(cond_dim=self.cond_dim, n_classes=self.n_classes, hidden_dims=self.hidden_dims).to(device=device)

    def save(self, path: pathlib.Path) -> None:
        torch = _require("torch")
        if self._net is None:
            raise RuntimeError("Model is not initialized. Train (fit) or load a checkpoint first.")
        p = pathlib.Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "synthpop.catmlp.v0",
            "cond_dim": self.cond_dim,
            "n_classes": self.n_classes,
            "hidden_dims": list(self.hidden_dims),
            "seed": self.seed,
            "state_dict": self._net.state_dict(),
        }
        torch.save(payload, p)

    def load(self, path: pathlib.Path) -> None:
        torch = _require("torch")
        p = pathlib.Path(path).expanduser().resolve()
        payload = torch.load(p, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != "synthpop.catmlp.v0":
            raise ValueError(f"Unsupported checkpoint format: {p}")
        self.cond_dim = int(payload.get("cond_dim", 0))
        self.n_classes = int(payload.get("n_classes", 46))
        self.hidden_dims = tuple(int(x) for x in payload.get("hidden_dims", [256, 256]))
        self.seed = int(payload.get("seed", 0))
        self._net = None
        self._init_model(device="cpu")
        assert self._net is not None
        self._net.load_state_dict(payload["state_dict"])

    def fit(
        self,
        *,
        y: Any,
        cond: Any,
        epochs: int,
        batch_size: int,
        device: str | None,
        lr: float = 1e-3,
        log_every: int = 200,
    ) -> dict[str, float]:
        torch = _require("torch")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model(device=device)
        assert self._net is not None
        self._net.train()

        y = y.to(device=device, dtype=torch.long)
        cond = cond.to(device=device, dtype=torch.float32)
        if y.ndim != 1:
            raise ValueError(f"y must be (N,), got {tuple(y.shape)}")
        if cond.ndim != 2 or cond.shape[0] != y.shape[0] or cond.shape[1] != int(self.cond_dim):
            raise ValueError(f"cond must be (N,{self.cond_dim}), got {tuple(cond.shape)}")

        optim = torch.optim.Adam(self._net.parameters(), lr=float(lr))
        loss_fn = torch.nn.CrossEntropyLoss()

        n = int(y.shape[0])
        num_steps = 0
        last_loss = float("nan")
        for _ in range(int(epochs)):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, int(batch_size)):
                batch = idx[start : start + int(batch_size)]
                logits = self._net(cond[batch])
                loss = loss_fn(logits, y[batch])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                last_loss = float(loss.detach().cpu().item())
                num_steps += 1
                if log_every > 0 and num_steps % int(log_every) == 0:
                    print(f"[train] step={num_steps} loss={last_loss:.6f}")
        return {"loss": float(last_loss)}

    def sample(self, *, n: int, cond: Any, device: str | None = None) -> Any:
        torch = _require("torch")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_model(device=device)
        assert self._net is not None
        self._net.eval()

        cond = cond.to(device=device, dtype=torch.float32)
        if cond.ndim != 2 or cond.shape[0] != int(n) or cond.shape[1] != int(self.cond_dim):
            raise ValueError(f"cond must be (N,{self.cond_dim}) where N==n, got {tuple(cond.shape)}")

        with torch.inference_mode():
            logits = self._net(cond)
            probs = torch.softmax(logits, dim=1)
            y = torch.multinomial(probs, num_samples=1).reshape(-1)
            return y.detach().cpu()


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_puma(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return str(int(float(value)))
    except Exception:
        return None


def _pick_col(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _age23_bins() -> list[tuple[int, int, str]]:
    """
    ACS B01001 age groups (23 bins), shared by male/female.
    Uses left-closed, right-open integer intervals on AGEP.
    """
    return [
        (0, 5, "0-4"),
        (5, 10, "5-9"),
        (10, 15, "10-14"),
        (15, 18, "15-17"),
        (18, 20, "18-19"),
        (20, 21, "20"),
        (21, 22, "21"),
        (22, 25, "22-24"),
        (25, 30, "25-29"),
        (30, 35, "30-34"),
        (35, 40, "35-39"),
        (40, 45, "40-44"),
        (45, 50, "45-49"),
        (50, 55, "50-54"),
        (55, 60, "55-59"),
        (60, 62, "60-61"),
        (62, 65, "62-64"),
        (65, 67, "65-66"),
        (67, 70, "67-69"),
        (70, 75, "70-74"),
        (75, 80, "75-79"),
        (80, 85, "80-84"),
        (85, 200, "85+"),
    ]


def _age23_index(age: Any) -> int | None:
    try:
        a = int(float(age))
    except Exception:
        return None
    if a < 0:
        a = 0
    for i, (lo, hi, _lab) in enumerate(_age23_bins()):
        if lo <= a < hi:
            return i
    return len(_age23_bins()) - 1


def _b01001_columns() -> tuple[list[str], list[str]]:
    male = [f"B01001_{i:03d}E" for i in range(3, 26)]  # 003..025 (23 bins)
    female = [f"B01001_{i:03d}E" for i in range(27, 50)]  # 027..049 (23 bins)
    return male, female


def _read_acs_b01001(path: pathlib.Path) -> Any:
    pd = _require("pandas")
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    needed = ["state", "county", "tract", "B01001_001E"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"ACS B01001 missing columns: {missing}. Columns: {list(df.columns)[:30]}")

    state = df["state"].astype(str).str.zfill(2)
    county = df["county"].astype(str).str.zfill(3)
    tract = df["tract"].astype(str).str.zfill(6)
    df["tract_geoid"] = (state + county + tract).astype(str)

    # Numericize target columns.
    df["B01001_001E"] = pd.to_numeric(df["B01001_001E"], errors="coerce").fillna(0.0).clip(lower=0.0)
    male_cols, female_cols = _b01001_columns()
    for c in male_cols + female_cols + ["B01001_002E", "B01001_026E"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    return df


def _b01001_targets_by_tract(df_b01001: Any, *, tracts: set[str]) -> dict[str, dict[str, Any]]:
    """
    Build per-tract targets:
    - total_pop
    - p_joint (46)
    - p_age (23)
    - p_sex (2)
    - counts (for ACS->PUMA baseline)
    """
    np = _require("numpy")

    male_cols, female_cols = _b01001_columns()
    missing = [c for c in (male_cols + female_cols) if c not in df_b01001.columns]
    if missing:
        raise SystemExit(f"ACS B01001 missing detailed columns (need 003..025 and 027..049). Missing: {missing[:10]}")

    out: dict[str, dict[str, Any]] = {}
    for r in df_b01001.itertuples(index=False):
        tg = str(getattr(r, "tract_geoid"))
        if tg not in tracts:
            continue
        total = float(getattr(r, "B01001_001E"))
        male = np.array([float(getattr(r, c)) for c in male_cols], dtype=float)
        female = np.array([float(getattr(r, c)) for c in female_cols], dtype=float)
        joint = np.concatenate([male, female], axis=0)  # 46, male then female
        denom = total if total > 0 else float(joint.sum())
        denom = denom if denom > 0 else 1.0
        p_joint = (joint / denom).astype(float)
        p_age = ((male + female) / denom).astype(float)
        p_sex = (np.array([male.sum(), female.sum()], dtype=float) / denom).astype(float)
        out[tg] = {
            "total_pop": float(total),
            "p_joint": p_joint,
            "p_age": p_age,
            "p_sex": p_sex,
            "counts_joint": joint.astype(float),
            "counts_age": (male + female).astype(float),
            "counts_sex": np.array([male.sum(), female.sum()], dtype=float),
        }
    if not out:
        raise SystemExit("No matching tracts found in ACS B01001 for the given study area.")
    return out


def _load_buildings(buildings_csv: pathlib.Path, *, n_tiers: int) -> Any:
    pd = _require("pandas")
    b = pd.read_csv(buildings_csv, low_memory=False)
    needed = ["bldg_id", "puma", "tract_geoid", "footprint_area_m2", "height_m", "cap_proxy"]
    missing = [c for c in needed if c not in b.columns]
    if missing:
        raise SystemExit(f"buildings_csv missing columns: {missing}")
    b["tract_geoid"] = b["tract_geoid"].astype(str)
    b["puma"] = b["puma"].map(_normalize_puma)
    b["footprint_area_m2"] = pd.to_numeric(b["footprint_area_m2"], errors="coerce").fillna(0.0).clip(lower=0.0)
    b["height_m"] = pd.to_numeric(b["height_m"], errors="coerce").fillna(0.0).clip(lower=0.0)
    b["cap_proxy"] = pd.to_numeric(b["cap_proxy"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "price_tier" in b.columns:
        b["price_tier"] = pd.to_numeric(b["price_tier"], errors="coerce")
        b.loc[(b["price_tier"] < 1) | (b["price_tier"] > int(n_tiers)), "price_tier"] = float("nan")
    return b


def _tract_to_puma_from_buildings(buildings: Any) -> dict[str, str]:
    pd = _require("pandas")
    tract_to_puma: dict[str, str] = {}
    g = buildings.dropna(subset=["tract_geoid", "puma"]).copy()
    if g.empty:
        return tract_to_puma
    for tract, sub in g.groupby("tract_geoid", sort=False):
        mode = sub["puma"].astype(str).value_counts(dropna=True)
        if mode.empty:
            continue
        tract_to_puma[str(tract)] = str(mode.index[0])
    return tract_to_puma


def _build_built_context(buildings: Any, *, n_tiers: int) -> Any:
    pd = _require("pandas")
    import numpy as np  # type: ignore

    g = buildings.groupby("tract_geoid", sort=False)
    out = pd.DataFrame(
        {
            "tract_geoid": g.size().index.astype(str),
            "n_buildings": g.size().to_numpy(dtype=float),
            "cap_proxy_sum": g["cap_proxy"].sum().to_numpy(dtype=float),
            "height_mean": g["height_m"].mean().to_numpy(dtype=float),
            "footprint_mean": g["footprint_area_m2"].mean().to_numpy(dtype=float),
        }
    )
    out["n_buildings_log"] = np.log1p(out["n_buildings"].astype(float))
    out["cap_proxy_sum_log"] = np.log1p(out["cap_proxy_sum"].astype(float))
    out["footprint_mean_log"] = np.log1p(out["footprint_mean"].astype(float))

    # Price tier histogram (proportions) if available.
    for k in range(1, int(n_tiers) + 1):
        out[f"price_tier_p{k}"] = 0.0
    if "price_tier" in buildings.columns:
        b2 = buildings.dropna(subset=["price_tier"]).copy()
        if not b2.empty:
            b2["price_tier"] = pd.to_numeric(b2["price_tier"], errors="coerce")
            b2 = b2.dropna(subset=["price_tier"]).copy()
            b2["price_tier"] = b2["price_tier"].astype(int)
            b2 = b2[(b2["price_tier"] >= 1) & (b2["price_tier"] <= int(n_tiers))].copy()
        if not b2.empty:
            counts = (
                b2.groupby(["tract_geoid", "price_tier"], sort=False)["bldg_id"]
                .size()
                .unstack(fill_value=0)
                .reindex(columns=list(range(1, int(n_tiers) + 1)), fill_value=0)
            )
            denom = counts.sum(axis=1).replace(0, 1).astype(float)
            props = counts.div(denom, axis=0).astype(float)
            props.columns = [f"price_tier_p{k}" for k in range(1, int(n_tiers) + 1)]
            props = props.reset_index()
            props["tract_geoid"] = props["tract_geoid"].astype(str)
            # Use explicit suffix to avoid pandas' default _x/_y, then prefer the right-side props.
            out = out.merge(props, on="tract_geoid", how="left", suffixes=("", "_tier"))
            for k in range(1, int(n_tiers) + 1):
                col = f"price_tier_p{k}"
                col_tier = f"{col}_tier"
                if col_tier in out.columns:
                    out[col] = pd.to_numeric(out[col_tier], errors="coerce").fillna(0.0).astype(float)
                else:
                    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
            drop_cols = [f"price_tier_p{k}_tier" for k in range(1, int(n_tiers) + 1) if f"price_tier_p{k}_tier" in out.columns]
            if drop_cols:
                out = out.drop(columns=drop_cols)
    return out


def _build_geo_context(*, tiger_tract_zip: pathlib.Path, tracts: set[str], cbd_lon: float, cbd_lat: float) -> Any:
    gpd = _require("geopandas")
    pd = _require("pandas")
    pyproj = _require("pyproj")

    tract = gpd.read_file(f"zip://{tiger_tract_zip}")
    if tract.crs is None:
        tract = tract.set_crs(4269, allow_override=True)
    tract = tract.to_crs(3857)

    geoid_col = _pick_col(list(tract.columns), ("GEOID", "GEOID20", "GEOID10"))
    if geoid_col is None:
        raise SystemExit(f"Cannot find tract GEOID column in: {tiger_tract_zip}")
    tract = tract[[geoid_col, "geometry"]].rename(columns={geoid_col: "tract_geoid"})
    tract["tract_geoid"] = tract["tract_geoid"].astype(str)
    tract = tract[tract["tract_geoid"].isin(sorted(tracts))].copy()
    if tract.empty:
        raise SystemExit("No matching tracts found in TIGER tract zip for the given study area.")

    cent = tract.geometry.centroid
    cent_gdf = gpd.GeoDataFrame(tract[["tract_geoid"]].copy(), geometry=cent, crs=3857)
    cent_ll = cent_gdf.to_crs(4326)

    # Area (km^2)
    area_km2 = tract.geometry.area.astype(float) / 1e6

    # Dist to CBD (in km)
    tr = pyproj.Transformer.from_crs(4326, 3857, always_xy=True)
    cbd_x, cbd_y = tr.transform(float(cbd_lon), float(cbd_lat))
    dx = cent_gdf.geometry.x.to_numpy(dtype=float) - float(cbd_x)
    dy = cent_gdf.geometry.y.to_numpy(dtype=float) - float(cbd_y)
    dist_cbd_km = (dx * dx + dy * dy) ** 0.5 / 1000.0

    out = pd.DataFrame(
        {
            "tract_geoid": cent_ll["tract_geoid"].astype(str),
            "centroid_lon": cent_ll.geometry.x.astype(float),
            "centroid_lat": cent_ll.geometry.y.astype(float),
            "area_km2": area_km2.to_numpy(dtype=float),
            "dist_cbd_km": dist_cbd_km.astype(float),
        }
    )
    return out


def _standardize(df: Any, *, cols: list[str], mean: dict[str, float] | None = None, std: dict[str, float] | None = None) -> tuple[Any, dict[str, float], dict[str, float]]:
    pd = _require("pandas")
    out = df.copy()
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
        mu = float(x.mean()) if mean is None else float(mean[c])
        sd = float(x.std()) if std is None else float(std[c])
        if not (sd > 1e-6):
            sd = 1.0
        out[c] = ((x - mu) / sd).astype(float)
        mean_out[c] = mu
        std_out[c] = sd
    return out, mean_out, std_out


def _tvd(p: Any, q: Any) -> float:
    np = _require("numpy")
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def _marginals_from_joint(p_joint: Any) -> tuple[Any, Any]:
    """
    Convert a 46-dim (sex x age23) joint distribution into:
      - p_age: (23,)
      - p_sex: (2,)
    """
    np = _require("numpy")
    p = np.asarray(p_joint, dtype=float).reshape(-1)
    if p.size != 46:
        raise ValueError(f"p_joint must have length 46, got {p.size}")
    male = p[:23]
    female = p[23:]
    p_age = (male + female).astype(float)
    p_sex = np.array([float(male.sum()), float(female.sum())], dtype=float)
    p_age = p_age / (float(p_age.sum()) if float(p_age.sum()) > 0 else 1.0)
    p_sex = p_sex / (float(p_sex.sum()) if float(p_sex.sum()) > 0 else 1.0)
    return p_age, p_sex


def _joint_from_marginals(*, p_age: Any, p_sex: Any) -> Any:
    """
    Build a 46-dim joint (male-then-female, each 23 age bins) assuming independence:
      p(age,sex) = p(age) * p(sex)
    """
    np = _require("numpy")
    a = np.asarray(p_age, dtype=float).reshape(-1)
    s = np.asarray(p_sex, dtype=float).reshape(-1)
    if a.size != 23:
        raise ValueError(f"p_age must have length 23, got {a.size}")
    if s.size != 2:
        raise ValueError(f"p_sex must have length 2, got {s.size}")
    a = np.clip(a, 0.0, None)
    s = np.clip(s, 0.0, None)
    a = a / (float(a.sum()) if float(a.sum()) > 0 else 1.0)
    s = s / (float(s.sum()) if float(s.sum()) > 0 else 1.0)
    male = a * float(s[0])
    female = a * float(s[1])
    out = np.concatenate([male, female], axis=0).astype(float)
    out = out / (float(out.sum()) if float(out.sum()) > 0 else 1.0)
    return out


def _ipf_age23_sex2(
    *,
    seed_joint: Any,
    target_p_age: Any,
    target_p_sex: Any,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Any:
    """
    IPF (raking) baseline for a 23x2 table:
    - Seed from training tracts' average joint (train-only).
    - Fit to target marginals (p_age, p_sex) for a test tract.

    Returns a 46-dim joint distribution (male then female).
    """
    np = _require("numpy")

    seed = np.asarray(seed_joint, dtype=float).reshape(-1)
    if seed.size != 46:
        raise ValueError(f"seed_joint must have length 46, got {seed.size}")
    seed = np.clip(seed, 0.0, None)
    seed = seed / (float(seed.sum()) if float(seed.sum()) > 0 else 1.0)

    r = np.asarray(target_p_age, dtype=float).reshape(-1)
    c = np.asarray(target_p_sex, dtype=float).reshape(-1)
    if r.size != 23:
        raise ValueError(f"target_p_age must have length 23, got {r.size}")
    if c.size != 2:
        raise ValueError(f"target_p_sex must have length 2, got {c.size}")
    r = np.clip(r, 0.0, None)
    c = np.clip(c, 0.0, None)
    r = r / (float(r.sum()) if float(r.sum()) > 0 else 1.0)
    c = c / (float(c.sum()) if float(c.sum()) > 0 else 1.0)

    table = np.stack([seed[:23], seed[23:]], axis=1).astype(float)  # (23,2)
    table = table + 1e-12  # avoid exact zeros in the seed
    table = table / float(table.sum())

    for _ in range(int(max_iter)):
        # Row scaling.
        row_sum = table.sum(axis=1)
        row_factor = np.zeros_like(row_sum)
        m = row_sum > 0
        row_factor[m] = r[m] / row_sum[m]
        table = table * row_factor.reshape(-1, 1)
        if bool((r <= 0).any()):
            table[r <= 0, :] = 0.0

        # Column scaling.
        col_sum = table.sum(axis=0)
        col_factor = np.zeros_like(col_sum)
        m = col_sum > 0
        col_factor[m] = c[m] / col_sum[m]
        table = table * col_factor.reshape(1, -1)
        if bool((c <= 0).any()):
            table[:, c <= 0] = 0.0

        # Convergence check.
        if float(np.max(np.abs(table.sum(axis=1) - r))) < float(tol) and float(np.max(np.abs(table.sum(axis=0) - c))) < float(tol):
            break

    out = np.concatenate([table[:, 0], table[:, 1]], axis=0).astype(float)
    out = np.clip(out, 0.0, None)
    out = out / (float(out.sum()) if float(out.sum()) > 0 else 1.0)
    return out


def _sample_pseudo(
    *,
    rng: Any,
    p_joint: Any,
    n: int,
) -> tuple[Any, Any]:
    """
    Sample pseudo individuals from a 46-dim joint distribution.
    Returns:
      age_idx: (n,) int in [0,22]
      sex01: (n,) int in {0,1} (0=male,1=female)
    """
    np = _require("numpy")
    p = np.asarray(p_joint, dtype=float)
    if p.size != 46:
        raise ValueError(f"p_joint must have length 46, got {p.size}")
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        p = np.full((46,), 1.0 / 46.0, dtype=float)
    else:
        p = p / s
    idx = rng.choice(46, size=int(n), replace=True, p=p)
    sex01 = (idx // 23).astype(int)
    age_idx = (idx % 23).astype(int)
    return age_idx, sex01


def _decode_samples(samples: Any) -> tuple[Any, Any]:
    """
    Decode sampled (age_u, sex_u) in [0,1]-like space into discrete bins.
    """
    np = _require("numpy")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"samples must be (N,2), got {x.shape}")
    age_u = np.clip(x[:, 0], 0.0, 1.0)
    sex_u = np.clip(x[:, 1], 0.0, 1.0)
    age_idx = np.clip(np.rint(age_u * 22.0), 0, 22).astype(int)
    sex01 = np.clip(np.rint(sex_u), 0, 1).astype(int)
    return age_idx, sex01


def _encode_onehot(*, age_idx: Any, sex01: Any) -> Any:
    """
    One-hot encoding:
      - age: 23 dims
      - sex: 2 dims (0=male,1=female)
    Returns x_u: (N,25)
    """
    np = _require("numpy")
    a = np.asarray(age_idx, dtype=int).reshape(-1)
    s = np.asarray(sex01, dtype=int).reshape(-1)
    if a.shape[0] != s.shape[0]:
        raise ValueError("age_idx and sex01 must have the same length")
    n = int(a.shape[0])
    age_oh = np.eye(23, dtype=np.float32)[np.clip(a, 0, 22)]
    sex_oh = np.eye(2, dtype=np.float32)[np.clip(s, 0, 1)]
    out = np.concatenate([age_oh, sex_oh], axis=1).reshape(n, 25)
    return out


def _decode_onehot(samples: Any) -> tuple[Any, Any]:
    """
    Decode one-hot-like continuous vectors into discrete bins via argmax.
    samples: (N,25)
    """
    np = _require("numpy")
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[1] != 25:
        raise ValueError(f"samples must be (N,25), got {x.shape}")
    age_logits = x[:, :23]
    sex_logits = x[:, 23:]
    age_idx = np.argmax(age_logits, axis=1).astype(int)
    sex01 = np.argmax(sex_logits, axis=1).astype(int)
    return age_idx, sex01


def _p_from_samples(*, age_idx: Any, sex01: Any) -> dict[str, Any]:
    np = _require("numpy")
    n = int(len(age_idx))
    if n <= 0:
        return {"p_joint": np.full((46,), 1.0 / 46.0), "p_age": np.full((23,), 1.0 / 23.0), "p_sex": np.full((2,), 0.5)}

    joint_counts = np.zeros((46,), dtype=float)
    for a, s in zip(age_idx, sex01, strict=False):
        idx = int(s) * 23 + int(a)
        joint_counts[idx] += 1.0
    p_joint = joint_counts / float(n)

    age_counts = np.zeros((23,), dtype=float)
    for a in age_idx:
        age_counts[int(a)] += 1.0
    p_age = age_counts / float(n)

    sex_counts = np.zeros((2,), dtype=float)
    for s in sex01:
        sex_counts[int(s)] += 1.0
    p_sex = sex_counts / float(n)

    return {"p_joint": p_joint, "p_age": p_age, "p_sex": p_sex}


def _load_pums_persons(*, data_root: pathlib.Path, pums_year: int, pums_period: str, statefp: str, pumas: set[str], n_rows: int) -> Any:
    """
    Load PUMS person file (minimal columns) for external validation.
    Uses the same default path/search as tools/poc_tabddpm_pums_buildingcond.py.
    """
    import zipfile

    pd = _require("pandas")

    statefp = str(statefp).zfill(2)
    state_postal_lower = "mi" if statefp == "26" else None
    if state_postal_lower is None:
        raise SystemExit(f"Unsupported --statefp={statefp}. v0 only supports MI (26).")

    raw_dir = data_root / "detroit" / "raw" / "pums" / f"pums_{pums_year}_{pums_period}"
    candidates = [
        raw_dir / f"psam_p{statefp}.zip",
        raw_dir / f"csv_p{state_postal_lower}.zip",
    ]
    zip_path = next((p for p in candidates if p.exists()), candidates[0])
    if not zip_path.exists():
        raise SystemExit(f"PUMS zip not found. Tried: {candidates[0]} and {candidates[1]}")

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise SystemExit(f"No CSV members found inside: {zip_path}")
        member = sorted(members)[0]
        with zf.open(member) as f:
            df = pd.read_csv(f, nrows=int(n_rows), low_memory=False)

    cols = ["AGEP", "SEX", "PUMA"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"PUMS person file missing columns: {missing}")
    df = df[cols].copy()
    df["AGEP"] = pd.to_numeric(df["AGEP"], errors="coerce")
    df["SEX"] = pd.to_numeric(df["SEX"], errors="coerce")
    df["PUMA"] = pd.to_numeric(df["PUMA"], errors="coerce")
    df = df.dropna().copy()
    df["PUMA_STR"] = df["PUMA"].astype(int).astype(str)
    df = df[df["PUMA_STR"].isin(set(map(str, pumas)))].copy()
    if df.empty:
        raise SystemExit("After filtering to study PUMAs, no PUMS rows remain.")
    return df.reset_index(drop=True)


def _pums_puma_distributions(df_pums: Any) -> dict[str, dict[str, Any]]:
    """
    Return per-PUMA distributions over:
      - p_joint (46)
      - p_age (23)
      - p_sex (2)
    """
    import numpy as np  # type: ignore
    pd = _require("pandas")

    out: dict[str, dict[str, Any]] = {}
    for puma, sub in df_pums.groupby("PUMA_STR", sort=False):
        age_idx = sub["AGEP"].apply(_age23_index)
        m = ~age_idx.isna()
        if not bool(m.any()):
            continue
        age_idx = age_idx[m].astype(int).to_numpy(dtype=int)
        sex = pd.to_numeric(sub.loc[m, "SEX"], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        sex01 = np.where(sex == 2, 1, 0).astype(int)  # default male for unknown
        stats = _p_from_samples(age_idx=age_idx, sex01=sex01)
        out[str(puma)] = stats
    if not out:
        raise SystemExit("Failed to build PUMS per-PUMA distributions (empty after binning).")
    return out


def _build_puma_blocks(*, tiger_puma_zip: pathlib.Path, pumas: list[str]) -> list[list[str]]:
    """
    Create 4 blocks by pairing adjacent PUMAs (greedy).
    """
    gpd = _require("geopandas")

    puma_gdf = gpd.read_file(f"zip://{tiger_puma_zip}")
    if puma_gdf.crs is None:
        puma_gdf = puma_gdf.set_crs(4269, allow_override=True)
    puma_gdf = puma_gdf.to_crs(3857)

    puma_col = _pick_col(list(puma_gdf.columns), ("PUMACE20", "PUMA", "PUMACE10"))
    if puma_col is None:
        raise SystemExit(f"Cannot find PUMA code column in: {tiger_puma_zip}")

    puma_gdf[puma_col] = puma_gdf[puma_col].map(_normalize_puma)
    p = sorted(set(map(str, pumas)))
    puma_gdf = puma_gdf[puma_gdf[puma_col].astype(str).isin(p)].copy()
    if puma_gdf.empty:
        raise SystemExit("No study PUMAs found in TIGER puma zip.")

    # Build adjacency list.
    geoms = {str(r[puma_col]): r.geometry for _, r in puma_gdf.iterrows()}
    adj: dict[str, set[str]] = {k: set() for k in geoms}
    keys = list(geoms.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            ga, gb = geoms[a], geoms[b]
            try:
                touch = bool(ga.touches(gb))
            except Exception:
                touch = False
            if touch:
                adj[a].add(b)
                adj[b].add(a)

    # Greedy pairing.
    unpaired = set(keys)
    blocks: list[list[str]] = []
    while unpaired:
        a = min(unpaired, key=lambda k: (len(adj.get(k, set())), k))
        neigh = sorted([b for b in adj.get(a, set()) if b in unpaired and b != a], key=lambda k: (len(adj.get(k, set())), k))
        if neigh:
            b = neigh[0]
        else:
            # Fallback: pair with any remaining (still gives 4 folds, but not adjacency-guaranteed).
            b = sorted([k for k in unpaired if k != a])[0]
        blocks.append([a, b])
        unpaired.remove(a)
        unpaired.remove(b)

    return blocks


def _parse_puma_blocks(spec: str) -> list[list[str]]:
    """
    Parse a user-specified PUMA block split, e.g.:
      "3202,3203;3208,3209;3210,3211;3212,3213"
    Returns a list of blocks (each a list of 2+ PUMA codes as strings).
    """
    blocks: list[list[str]] = []
    for part in str(spec).split(";"):
        part = part.strip()
        if not part:
            continue
        items = [s.strip() for s in part.split(",") if s.strip()]
        if len(items) < 2:
            raise ValueError(f"Each block must contain >=2 PUMAs; got: {part!r}")
        blocks.append([str(_normalize_puma(x) or x) for x in items])
    if not blocks:
        raise ValueError("Empty --puma_blocks spec.")
    return blocks


def main() -> None:
    np = _require("numpy")
    pd = _require("pandas")
    torch = _require("torch")

    from src.synthpop.model.diffusion_tabular import DiffusionTabularModel, TabDDPMConfig
    from src.synthpop.model.diffusion_categorical import CategoricalDiffusionConfig, CategoricalDiffusionModel
    from src.synthpop.pipeline.detroit_v0 import make_run_id

    p = argparse.ArgumentParser(prog="poc_tabddpm_acs_supervised_b01001")
    p.add_argument("--acs_b01001_csv_gz", required=True, help="ACS B01001 tract CSV.gz (downloaded by detroit_fetch_public_data.py).")
    p.add_argument("--buildings_csv", required=True, help="Buildings CSV with tract_geoid and puma (optionally price_tier).")
    p.add_argument("--tiger_tract_zip", required=True, help="TIGER tract zip (tl_2023_26_tract.zip).")
    p.add_argument("--tiger_puma_zip", required=True, help="TIGER puma zip (tl_2023_26_puma20.zip).")
    p.add_argument("--data_root", default=None, help="Detroit data_root (only for external PUMS validation).")
    p.add_argument("--pums_year", type=int, default=2023)
    p.add_argument("--pums_period", default="5-Year")
    p.add_argument("--statefp", default="26")
    p.add_argument("--pums_n_rows", type=int, default=200_000)
    p.add_argument("--n_tiers", type=int, default=5)
    p.add_argument("--cbd_lon", type=float, default=-83.0458)
    p.add_argument("--cbd_lat", type=float, default=42.3314)
    p.add_argument(
        "--exclude_pumas",
        default="",
        help='Optional comma-separated PUMA codes to exclude (e.g. "3202,3203"). Useful to bypass known ACS coverage issues while debugging.',
    )

    p.add_argument("--n_pseudo_base", type=int, default=500, help="Base pseudo-individuals per tract (scaled by sqrt(pop)).")
    p.add_argument("--n_pseudo_min", type=int, default=100)
    p.add_argument("--n_pseudo_max", type=int, default=1500)
    p.add_argument("--n_eval_per_tract", type=int, default=2000)

    p.add_argument(
        "--x_model",
        default="tabddpm_scalar",
        choices=[
            "tabddpm_scalar",
            "tabddpm_onehot",
            "cat_mlp",
            "cat_diffusion_concat",
            "cat_diffusion_xattn",
            "joint_tabddpm_logp",
        ],
        help=(
            "Generative model / encoding for discrete age×sex. "
            "tabddpm_scalar: current (age_u,sex_u) in R^2. "
            "tabddpm_onehot: one-hot age(23)+sex(2) with Gaussian DDPM. "
            "cat_mlp: categorical MLP over 46 joint states (KISS discrete baseline). "
            "cat_diffusion_concat: multinomial diffusion over 46 joint states (concat conditioning). "
            "cat_diffusion_xattn: multinomial diffusion over 46 joint states (cross-attn conditioning)."
            "joint_tabddpm_logp: Gaussian DDPM over tract-level 46-dim log-prob vectors; decode with softmax to p_joint. "
            "This is a distribution-to-distribution model (one sample per tract), not pseudo-individual generation."
        ),
    )
    p.add_argument("--cat_mlp_lr", type=float, default=1e-3)
    p.add_argument("--cat_diffusion_lr", type=float, default=1e-3)
    p.add_argument(
        "--n_eval_joint_samples",
        type=int,
        default=64,
        help="For x_model=joint_tabddpm_logp: number of joint-vector samples per tract to average at evaluation time.",
    )

    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument(
        "--puma_blocks",
        default=None,
        help='Optional explicit PUMA blocks (overrides adjacency-based pairing). Example: "3202,3203;3208,3209;3210,3211;3212,3213".',
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed run_dir: skip finished (fold,condition) pairs and reuse existing checkpoints for evaluation.",
    )

    p.add_argument(
        "--conditions",
        default="none,geo-only,built-only,geo+built",
        help='Comma-separated conditions: "none", "marginal", "geo-only", "built-only", "geo+built". Default keeps the original ablations; use --conditions "none,marginal" for the ecological-inference setting.',
    )
    p.add_argument("--fold", type=int, default=-1, help="Run a single fold index (0..3). -1 = run all folds.")
    p.add_argument("--out_dir", default=None, help="Output directory (default: outputs/<run_id>).")
    args = p.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    acs_path = pathlib.Path(args.acs_b01001_csv_gz).expanduser().resolve()
    buildings_csv = pathlib.Path(args.buildings_csv).expanduser().resolve()
    tiger_tract_zip = pathlib.Path(args.tiger_tract_zip).expanduser().resolve()
    tiger_puma_zip = pathlib.Path(args.tiger_puma_zip).expanduser().resolve()

    if args.out_dir:
        out_root = pathlib.Path(args.out_dir).expanduser().resolve()
    else:
        out_root = pathlib.Path("outputs") / make_run_id(prefix="poc_acs_supervised_b01001")
    out_root.mkdir(parents=True, exist_ok=True)

    buildings = _load_buildings(buildings_csv, n_tiers=int(args.n_tiers))
    tract_to_puma = _tract_to_puma_from_buildings(buildings)
    exclude_pumas = {str(_normalize_puma(x) or x) for x in str(args.exclude_pumas).split(",") if str(x).strip()}
    exclude_pumas = {p for p in exclude_pumas if p and p.lower() not in {"nan", "none"}}

    study_tracts = {tg for tg, p in tract_to_puma.items() if str(p) not in exclude_pumas}
    study_pumas = sorted({str(p) for tg, p in tract_to_puma.items() if tg in study_tracts})
    if len(study_pumas) < 2:
        raise SystemExit(f"Too few study PUMAs inferred from buildings_csv: {study_pumas}")

    # Targets from ACS.
    b01001 = _read_acs_b01001(acs_path)
    targets_by_tract = _b01001_targets_by_tract(b01001, tracts=study_tracts)

    # --- Coverage check (fail-fast): ACS must cover all study PUMAs ---
    cov_by_puma: dict[str, Any] = {}
    bad_pumas: list[str] = []
    for puma in study_pumas:
        tracts = sorted([tg for tg, p in tract_to_puma.items() if str(p) == str(puma) and tg in study_tracts])
        in_targets = [tg for tg in tracts if tg in targets_by_tract]
        pop_sum = float(sum(float(targets_by_tract[tg]["total_pop"]) for tg in in_targets))
        cov_by_puma[str(puma)] = {
            "n_tracts_study": int(len(tracts)),
            "n_tracts_in_acs": int(len(in_targets)),
            "total_pop_sum": pop_sum,
        }
        if len(in_targets) == 0 or not (pop_sum > 0):
            bad_pumas.append(str(puma))

    _write_json(out_root / "metrics" / "acs_b01001_coverage.json", {"by_puma": cov_by_puma, "bad_pumas": sorted(set(bad_pumas))})
    if bad_pumas:
        hint = (
            "ACS B01001 coverage is incomplete for some study PUMAs. "
            "This will silently produce TVD=0.5 artifacts. "
            f"Bad PUMAs: {sorted(set(bad_pumas))}. "
            "Inspect outputs/<run_id>/metrics/acs_b01001_coverage.json and run tools/diagnose_acs_b01001_coverage.py on the workstation. "
            'Workaround: re-run with --exclude_pumas "3202,3203" (or fix ACS download scope / tract_geoid mapping).'
        )
        raise SystemExit(hint)

    # Context features.
    geo_ctx = _build_geo_context(tiger_tract_zip=tiger_tract_zip, tracts=study_tracts, cbd_lon=float(args.cbd_lon), cbd_lat=float(args.cbd_lat))
    built_ctx = _build_built_context(buildings, n_tiers=int(args.n_tiers))
    ctx = geo_ctx.merge(built_ctx, on="tract_geoid", how="left")
    for c in ctx.columns:
        if c == "tract_geoid":
            continue
        ctx[c] = pd.to_numeric(ctx[c], errors="coerce").fillna(0.0).astype(float)
    ctx["puma"] = ctx["tract_geoid"].map(lambda tg: tract_to_puma.get(str(tg)))
    ctx = ctx.dropna(subset=["puma"]).copy()
    if exclude_pumas:
        ctx = ctx[~ctx["puma"].astype(str).isin(sorted(exclude_pumas))].copy()

    # PUMA blocks for spatial holdout.
    if args.puma_blocks:
        blocks = _parse_puma_blocks(str(args.puma_blocks))
        # Filter to study pumas and validate coverage.
        blocks = [[p for p in b if str(p) in set(study_pumas)] for b in blocks]
        blocks = [b for b in blocks if len(b) >= 2]
        seen = sorted({p for b in blocks for p in b})
        missing = sorted(set(study_pumas) - set(seen))
        if missing:
            raise SystemExit(f"--puma_blocks does not cover all study PUMAs: missing={missing}")
    else:
        blocks = _build_puma_blocks(tiger_puma_zip=tiger_puma_zip, pumas=study_pumas)
    blocks = [sorted(list(map(str, b))) for b in blocks]
    blocks = sorted(blocks, key=lambda b: ",".join(b))
    if len(blocks) < 2:
        raise SystemExit(f"Failed to build PUMA blocks. Blocks: {blocks}")

    # Load external PUMS (optional but recommended).
    pums_puma_dist = None
    if args.data_root:
        data_root = pathlib.Path(args.data_root).expanduser().resolve()
        df_pums = _load_pums_persons(
            data_root=data_root,
            pums_year=int(args.pums_year),
            pums_period=str(args.pums_period),
            statefp=str(args.statefp),
            pumas=set(study_pumas),
            n_rows=int(args.pums_n_rows),
        )
        pums_puma_dist = _pums_puma_distributions(df_pums)

    cond_list = [c.strip() for c in str(args.conditions).split(",") if c.strip()]
    valid_cond = {"none", "marginal", "geo-only", "built-only", "geo+built"}
    for c in cond_list:
        if c not in valid_cond:
            raise SystemExit(f"Unknown condition: {c}. Valid: {sorted(valid_cond)}")

    # Baseline gap (ACS->PUMA vs PUMS), method-independent.
    baseline_gap = None
    if pums_puma_dist is not None:
        # Aggregate ACS counts to PUMA.
        acs_counts_by_puma: dict[str, Any] = {p: np.zeros((46,), dtype=float) for p in study_pumas}
        for tg, t in targets_by_tract.items():
            puma = tract_to_puma.get(str(tg))
            if not puma:
                continue
            acs_counts_by_puma[str(puma)] += np.asarray(t["counts_joint"], dtype=float)
        baseline_by_puma: dict[str, float] = {}
        for puma in study_pumas:
            ac = acs_counts_by_puma[str(puma)]
            ac_p = ac / (ac.sum() if ac.sum() > 0 else 1.0)
            pu_p = np.asarray(pums_puma_dist[str(puma)]["p_joint"], dtype=float)
            ac_age, ac_sex = _marginals_from_joint(ac_p)
            pu_age, pu_sex = _marginals_from_joint(pu_p)
            baseline_by_puma[str(puma)] = {
                "tvd_joint": float(_tvd(ac_p, pu_p)),
                "tvd_age": float(_tvd(ac_age, pu_age)),
                "tvd_sex": float(_tvd(ac_sex, pu_sex)),
            }
        vals_joint = [v["tvd_joint"] for v in baseline_by_puma.values()]
        vals_age = [v["tvd_age"] for v in baseline_by_puma.values()]
        vals_sex = [v["tvd_sex"] for v in baseline_by_puma.values()]
        baseline_gap = {
            "by_puma": baseline_by_puma,
            "summary": {
                "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint))} if vals_joint else None,
                "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age))} if vals_age else None,
                "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex))} if vals_sex else None,
            },
        }
        _write_json(out_root / "metrics" / "acs_pums_baseline_gap.json", baseline_gap)

    # Run folds (or a single fold).
    fold_indices = list(range(len(blocks)))
    if int(args.fold) >= 0:
        if int(args.fold) >= len(blocks):
            raise SystemExit(f"--fold out of range: {args.fold} (n_folds={len(blocks)})")
        fold_indices = [int(args.fold)]

    run_meta = {
        "out_root": str(out_root),
        "acs_b01001_csv_gz": str(acs_path),
        "buildings_csv": str(buildings_csv),
        "tiger_tract_zip": str(tiger_tract_zip),
        "tiger_puma_zip": str(tiger_puma_zip),
        "study_pumas": study_pumas,
        "n_tracts": int(len(set(ctx["tract_geoid"].astype(str).tolist()))),
        "puma_blocks": blocks,
        "conditions": cond_list,
        "x_model": str(args.x_model),
        "cat_mlp_lr": float(args.cat_mlp_lr),
        "cat_diffusion_lr": float(args.cat_diffusion_lr),
        "n_pseudo_base": int(args.n_pseudo_base),
        "n_eval_per_tract": int(args.n_eval_per_tract),
        "n_eval_joint_samples": int(args.n_eval_joint_samples),
        "timesteps": int(args.timesteps),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "baseline_gap": baseline_gap,
        "external_validation": {"enabled": bool(pums_puma_dist is not None), "pums_year": int(args.pums_year), "pums_period": str(args.pums_period)},
    }
    _write_json(out_root / "run_summary.json", run_meta)

    geo_cols = ["centroid_lon", "centroid_lat", "area_km2", "dist_cbd_km"]
    built_cols = [
        "n_buildings_log",
        "cap_proxy_sum_log",
        "height_mean",
        "footprint_mean_log",
    ] + [f"price_tier_p{k}" for k in range(1, int(args.n_tiers) + 1)]

    # Helper: build a tract->cond vector dict for a given fold+condition.
    def _cond_for_fold(condition: str, train_tracts: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        if condition == "none":
            return {}, {"cond_dim": 0, "cols": []}
        if condition == "marginal":
            cols = [f"p_age_{lab}" for (_lo, _hi, lab) in _age23_bins()] + ["p_sex_male", "p_sex_female"]
            out_map: dict[str, Any] = {}
            all_tracts = set(ctx["tract_geoid"].astype(str).tolist())
            for tg, t in targets_by_tract.items():
                if str(tg) not in all_tracts:
                    continue
                p_age = np.asarray(t["p_age"], dtype=np.float32).reshape(-1)
                p_sex = np.asarray(t["p_sex"], dtype=np.float32).reshape(-1)
                if p_age.size != 23 or p_sex.size != 2:
                    continue
                out_map[str(tg)] = np.concatenate([p_age, p_sex], axis=0).astype(np.float32)
            return out_map, {"cond_dim": 25, "cols": cols}
        if condition == "geo-only":
            cols = geo_cols
        elif condition == "built-only":
            cols = built_cols
        else:
            cols = geo_cols + built_cols

        sub = ctx[ctx["tract_geoid"].astype(str).isin(sorted(train_tracts))].copy()
        sub_train, mu, sd = _standardize(sub, cols=cols)
        # Apply the same scaler to all tracts (train+test).
        full, _, _ = _standardize(ctx, cols=cols, mean=mu, std=sd)
        full = full.set_index("tract_geoid", drop=False)
        out_map = {str(tg): full.loc[str(tg), cols].to_numpy(dtype=float) for tg in full.index.astype(str).tolist()}
        return out_map, {"cond_dim": int(len(cols)), "cols": cols, "mean": mu, "std": sd}

    # Collect fold-level summaries for ablation report.
    ablation_internal: dict[str, dict[int, Any]] = {c: {} for c in cond_list}
    ablation_external: dict[str, dict[int, Any]] = {c: {} for c in cond_list}
    ablation_baselines: dict[str, dict[int, Any]] = {"independence": {}, "ipf_train_seed": {}}
    # Collect per-tract TVDs (across folds) for simple global summaries + plots.
    per_tract_tvd: dict[str, dict[str, list[float]]] = {c: {"tvd_joint": [], "tvd_age": [], "tvd_sex": []} for c in cond_list}

    # Main loop.
    for fold_idx in fold_indices:
        test_pumas = set(blocks[int(fold_idx)])
        train_pumas = set(study_pumas) - set(test_pumas)
        train_tracts = set(ctx[ctx["puma"].astype(str).isin(sorted(train_pumas))]["tract_geoid"].astype(str).tolist())
        test_tracts = set(ctx[ctx["puma"].astype(str).isin(sorted(test_pumas))]["tract_geoid"].astype(str).tolist())
        if not train_tracts or not test_tracts:
            raise SystemExit(f"Empty train/test tracts in fold={fold_idx}. train={len(train_tracts)} test={len(test_tracts)}")

        # --- Baselines (fold-level, condition-agnostic): independence and IPF(train-seed) ---
        fold_root = out_root / f"fold_{fold_idx}"
        baseline_internal_path = fold_root / "metrics" / "baselines_internal.json"
        if args.resume and baseline_internal_path.exists():
            try:
                baselines = json.loads(baseline_internal_path.read_text(encoding="utf-8"))
                if isinstance(baselines, dict):
                    by_base = baselines.get("by_baseline", {})
                    if isinstance(by_base, dict):
                        for name in ["independence", "ipf_train_seed"]:
                            blk = by_base.get(name, {})
                            if isinstance(blk, dict) and isinstance(blk.get("summary"), dict):
                                ablation_baselines[name][int(fold_idx)] = dict(blk["summary"])
            except Exception:
                pass
        else:
            # Seed joint from TRAIN tracts only (ACS counts_joint).
            seed_counts = np.zeros((46,), dtype=float)
            for tg in sorted(train_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                seed_counts += np.asarray(t["counts_joint"], dtype=float).reshape(-1)
            seed_p = seed_counts / (float(seed_counts.sum()) if float(seed_counts.sum()) > 0 else 1.0)

            def _summarize_by_tract(by_tract: dict[str, Any]) -> dict[str, Any]:
                vals_joint = [float(v["tvd_joint"]) for v in by_tract.values() if isinstance(v, dict) and "tvd_joint" in v]
                vals_age = [float(v["tvd_age"]) for v in by_tract.values() if isinstance(v, dict) and "tvd_age" in v]
                vals_sex = [float(v["tvd_sex"]) for v in by_tract.values() if isinstance(v, dict) and "tvd_sex" in v]
                return {
                    "by_tract": by_tract,
                    "summary": {
                        "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint)), "p90": float(np.quantile(vals_joint, 0.9))} if vals_joint else None,
                        "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age)), "p90": float(np.quantile(vals_age, 0.9))} if vals_age else None,
                        "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex)), "p90": float(np.quantile(vals_sex, 0.9))} if vals_sex else None,
                    },
                }

            ind_by_tract: dict[str, Any] = {}
            ipf_by_tract: dict[str, Any] = {}
            for tg in sorted(test_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                p_age = np.asarray(t["p_age"], dtype=float)
                p_sex = np.asarray(t["p_sex"], dtype=float)
                p_true = np.asarray(t["p_joint"], dtype=float)

                p_ind = _joint_from_marginals(p_age=p_age, p_sex=p_sex)
                p_ipf = _ipf_age23_sex2(seed_joint=seed_p, target_p_age=p_age, target_p_sex=p_sex)

                ind_age, ind_sex = _marginals_from_joint(p_ind)
                ipf_age, ipf_sex = _marginals_from_joint(p_ipf)

                ind_by_tract[str(tg)] = {
                    "tvd_joint": float(_tvd(p_ind, p_true)),
                    "tvd_age": float(_tvd(ind_age, p_age)),
                    "tvd_sex": float(_tvd(ind_sex, p_sex)),
                }
                ipf_by_tract[str(tg)] = {
                    "tvd_joint": float(_tvd(p_ipf, p_true)),
                    "tvd_age": float(_tvd(ipf_age, p_age)),
                    "tvd_sex": float(_tvd(ipf_sex, p_sex)),
                }

            baselines_out = {
                "fold": int(fold_idx),
                "train_pumas": sorted(train_pumas),
                "test_pumas": sorted(test_pumas),
                "seed_joint": {"source": "train_tracts_acs_counts_joint", "p_joint": [float(x) for x in seed_p.tolist()]},
                "by_baseline": {
                    "independence": _summarize_by_tract(ind_by_tract),
                    "ipf_train_seed": _summarize_by_tract(ipf_by_tract),
                },
            }
            _write_json(baseline_internal_path, baselines_out)
            for name in ["independence", "ipf_train_seed"]:
                blk = baselines_out["by_baseline"].get(name, {})
                if isinstance(blk, dict) and isinstance(blk.get("summary"), dict):
                    ablation_baselines[name][int(fold_idx)] = dict(blk["summary"])

        for condition in cond_list:
            fold_dir = out_root / f"fold_{fold_idx}" / condition
            fold_dir.mkdir(parents=True, exist_ok=True)

            ckpt = fold_dir / "model.pt"
            train_summary_path = fold_dir / "train_summary.json"
            internal_path = fold_dir / "metrics" / "internal_acs_holdout.json"
            external_path = fold_dir / "metrics" / "external_pums_by_puma.json"

            if args.resume and internal_path.exists() and (pums_puma_dist is None or external_path.exists()):
                try:
                    internal = json.loads(internal_path.read_text(encoding="utf-8"))
                    if isinstance(internal, dict) and isinstance(internal.get("summary"), dict):
                        ablation_internal[condition][int(fold_idx)] = dict(internal["summary"])
                    by_tract = dict(internal.get("by_tract", {})) if isinstance(internal, dict) else {}
                    for v in by_tract.values():
                        if not isinstance(v, dict):
                            continue
                        for k in ["tvd_joint", "tvd_age", "tvd_sex"]:
                            if k in v:
                                per_tract_tvd[condition][k].append(float(v[k]))
                except Exception:
                    pass
                if pums_puma_dist is not None and external_path.exists():
                    try:
                        external = json.loads(external_path.read_text(encoding="utf-8"))
                        if isinstance(external, dict) and isinstance(external.get("summary"), dict):
                            ablation_external[condition][int(fold_idx)] = dict(external["summary"])
                    except Exception:
                        pass
                continue

            tract_cond, scaler = _cond_for_fold(condition, train_tracts=train_tracts)
            cond_dim = int(scaler["cond_dim"])

            x_model = str(args.x_model)
            model: Any = None
            x_mean: Any = None
            x_std: Any = None

            if args.resume and ckpt.exists() and train_summary_path.exists():
                train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
                saved_x_model = str(train_summary.get("x_model", "tabddpm_scalar"))
                if saved_x_model != x_model:
                    raise SystemExit(f"Checkpoint x_model mismatch in {train_summary_path}: saved={saved_x_model} vs args={x_model}")
                if x_model in {"tabddpm_scalar", "tabddpm_onehot", "joint_tabddpm_logp"}:
                    x_mean = np.asarray(train_summary.get("x_mean"), dtype=np.float32)
                    x_std = np.asarray(train_summary.get("x_std"), dtype=np.float32)
                    cfg = TabDDPMConfig(timesteps=int(args.timesteps))
                    model = DiffusionTabularModel(input_dim=int(x_mean.shape[0]), cond_dim=int(cond_dim), seed=int(args.seed), config=cfg)
                    model.load(ckpt)
                elif x_model == "cat_mlp":
                    model = _CategoricalMLPModel(cond_dim=int(cond_dim), seed=int(args.seed))
                    model.load(ckpt)
                elif x_model in {"cat_diffusion_concat", "cat_diffusion_xattn"}:
                    if x_model == "cat_diffusion_xattn" and int(cond_dim) <= 0:
                        raise SystemExit("x_model=cat_diffusion_xattn requires cond_dim>0 (pick a non-'none' condition).")
                    fusion = "cross_attn" if x_model == "cat_diffusion_xattn" else "concat"
                    cfg = CategoricalDiffusionConfig(timesteps=int(args.timesteps), lr=float(args.cat_diffusion_lr))
                    model = CategoricalDiffusionModel(n_classes=46, cond_dim=int(cond_dim), cond_fusion=fusion, seed=int(args.seed), config=cfg)
                    model.load(ckpt)
                    if int(getattr(model, "cond_dim", cond_dim)) != int(cond_dim):
                        raise SystemExit(f"Checkpoint cond_dim mismatch in {ckpt}: saved={getattr(model, 'cond_dim', None)} vs expected={cond_dim}")
                else:
                    raise SystemExit(f"Unknown x_model: {x_model}")
            else:
                # Build training dataset (pseudo individuals).
                xs: list[Any] = []
                ys: list[Any] = []
                cs: list[Any] = []
                weights = []
                for tg in sorted(train_tracts):
                    t = targets_by_tract.get(str(tg))
                    if t is None:
                        continue
                    total = float(t["total_pop"])
                    weights.append(math.sqrt(max(1.0, total)))
                w_mean = float(np.mean(weights)) if weights else 1.0

                for tg in sorted(train_tracts):
                    t = targets_by_tract.get(str(tg))
                    if t is None:
                        continue
                    total = float(t["total_pop"])
                    w = math.sqrt(max(1.0, total))
                    n_i = int(round(float(args.n_pseudo_base) * (w / w_mean)))
                    n_i = max(int(args.n_pseudo_min), min(int(args.n_pseudo_max), n_i))
                    if x_model == "joint_tabddpm_logp":
                        # Distribution-to-distribution training: one sample per tract (the full joint distribution).
                        p_joint = np.asarray(t["p_joint"], dtype=np.float32).reshape(-1)
                        if p_joint.size != 46:
                            continue
                        xs.append(np.log(np.clip(p_joint, 0.0, None) + 1e-6).reshape(1, 46).astype(np.float32))
                    else:
                        age_idx, sex01 = _sample_pseudo(rng=rng, p_joint=t["p_joint"], n=n_i)

                        if x_model == "tabddpm_scalar":
                            age_u = age_idx.astype(float) / 22.0
                            sex_u = sex01.astype(float)
                            x_u = np.stack([age_u, sex_u], axis=1).astype(np.float32)
                            xs.append(x_u)
                        elif x_model == "tabddpm_onehot":
                            xs.append(_encode_onehot(age_idx=age_idx, sex01=sex01).astype(np.float32))
                        elif x_model in {"cat_mlp", "cat_diffusion_concat", "cat_diffusion_xattn"}:
                            y = (sex01.astype(int) * 23 + age_idx.astype(int)).astype(np.int64)
                            ys.append(y)
                        else:
                            raise SystemExit(f"Unknown x_model: {x_model}")

                    if cond_dim > 0:
                        c = tract_cond.get(str(tg))
                        if c is None:
                            continue
                        c = np.asarray(c, dtype=np.float32)
                        if x_model == "joint_tabddpm_logp":
                            cs.append(c.reshape(1, -1))
                        else:
                            c_rep = np.repeat(c.reshape(1, -1), repeats=int(n_i), axis=0)
                            cs.append(c_rep)

                cond_all = np.concatenate(cs, axis=0).astype(np.float32) if cond_dim > 0 else None

                if x_model in {"tabddpm_scalar", "tabddpm_onehot", "joint_tabddpm_logp"}:
                    if not xs:
                        raise SystemExit(f"No training samples constructed for fold={fold_idx}, condition={condition}.")
                    x_u_all = np.concatenate(xs, axis=0).astype(np.float32)

                    # Standardize x_u (train-only).
                    x_mean = x_u_all.mean(axis=0).astype(np.float32)
                    x_std = x_u_all.std(axis=0).astype(np.float32)
                    x_std = np.where(x_std <= 1e-6, 1.0, x_std).astype(np.float32)
                    x_z = ((x_u_all - x_mean) / x_std).astype(np.float32)

                    x = torch.from_numpy(x_z)
                    cond = torch.from_numpy(cond_all) if cond_all is not None else None

                    cfg = TabDDPMConfig(timesteps=int(args.timesteps))
                    model = DiffusionTabularModel(
                        input_dim=int(x.shape[1]),
                        cond_dim=int(cond.shape[1]) if cond is not None else 0,
                        seed=int(args.seed),
                        config=cfg,
                    )
                    train_metrics = model.fit(
                        x=x,
                        cond=cond,
                        epochs=int(args.epochs),
                        batch_size=int(args.batch_size),
                        device=args.device,
                        log_every=int(args.log_every),
                    )
                    model.save(ckpt)
                    n_train_samples = int(x.shape[0])
                else:
                    if not ys:
                        raise SystemExit(f"No training samples constructed for fold={fold_idx}, condition={condition}.")
                    y_all = np.concatenate(ys, axis=0).astype(np.int64)
                    y_t = torch.from_numpy(y_all)
                    if x_model == "cat_mlp":
                        if cond_all is None:
                            cond_t = torch.zeros((int(y_t.shape[0]), 0), dtype=torch.float32)
                        else:
                            cond_t = torch.from_numpy(cond_all)
                        model = _CategoricalMLPModel(cond_dim=int(cond_dim), seed=int(args.seed))
                        train_metrics = model.fit(
                            y=y_t,
                            cond=cond_t,
                            epochs=int(args.epochs),
                            batch_size=int(args.batch_size),
                            device=args.device,
                            lr=float(args.cat_mlp_lr),
                            log_every=int(args.log_every),
                        )
                        model.save(ckpt)
                        n_train_samples = int(y_t.shape[0])
                    elif x_model in {"cat_diffusion_concat", "cat_diffusion_xattn"}:
                        if x_model == "cat_diffusion_xattn" and int(cond_dim) <= 0:
                            raise SystemExit("x_model=cat_diffusion_xattn requires cond_dim>0 (pick a non-'none' condition).")
                        fusion = "cross_attn" if x_model == "cat_diffusion_xattn" else "concat"
                        cfg = CategoricalDiffusionConfig(timesteps=int(args.timesteps), lr=float(args.cat_diffusion_lr))
                        model = CategoricalDiffusionModel(n_classes=46, cond_dim=int(cond_dim), cond_fusion=fusion, seed=int(args.seed), config=cfg)
                        cond_t = torch.from_numpy(cond_all) if cond_all is not None else None
                        train_metrics = model.fit(
                            x0=y_t,
                            cond=cond_t,
                            epochs=int(args.epochs),
                            batch_size=int(args.batch_size),
                            device=args.device,
                            log_every=int(args.log_every),
                        )
                        model.save(ckpt)
                        n_train_samples = int(y_t.shape[0])
                    else:
                        raise SystemExit(f"Unknown x_model: {x_model}")

                train_summary = {
                    "fold": int(fold_idx),
                    "condition": condition,
                    "x_model": x_model,
                    "train_pumas": sorted(train_pumas),
                    "test_pumas": sorted(test_pumas),
                    "n_train_tracts": int(len(train_tracts)),
                    "n_test_tracts": int(len(test_tracts)),
                    "n_train_samples": int(n_train_samples),
                    "cond_dim": cond_dim,
                    "cond_cols": scaler.get("cols", []),
                    "train_metrics": train_metrics,
                    "ckpt": str(ckpt),
                }
                if x_mean is not None and x_std is not None:
                    train_summary["x_mean"] = [float(v) for v in np.asarray(x_mean).tolist()]
                    train_summary["x_std"] = [float(v) for v in np.asarray(x_std).tolist()]
                if x_model == "cat_mlp":
                    train_summary["cat_mlp_lr"] = float(args.cat_mlp_lr)
                if x_model in {"cat_diffusion_concat", "cat_diffusion_xattn"}:
                    train_summary["cat_diffusion_lr"] = float(args.cat_diffusion_lr)
                    train_summary["timesteps"] = int(args.timesteps)
                if x_model == "joint_tabddpm_logp":
                    train_summary["joint_space"] = "logp_softmax"
                _write_json(train_summary_path, train_summary)

            # --- Internal evaluation vs ACS on held-out tracts ---
            internal_by_tract: dict[str, Any] = {}
            diag_age_pred = np.zeros((23,), dtype=float)
            diag_sex_pred = np.zeros((2,), dtype=float)
            diag_age_tgt = np.zeros((23,), dtype=float)
            diag_sex_tgt = np.zeros((2,), dtype=float)
            age_u_hist = None
            if x_model == "tabddpm_scalar":
                age_u_hist = {"bins": 50, "hist": np.zeros((50,), dtype=float).tolist()}
            for tg in sorted(test_tracts):
                t = targets_by_tract.get(str(tg))
                if t is None:
                    continue
                n_eval = int(args.n_eval_joint_samples) if x_model == "joint_tabddpm_logp" else int(args.n_eval_per_tract)
                if cond_dim > 0:
                    c = tract_cond.get(str(tg))
                    if c is None:
                        continue
                    c = np.asarray(c, dtype=np.float32)
                    c_rep = np.repeat(c.reshape(1, -1), repeats=n_eval, axis=0)
                    c_t = torch.from_numpy(c_rep)
                else:
                    if x_model == "cat_mlp":
                        c_t = torch.zeros((n_eval, 0), dtype=torch.float32)
                    else:
                        c_t = None

                if x_model in {"tabddpm_scalar", "tabddpm_onehot"}:
                    if x_mean is None or x_std is None:
                        raise RuntimeError("Missing x_mean/x_std for tabddpm evaluation.")
                    z = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(np.float32)
                    x_u = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                    if x_model == "tabddpm_scalar":
                        age_idx, sex01 = _decode_samples(x_u)
                        # Scalar-specific histogram diagnostic on age_u (post de-standardize).
                        bins = int(age_u_hist["bins"]) if isinstance(age_u_hist, dict) else 50
                        h, _ = np.histogram(np.clip(x_u[:, 0], 0.0, 1.0), bins=bins, range=(0.0, 1.0))
                        age_u_hist["hist"] = (np.asarray(age_u_hist["hist"], dtype=float) + h.astype(float)).tolist()
                    else:
                        age_idx, sex01 = _decode_onehot(x_u)
                elif x_model == "cat_mlp":
                    y = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(int)
                    sex01 = (y // 23).astype(int)
                    age_idx = (y % 23).astype(int)
                elif x_model in {"cat_diffusion_concat", "cat_diffusion_xattn"}:
                    y = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(int)
                    sex01 = (y // 23).astype(int)
                    age_idx = (y % 23).astype(int)
                elif x_model == "joint_tabddpm_logp":
                    if x_mean is None or x_std is None:
                        raise RuntimeError("Missing x_mean/x_std for joint_tabddpm_logp evaluation.")
                    z = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(np.float32)
                    logp = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                    logp = logp - logp.max(axis=1, keepdims=True)
                    p = np.exp(logp)
                    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                    p_joint_raw = p.mean(axis=0).astype(float)
                    # If marginals are available, project to match exactly (guidance/projection).
                    p_joint = p_joint_raw
                    if condition == "marginal":
                        p_joint = _ipf_age23_sex2(seed_joint=p_joint_raw, target_p_age=t["p_age"], target_p_sex=t["p_sex"])
                    p_age_hat, p_sex_hat = _marginals_from_joint(p_joint)
                    tvd_joint_raw = _tvd(p_joint_raw, t["p_joint"])
                    tvd_age_raw = _tvd(_marginals_from_joint(p_joint_raw)[0], t["p_age"])
                    tvd_sex_raw = _tvd(_marginals_from_joint(p_joint_raw)[1], t["p_sex"])
                    internal_by_tract[str(tg)] = {
                        "tvd_joint": float(_tvd(p_joint, t["p_joint"])),
                        "tvd_age": float(_tvd(p_age_hat, t["p_age"])),
                        "tvd_sex": float(_tvd(p_sex_hat, t["p_sex"])),
                        "tvd_joint_raw": float(tvd_joint_raw),
                        "tvd_age_raw": float(tvd_age_raw),
                        "tvd_sex_raw": float(tvd_sex_raw),
                        "projected_to_marginal": bool(condition == "marginal"),
                    }
                    diag_age_pred += np.asarray(p_age_hat, dtype=float) * float(n_eval)
                    diag_sex_pred += np.asarray(p_sex_hat, dtype=float) * float(n_eval)
                    diag_age_tgt += np.asarray(t["p_age"], dtype=float) * float(n_eval)
                    diag_sex_tgt += np.asarray(t["p_sex"], dtype=float) * float(n_eval)
                    continue
                else:
                    raise SystemExit(f"Unknown x_model: {x_model}")

                phat = _p_from_samples(age_idx=age_idx, sex01=sex01)
                tvd_joint = _tvd(phat["p_joint"], t["p_joint"])
                tvd_age = _tvd(phat["p_age"], t["p_age"])
                tvd_sex = _tvd(phat["p_sex"], t["p_sex"])
                internal_by_tract[str(tg)] = {"tvd_joint": float(tvd_joint), "tvd_age": float(tvd_age), "tvd_sex": float(tvd_sex)}
                diag_age_pred += np.asarray(phat["p_age"], dtype=float) * float(n_eval)
                diag_sex_pred += np.asarray(phat["p_sex"], dtype=float) * float(n_eval)
                diag_age_tgt += np.asarray(t["p_age"], dtype=float) * float(n_eval)
                diag_sex_tgt += np.asarray(t["p_sex"], dtype=float) * float(n_eval)

            vals_joint = [v["tvd_joint"] for v in internal_by_tract.values()]
            vals_age = [v["tvd_age"] for v in internal_by_tract.values()]
            vals_sex = [v["tvd_sex"] for v in internal_by_tract.values()]
            internal = {
                "by_tract": internal_by_tract,
                "summary": {
                    "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint)), "p90": float(np.quantile(vals_joint, 0.9))},
                    "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age)), "p90": float(np.quantile(vals_age, 0.9))},
                    "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex)), "p90": float(np.quantile(vals_sex, 0.9))},
                },
            }
            _write_json(fold_dir / "metrics" / "internal_acs_holdout.json", internal)
            # Sampling distribution diagnostics (aggregate over held-out tracts).
            diag = {
                "x_model": x_model,
                "condition": condition,
                "fold": int(fold_idx),
                "n_eval_per_tract": int(args.n_eval_per_tract),
                "pred": {
                    "p_age": (diag_age_pred / float(diag_age_pred.sum()) if float(diag_age_pred.sum()) > 0 else np.full((23,), 1.0 / 23.0)).tolist(),
                    "p_sex": (diag_sex_pred / float(diag_sex_pred.sum()) if float(diag_sex_pred.sum()) > 0 else np.full((2,), 0.5)).tolist(),
                },
                "target": {
                    "p_age": (diag_age_tgt / float(diag_age_tgt.sum()) if float(diag_age_tgt.sum()) > 0 else np.full((23,), 1.0 / 23.0)).tolist(),
                    "p_sex": (diag_sex_tgt / float(diag_sex_tgt.sum()) if float(diag_sex_tgt.sum()) > 0 else np.full((2,), 0.5)).tolist(),
                },
            }
            if age_u_hist is not None:
                diag["age_u_hist"] = age_u_hist
            _write_json(fold_dir / "metrics" / "sampling_diagnostics.json", diag)
            ablation_internal[condition][int(fold_idx)] = dict(internal.get("summary", {}))
            per_tract_tvd[condition]["tvd_joint"].extend([float(v["tvd_joint"]) for v in internal_by_tract.values()])
            per_tract_tvd[condition]["tvd_age"].extend([float(v["tvd_age"]) for v in internal_by_tract.values()])
            per_tract_tvd[condition]["tvd_sex"].extend([float(v["tvd_sex"]) for v in internal_by_tract.values()])

            # Worst tracts diagnosis (top 10 by joint TVD).
            worst = sorted(internal_by_tract.items(), key=lambda kv: kv[1]["tvd_joint"], reverse=True)[:10]
            _write_json(
                fold_dir / "metrics" / "worst_tracts_diagnosis.json",
                {"worst_tracts": [{"tract_geoid": k, **v} for k, v in worst]},
            )

            # --- External evaluation vs PUMS at PUMA level ---
            if pums_puma_dist is not None:
                # First estimate p_hat for each tract (reuse internal estimates if tract in test; otherwise sample now).
                p_hat_by_tract: dict[str, Any] = {}
                for tg in sorted(set(ctx["tract_geoid"].astype(str).tolist())):
                    t = targets_by_tract.get(str(tg))
                    if t is None:
                        continue
                    n_eval = int(args.n_eval_joint_samples) if x_model == "joint_tabddpm_logp" else int(args.n_eval_per_tract)
                    if cond_dim > 0:
                        c = tract_cond.get(str(tg))
                        if c is None:
                            continue
                        c = np.asarray(c, dtype=np.float32)
                        c_rep = np.repeat(c.reshape(1, -1), repeats=n_eval, axis=0)
                        c_t = torch.from_numpy(c_rep)
                    else:
                        if x_model == "cat_mlp":
                            c_t = torch.zeros((n_eval, 0), dtype=torch.float32)
                        else:
                            c_t = None

                    if x_model in {"tabddpm_scalar", "tabddpm_onehot"}:
                        if x_mean is None or x_std is None:
                            raise RuntimeError("Missing x_mean/x_std for tabddpm evaluation.")
                        z = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(np.float32)
                        x_u = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                        if x_model == "tabddpm_scalar":
                            age_idx, sex01 = _decode_samples(x_u)
                        else:
                            age_idx, sex01 = _decode_onehot(x_u)
                    elif x_model == "cat_mlp":
                        y = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(int)
                        sex01 = (y // 23).astype(int)
                        age_idx = (y % 23).astype(int)
                    elif x_model in {"cat_diffusion_concat", "cat_diffusion_xattn"}:
                        y = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(int)
                        sex01 = (y // 23).astype(int)
                        age_idx = (y % 23).astype(int)
                    elif x_model == "joint_tabddpm_logp":
                        if x_mean is None or x_std is None:
                            raise RuntimeError("Missing x_mean/x_std for joint_tabddpm_logp evaluation.")
                        z = model.sample(n=n_eval, cond=c_t, device=args.device).numpy().astype(np.float32)
                        logp = (z * x_std.reshape(1, -1) + x_mean.reshape(1, -1)).astype(np.float32)
                        logp = logp - logp.max(axis=1, keepdims=True)
                        p = np.exp(logp)
                        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
                        p_joint_raw = p.mean(axis=0).astype(float)
                        p_joint = p_joint_raw
                        if condition == "marginal":
                            p_joint = _ipf_age23_sex2(seed_joint=p_joint_raw, target_p_age=t["p_age"], target_p_sex=t["p_sex"])
                        p_age_hat, p_sex_hat = _marginals_from_joint(p_joint)
                        p_hat_by_tract[str(tg)] = {"p_joint": p_joint, "p_age": p_age_hat, "p_sex": p_sex_hat}
                        continue
                    else:
                        raise SystemExit(f"Unknown x_model: {x_model}")
                    phat = _p_from_samples(age_idx=age_idx, sex01=sex01)
                    p_hat_by_tract[str(tg)] = phat

                # Aggregate to PUMA using ACS tract populations.
                counts_hat_by_puma: dict[str, Any] = {p: np.zeros((46,), dtype=float) for p in study_pumas}
                for tg, phat in p_hat_by_tract.items():
                    puma = tract_to_puma.get(str(tg))
                    t = targets_by_tract.get(str(tg))
                    if not puma or t is None:
                        continue
                    pop = float(t["total_pop"])
                    counts_hat_by_puma[str(puma)] += float(pop) * np.asarray(phat["p_joint"], dtype=float)

                external_by_puma: dict[str, Any] = {}
                for puma in study_pumas:
                    hat = counts_hat_by_puma[str(puma)]
                    hat_p = hat / (hat.sum() if hat.sum() > 0 else 1.0)
                    ref_p = np.asarray(pums_puma_dist[str(puma)]["p_joint"], dtype=float)
                    hat_age, hat_sex = _marginals_from_joint(hat_p)
                    ref_age, ref_sex = _marginals_from_joint(ref_p)
                    external_by_puma[str(puma)] = {
                        "tvd_joint": float(_tvd(hat_p, ref_p)),
                        "tvd_age": float(_tvd(hat_age, ref_age)),
                        "tvd_sex": float(_tvd(hat_sex, ref_sex)),
                    }

                vals_joint = [v["tvd_joint"] for v in external_by_puma.values()]
                vals_age = [v["tvd_age"] for v in external_by_puma.values()]
                vals_sex = [v["tvd_sex"] for v in external_by_puma.values()]
                external = {
                    "by_puma": external_by_puma,
                    "summary": {
                        "tvd_joint": {"mean": float(np.mean(vals_joint)), "max": float(np.max(vals_joint))} if vals_joint else None,
                        "tvd_age": {"mean": float(np.mean(vals_age)), "max": float(np.max(vals_age))} if vals_age else None,
                        "tvd_sex": {"mean": float(np.mean(vals_sex)), "max": float(np.max(vals_sex))} if vals_sex else None,
                    },
                }
                _write_json(fold_dir / "metrics" / "external_pums_by_puma.json", external)
                ablation_external[condition][int(fold_idx)] = dict(external.get("summary", {}))

    # --- Write ablation summary (mean±std across folds) ---
    def _mean_std(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

    def _summarize_across_folds(per_fold: dict[int, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"by_fold": {str(k): v for k, v in sorted(per_fold.items())}}
        for metric in ["tvd_joint", "tvd_age", "tvd_sex"]:
            mean_vals = [float(v.get(metric, {}).get("mean")) for v in per_fold.values() if v.get(metric) and v[metric].get("mean") is not None]
            max_vals = [float(v.get(metric, {}).get("max")) for v in per_fold.values() if v.get(metric) and v[metric].get("max") is not None]
            p90_vals = [float(v.get(metric, {}).get("p90")) for v in per_fold.values() if v.get(metric) and v[metric].get("p90") is not None]
            out[metric] = {
                "mean": _mean_std(mean_vals),
                "max": _mean_std(max_vals),
            }
            if p90_vals:
                out[metric]["p90"] = _mean_std(p90_vals)
        return out

    ablation_summary: dict[str, Any] = {
        "folds": [int(i) for i in fold_indices],
        "conditions": cond_list,
        "internal_acs": {c: _summarize_across_folds(ablation_internal.get(c, {})) for c in cond_list},
        "external_pums": {c: _summarize_across_folds(ablation_external.get(c, {})) for c in cond_list},
        "baselines_internal": {b: _summarize_across_folds(ablation_baselines.get(b, {})) for b in sorted(ablation_baselines)},
        "baseline_gap": baseline_gap,
    }
    _write_json(out_root / "metrics" / "ablation_summary.json", ablation_summary)
    _write_json(
        out_root / "metrics" / "baselines_internal.json",
        {
            "by_baseline": {b: _summarize_across_folds(ablation_baselines.get(b, {})) for b in sorted(ablation_baselines)},
            "by_fold": {b: {str(k): v for k, v in sorted(ablation_baselines.get(b, {}).items())} for b in sorted(ablation_baselines)},
        },
    )

    # --- Write "spec-friendly" internal/external summaries (single files) ---
    def _global_summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(arr.mean()),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    internal_by_condition: dict[str, Any] = {}
    for c in cond_list:
        internal_by_condition[c] = {
            "tvd_joint": _global_summary(per_tract_tvd[c]["tvd_joint"]),
            "tvd_age": _global_summary(per_tract_tvd[c]["tvd_age"]),
            "tvd_sex": _global_summary(per_tract_tvd[c]["tvd_sex"]),
        }
    _write_json(
        out_root / "metrics" / "internal_acs_holdout.json",
        {
            "by_condition": internal_by_condition,
            "by_fold": {c: {str(k): v for k, v in sorted(ablation_internal.get(c, {}).items())} for c in cond_list},
        },
    )
    if pums_puma_dist is not None:
        _write_json(
            out_root / "metrics" / "external_pums_by_puma.json",
            {
                "by_condition": {c: _summarize_across_folds(ablation_external.get(c, {})) for c in cond_list},
                "baseline_gap": baseline_gap,
            },
        )

    # --- Simple figure: tract-level TVD boxplot by condition ---
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig_dir = out_root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        order = [c for c in ["none", "marginal", "geo-only", "built-only", "geo+built"] if c in cond_list]
        data = [per_tract_tvd[c]["tvd_joint"] for c in order]
        if any(len(x) > 0 for x in data):
            plt.figure(figsize=(10, 4))
            plt.boxplot(data, labels=order, showfliers=False)
            plt.ylabel("TVD (joint age×sex) on held-out tracts")
            plt.title("ACS-supervised tract conditional diffusion (4-fold CV)")
            plt.tight_layout()
            plt.savefig(fig_dir / "tvd_by_condition.png", dpi=200)
            plt.close()
    except Exception:
        pass

    print(f"[ok] wrote: {out_root}")


if __name__ == "__main__":
    main()

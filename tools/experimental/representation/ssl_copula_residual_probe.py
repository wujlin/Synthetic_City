#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.model.external_c2f_full_earn_schema import (
    CHILD_PARENT_INDEX_FULL,
    COARSE_K,
    COARSE_SHAPE,
    FULL_CATEGORIES,
    FULL_SHAPE,
    FULL_VARIABLE_ORDER,
)


PREFIX_BY_VARIABLE = {
    "AGEP_bin": "p_age",
    "SEX": "p_sex",
    "SCHL_allpop": "p_schl",
    "ESR_allpop": "p_esr",
    "EARN_16p_bin": "p_earn",
}


def _utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _canon_statefp(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(int(float(x))).zfill(2) if str(x).replace(".", "", 1).isdigit() else str(x).zfill(2)


def _canon_puma5(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(int(float(x))).zfill(5) if str(x).replace(".", "", 1).isdigit() else str(x).zfill(5)


def _add_puma_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"statefp", "puma"}.issubset(out.columns):
        out["statefp"] = out["statefp"].map(_canon_statefp)
        out["puma5"] = out["puma"].map(_canon_puma5)
        out["puma_uid_key"] = out["statefp"] + out["puma5"]
    elif {"statefp", "puma5"}.issubset(out.columns):
        out["statefp"] = out["statefp"].map(_canon_statefp)
        out["puma5"] = out["puma5"].map(_canon_puma5)
        out["puma_uid_key"] = out["statefp"] + out["puma5"]
    else:
        raw = out["puma_uid"].astype(str).str.replace(r"\.0$", "", regex=True)
        out["puma_uid_key"] = raw.str.zfill(7)
        out["statefp"] = out["puma_uid_key"].str[:2]
        out["puma5"] = out["puma_uid_key"].str[2:]
    return out


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    sums = x.sum(axis=1, keepdims=True)
    fallback = np.full_like(x, 1.0 / max(x.shape[1], 1))
    return np.where(sums > 0.0, x / np.clip(sums, 1e-12, None), fallback)


def _outer_joint(marginals: list[np.ndarray]) -> np.ndarray:
    age, sex, schl, esr, earn = [_normalize_rows(m) for m in marginals]
    out = (
        age[:, :, None, None, None, None]
        * sex[:, None, :, None, None, None]
        * schl[:, None, None, :, None, None]
        * esr[:, None, None, None, :, None]
        * earn[:, None, None, None, None, :]
    )
    return out.reshape((age.shape[0], -1))


def _load_target(path: pathlib.Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    df = _add_puma_uid(pd.read_csv(path, low_memory=False))
    df = df.sort_values("puma_uid_key").reset_index(drop=True)
    joint_cols = sorted(
        [c for c in df.columns if c.startswith("p_joint_")],
        key=lambda c: int(re.search(r"(\d+)$", c).group(1)) if re.search(r"(\d+)$", c) else -1,
    )
    expected_k = int(np.prod(FULL_SHAPE))
    if len(joint_cols) != expected_k:
        raise SystemExit(f"expected {expected_k} joint columns, got {len(joint_cols)}")
    p_true = df[joint_cols].to_numpy(dtype=np.float64)
    p_true = p_true / np.clip(p_true.sum(axis=1, keepdims=True), 1e-12, None)

    target_marginals: dict[str, np.ndarray] = {}
    for var in FULL_VARIABLE_ORDER:
        prefix = PREFIX_BY_VARIABLE[var]
        cols = [f"{prefix}_{i:02d}" for i in range(len(FULL_CATEGORIES[var]))]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"target missing marginal columns for {var}: {missing}")
        target_marginals[var] = _normalize_rows(df[cols].to_numpy(dtype=np.float64))
    p_eq_target = _outer_joint([target_marginals[v] for v in FULL_VARIABLE_ORDER])
    return df, p_true, p_eq_target, target_marginals


def _load_acs_conditions(
    path: pathlib.Path,
    target_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    cond = _add_puma_uid(pd.read_csv(path, low_memory=False))
    cond["target"] = pd.to_numeric(cond["target"], errors="coerce").fillna(0.0).clip(lower=0.0)
    grouped = cond.groupby(["puma_uid_key", "variable", "category"], sort=False)["target"].sum().reset_index()

    feature_frames: list[pd.DataFrame] = []
    scale_frames: list[pd.DataFrame] = []
    one_d_cols: list[str] = []
    all_cols: list[str] = []
    scale_cols: list[str] = []
    acs_marginals: dict[str, np.ndarray] = {}

    for var in FULL_VARIABLE_ORDER:
        cats = list(FULL_CATEGORIES[var])
        sub = grouped[grouped["variable"] == var]
        wide = sub.pivot(index="puma_uid_key", columns="category", values="target")
        wide = wide.reindex(index=target_keys["puma_uid_key"], columns=cats).fillna(0.0)
        raw = wide.to_numpy(dtype=np.float64)
        total = np.clip(raw.sum(axis=1), 0.0, None)
        vals = _normalize_rows(raw)
        acs_marginals[var] = vals
        cols = [f"acs__{var}__{cat}" for cat in cats]
        feature_frames.append(pd.DataFrame(vals, columns=cols, index=target_keys["puma_uid_key"]))
        one_d_cols.extend(cols)
        all_cols.extend(cols)
        scale_col = f"scale__log1p_total__{var}"
        scale_frames.append(pd.DataFrame({scale_col: np.log1p(total)}, index=target_keys["puma_uid_key"]))
        scale_cols.append(scale_col)

    # AGEP_SEX_cross is observed by ACS and useful as an inference-time view, but it is not
    # included in the one-dimensional max-ent reference used in this first diagnostic.
    if "AGEP_SEX_cross" in set(grouped["variable"].astype(str)):
        sub = grouped[grouped["variable"] == "AGEP_SEX_cross"]
        cats = sorted(sub["category"].astype(str).unique().tolist())
        wide = sub.pivot(index="puma_uid_key", columns="category", values="target")
        wide = wide.reindex(index=target_keys["puma_uid_key"], columns=cats).fillna(0.0)
        raw = wide.to_numpy(dtype=np.float64)
        total = np.clip(raw.sum(axis=1), 0.0, None)
        vals = _normalize_rows(raw)
        cols = [f"acs__AGEP_SEX_cross__{cat}" for cat in cats]
        feature_frames.append(pd.DataFrame(vals, columns=cols, index=target_keys["puma_uid_key"]))
        all_cols.extend(cols)
        scale_col = "scale__log1p_total__AGEP_SEX_cross"
        scale_frames.append(pd.DataFrame({scale_col: np.log1p(total)}, index=target_keys["puma_uid_key"]))
        scale_cols.append(scale_col)

    feat = pd.concat(feature_frames, axis=1).reset_index(drop=True)
    scale_feat = pd.concat(scale_frames, axis=1).reset_index(drop=True)
    x_1d = feat[one_d_cols].to_numpy(dtype=np.float64)
    x_all = feat[all_cols].to_numpy(dtype=np.float64)
    x_scale = scale_feat[scale_cols].to_numpy(dtype=np.float64)
    return feat, acs_marginals, x_1d, x_all, x_scale, one_d_cols, all_cols, scale_cols


def _load_spatial(path: pathlib.Path, target_keys: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    spatial = _add_puma_uid(pd.read_csv(path, low_memory=False))
    spatial = spatial.drop_duplicates("puma_uid_key").set_index("puma_uid_key")
    numeric_cols = [
        c
        for c in spatial.columns
        if c not in {"statefp", "puma", "puma5", "puma_uid", "puma_uid_key"}
        and pd.api.types.is_numeric_dtype(spatial[c])
    ]
    aligned = spatial.reindex(index=target_keys["puma_uid_key"])[numeric_cols]
    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    med = aligned.median(numeric_only=True)
    aligned = aligned.fillna(med).fillna(0.0)
    return aligned.to_numpy(dtype=np.float64), [f"spatial__{c}" for c in numeric_cols]


def _residual_arrays(p_true: np.ndarray, p_eq: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_eq = p_eq / np.clip(p_eq.sum(axis=1, keepdims=True), 1e-12, None)
    tvd = 0.5 * np.abs(p_true - p_eq).sum(axis=1)
    kl = np.sum(p_true * (np.log(p_true + eps) - np.log(p_eq + eps)), axis=1)
    log_ratio = np.log(p_true + eps) - np.log(p_eq + eps)
    return tvd, kl, log_ratio


def _aggregate_full_to_coarse(p_full: np.ndarray) -> np.ndarray:
    p_full = np.asarray(p_full, dtype=np.float64)
    out = np.zeros((p_full.shape[0], int(COARSE_K)), dtype=np.float64)
    for child_idx, parent_idx in enumerate(CHILD_PARENT_INDEX_FULL.astype(int).tolist()):
        out[:, int(parent_idx)] += p_full[:, int(child_idx)]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def _summary(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float).reshape(-1)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.quantile(x, 0.5)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _fit_pc_scores(z: np.ndarray, train_mask: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    n_train = int(train_mask.sum())
    n_components = max(1, min(int(n_components), n_train - 1, z.shape[1]))
    scaler = StandardScaler(with_mean=True, with_std=False)
    z_train = scaler.fit_transform(z[train_mask])
    z_all = scaler.transform(z)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    scores_train = pca.fit_transform(z_train)
    scores_all = pca.transform(z_all)
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    return scores_all, ratios, float(ratios.sum())


def _ridge_probe(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    alphas: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    weighted_nonneg_r2 = 0.0
    weighted_raw_r2 = 0.0
    positive_pc_share = 0.0
    top5_r2: list[float] = []
    for pc_idx in range(scores.shape[1]):
        y = scores[:, pc_idx]
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas, dtype=float)))
        model.fit(x[train_mask], y[train_mask])
        pred_train = model.predict(x[train_mask])
        pred_test = model.predict(x[~train_mask])
        r2_train = float(r2_score(y[train_mask], pred_train))
        r2_test = float(r2_score(y[~train_mask], pred_test))
        evr = float(explained_ratio[pc_idx])
        weighted_nonneg_r2 += max(0.0, r2_test) * evr
        weighted_raw_r2 += r2_test * evr
        if r2_test > 0.0:
            positive_pc_share += evr
        if pc_idx < 5:
            top5_r2.append(r2_test)
        ridge_step = model.named_steps["ridgecv"]
        rows.append(
            {
                "pc_index": int(pc_idx + 1),
                "explained_variance_ratio": evr,
                "ridge_alpha": float(ridge_step.alpha_),
                "r2_train": r2_train,
                "r2_test": r2_test,
            }
        )
    summary = {
        "n_pcs": int(scores.shape[1]),
        "weighted_nonnegative_r2": float(weighted_nonneg_r2),
        "weighted_raw_r2": float(weighted_raw_r2),
        "positive_pc_explained_share": float(positive_pc_share),
        "mean_r2_top5": float(np.mean(top5_r2)) if top5_r2 else float("nan"),
        "median_r2_all": float(np.median([r["r2_test"] for r in rows])) if rows else float("nan"),
    }
    return summary, rows


def _mlp_direct_probe(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    *,
    seeds: list[int],
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    device: str,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int, d_out: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed),
                nn.ReLU(),
                nn.Linear(embed, d_out),
            )

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.net(x_in)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_test = x_scaler.transform(x[~train_mask]).astype(np.float32)
    y_train_raw = scores[train_mask].astype(np.float64)
    y_test_raw = scores[~train_mask].astype(np.float64)
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)

    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=dev)
    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    weights = np.asarray(explained_ratio, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights_t = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=dev).reshape(1, -1)

    out: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        model = Model(
            d_in=x_train.shape[1],
            hidden=int(hidden_dim),
            embed=int(embed_dim),
            d_out=scores.shape[1],
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        for _ in range(int(epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                pred = model(x_train_t[idx])
                loss = torch.mean(((pred - y_train_t[idx]) ** 2) * weights_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred_train = model(x_train_t).cpu().numpy().astype(np.float64) * y_std + y_mean
            pred_test = model(x_test_t).cpu().numpy().astype(np.float64) * y_std + y_mean

        rows: list[dict[str, Any]] = []
        weighted_nonneg = 0.0
        weighted_raw = 0.0
        positive_share = 0.0
        top5: list[float] = []
        for pc_idx in range(scores.shape[1]):
            r2_train = float(r2_score(y_train_raw[:, pc_idx], pred_train[:, pc_idx]))
            r2_test = float(r2_score(y_test_raw[:, pc_idx], pred_test[:, pc_idx]))
            evr = float(explained_ratio[pc_idx])
            weighted_nonneg += max(0.0, r2_test) * evr
            weighted_raw += r2_test * evr
            if r2_test > 0.0:
                positive_share += evr
            if pc_idx < 5:
                top5.append(r2_test)
            rows.append(
                {
                    "pc_index": int(pc_idx + 1),
                    "explained_variance_ratio": evr,
                    "ridge_alpha": float("nan"),
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                }
            )
        summary = {
            "n_pcs": int(scores.shape[1]),
            "weighted_nonnegative_r2": float(weighted_nonneg),
            "weighted_raw_r2": float(weighted_raw),
            "positive_pc_explained_share": float(positive_share),
            "mean_r2_top5": float(np.mean(top5)) if top5 else float("nan"),
            "median_r2_all": float(np.median([r["r2_test"] for r in rows])) if rows else float("nan"),
        }
        out.append((f"seed{seed}", summary, rows))
    return out


def _mlp_heteroskedastic_probe(
    x: np.ndarray,
    scores: np.ndarray,
    train_mask: np.ndarray,
    explained_ratio: np.ndarray,
    *,
    seeds: list[int],
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    batch_size: int,
    device: str,
) -> list[tuple[str, dict[str, Any], list[dict[str, Any]]]]:
    """Predict residual PC scores with a heteroskedastic Gaussian loss.

    The model predicts both the standardized PC mean and log-variance. This is
    useful when the feature set includes survey reliability signals: uncertain
    regions can be assigned larger residual variance instead of forcing all
    regions into an equal-MSE objective.
    """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int, d_out: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(embed, d_out)
            self.logvar_head = nn.Linear(embed, d_out)

        def forward(self, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(x_in)
            return self.mean_head(h), self.logvar_head(h)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_test = x_scaler.transform(x[~train_mask]).astype(np.float32)
    y_train_raw = scores[train_mask].astype(np.float64)
    y_test_raw = scores[~train_mask].astype(np.float64)
    y_mean = y_train_raw.mean(axis=0, keepdims=True)
    y_std = y_train_raw.std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test_raw - y_mean) / y_std).astype(np.float32)

    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=dev)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=dev)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=dev)
    n_train = int(x_train.shape[0])
    batch_size = min(int(batch_size), n_train)
    weights = np.asarray(explained_ratio, dtype=np.float64)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights_t = torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=dev).reshape(1, -1)

    out: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        model = Model(
            d_in=x_train.shape[1],
            hidden=int(hidden_dim),
            embed=int(embed_dim),
            d_out=scores.shape[1],
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        for _ in range(int(epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                pred_mean, pred_logvar = model(x_train_t[idx])
                pred_logvar = torch.clamp(pred_logvar, min=-6.0, max=4.0)
                nll = 0.5 * (((pred_mean - y_train_t[idx]) ** 2) * torch.exp(-pred_logvar) + pred_logvar)
                loss = torch.mean(nll * weights_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            mean_train, logvar_train = model(x_train_t)
            mean_test, logvar_test = model(x_test_t)
            logvar_train = torch.clamp(logvar_train, min=-6.0, max=4.0)
            logvar_test = torch.clamp(logvar_test, min=-6.0, max=4.0)
            train_nll_t = 0.5 * (((mean_train - y_train_t) ** 2) * torch.exp(-logvar_train) + logvar_train)
            test_nll_t = 0.5 * (((mean_test - y_test_t) ** 2) * torch.exp(-logvar_test) + logvar_test)
            pred_train = mean_train.cpu().numpy().astype(np.float64) * y_std + y_mean
            pred_test = mean_test.cpu().numpy().astype(np.float64) * y_std + y_mean
            pred_logvar_test = logvar_test.cpu().numpy().astype(np.float64)
            train_nll = float(torch.mean(train_nll_t * weights_t).cpu().item())
            test_nll = float(torch.mean(test_nll_t * weights_t).cpu().item())

        rows: list[dict[str, Any]] = []
        weighted_nonneg = 0.0
        weighted_raw = 0.0
        positive_share = 0.0
        top5: list[float] = []
        var_error_corrs: list[float] = []
        abs_err_std = np.abs((y_test_raw - pred_test) / y_std)
        pred_var = np.exp(pred_logvar_test)
        for pc_idx in range(scores.shape[1]):
            r2_train = float(r2_score(y_train_raw[:, pc_idx], pred_train[:, pc_idx]))
            r2_test = float(r2_score(y_test_raw[:, pc_idx], pred_test[:, pc_idx]))
            evr = float(explained_ratio[pc_idx])
            weighted_nonneg += max(0.0, r2_test) * evr
            weighted_raw += r2_test * evr
            if r2_test > 0.0:
                positive_share += evr
            if pc_idx < 5:
                top5.append(r2_test)
            corr = float("nan")
            if np.std(pred_var[:, pc_idx]) > 1e-12 and np.std(abs_err_std[:, pc_idx]) > 1e-12:
                corr = float(np.corrcoef(pred_var[:, pc_idx], abs_err_std[:, pc_idx])[0, 1])
                if np.isfinite(corr):
                    var_error_corrs.append(corr)
            rows.append(
                {
                    "pc_index": int(pc_idx + 1),
                    "explained_variance_ratio": evr,
                    "ridge_alpha": float("nan"),
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "test_nll": float(np.mean(0.5 * (abs_err_std[:, pc_idx] ** 2 / np.clip(pred_var[:, pc_idx], 1e-12, None) + pred_logvar_test[:, pc_idx]))),
                    "pred_var_abs_error_corr": corr,
                }
            )
        summary = {
            "n_pcs": int(scores.shape[1]),
            "weighted_nonnegative_r2": float(weighted_nonneg),
            "weighted_raw_r2": float(weighted_raw),
            "positive_pc_explained_share": float(positive_share),
            "mean_r2_top5": float(np.mean(top5)) if top5 else float("nan"),
            "median_r2_all": float(np.median([r["r2_test"] for r in rows])) if rows else float("nan"),
            "train_weighted_nll": train_nll,
            "test_weighted_nll": test_nll,
            "mean_pred_var_abs_error_corr": float(np.mean(var_error_corrs)) if var_error_corrs else float("nan"),
        }
        out.append((f"seed{seed}", summary, rows))
    return out


@dataclass
class SslConfig:
    seeds: list[int]
    epochs: int
    embed_dim: int
    hidden_dim: int
    mask_rate: float
    temperature: float
    lr: float
    batch_size: int


def _masked_autoencoder_embeddings(
    x: np.ndarray,
    train_mask: np.ndarray,
    cfg: SslConfig,
    device: str,
    label: str = "acs_all",
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, embed), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(embed, hidden), nn.ReLU(), nn.Linear(hidden, d_in))

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x_in))

        def embed(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.encoder(x_in)

    scaler = StandardScaler()
    xtr = scaler.fit_transform(x[train_mask]).astype(np.float32)
    xall = scaler.transform(x).astype(np.float32)
    out: dict[str, np.ndarray] = {}
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(label).strip()) or "acs_all"

    dev = torch.device(device)
    n_train, d_in = xtr.shape
    for seed in cfg.seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        model = Model(d_in=d_in, hidden=int(cfg.hidden_dim), embed=int(cfg.embed_dim)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
        x_train_t = torch.tensor(xtr, dtype=torch.float32, device=dev)
        batch_size = min(int(cfg.batch_size), n_train)
        for _ in range(int(cfg.epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                xb = x_train_t[idx]
                mask = (torch.rand_like(xb) < float(cfg.mask_rate)).float()
                corrupted = xb * (1.0 - mask)
                pred = model(corrupted)
                masked_loss = ((pred - xb) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
                all_loss = 0.10 * torch.mean((pred - xb) ** 2)
                loss = masked_loss + all_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            emb = model.embed(torch.tensor(xall, dtype=torch.float32, device=dev)).cpu().numpy().astype(np.float64)
        out[f"ssl_masked_{label}_seed{seed}"] = emb
    return out


def _build_acs_views(feature_df: pd.DataFrame) -> dict[str, np.ndarray]:
    def cols_for(prefix: str) -> list[str]:
        return [c for c in feature_df.columns if c.startswith(prefix)]

    view_cols = {
        "age_sex_1d": cols_for("acs__AGEP_bin__") + cols_for("acs__SEX__"),
        "age_sex_cross": cols_for("acs__AGEP_SEX_cross__"),
        "education": cols_for("acs__SCHL_allpop__"),
        "employment": cols_for("acs__ESR_allpop__"),
        "earnings": cols_for("acs__EARN_16p_bin__"),
    }
    return {
        name: feature_df[cols].to_numpy(dtype=np.float64)
        for name, cols in view_cols.items()
        if cols
    }


def _multiview_contrastive_embeddings(
    views: dict[str, np.ndarray],
    train_mask: np.ndarray,
    cfg: SslConfig,
    device: str,
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ViewEncoder(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, embed))

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x_in), dim=1)

    view_names = sorted(views)
    if len(view_names) < 2:
        return {}

    scalers: dict[str, StandardScaler] = {}
    x_train: dict[str, np.ndarray] = {}
    x_all: dict[str, np.ndarray] = {}
    for name in view_names:
        scaler = StandardScaler()
        x_train[name] = scaler.fit_transform(views[name][train_mask]).astype(np.float32)
        x_all[name] = scaler.transform(views[name]).astype(np.float32)
        scalers[name] = scaler

    out: dict[str, np.ndarray] = {}
    dev = torch.device(device)
    n_train = int(train_mask.sum())
    for seed in cfg.seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        encoders = nn.ModuleDict(
            {
                name: ViewEncoder(d_in=x_train[name].shape[1], hidden=int(cfg.hidden_dim), embed=int(cfg.embed_dim))
                for name in view_names
            }
        ).to(dev)
        opt = torch.optim.AdamW(encoders.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
        x_train_t = {
            name: torch.tensor(x_train[name], dtype=torch.float32, device=dev)
            for name in view_names
        }
        batch_size = min(int(cfg.batch_size), n_train)
        temp = float(cfg.temperature)

        for _ in range(int(cfg.epochs)):
            order = torch.randperm(n_train, device=dev)
            for start in range(0, n_train, batch_size):
                idx = order[start : start + batch_size]
                if idx.numel() < 2:
                    continue
                z = {name: encoders[name](x_train_t[name][idx]) for name in view_names}
                labels = torch.arange(idx.numel(), device=dev)
                loss = torch.zeros((), dtype=torch.float32, device=dev)
                n_pairs = 0
                for i, name_i in enumerate(view_names):
                    for name_j in view_names[i + 1 :]:
                        logits = z[name_i] @ z[name_j].T / max(temp, 1e-6)
                        loss = loss + F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                        n_pairs += 2
                loss = loss / max(n_pairs, 1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        with torch.no_grad():
            emb_parts = []
            for name in view_names:
                z_all = encoders[name](torch.tensor(x_all[name], dtype=torch.float32, device=dev))
                emb_parts.append(z_all)
            emb = F.normalize(torch.stack(emb_parts, dim=0).mean(dim=0), dim=1).cpu().numpy().astype(np.float64)
        out[f"ssl_multiview_acs_seed{seed}"] = emb
    return out


def _functional_contrastive_embeddings(
    x: np.ndarray,
    target_keys: pd.DataFrame,
    train_mask: np.ndarray,
    edges_csv: pathlib.Path,
    graph_label: str,
    cfg: SslConfig,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not edges_csv.exists():
        raise SystemExit(f"functional_edges_csv not found: {edges_csv}")
    edges = pd.read_csv(edges_csv, dtype=str)
    required = {"home_puma_uid", "work_puma_uid"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise SystemExit(f"functional edges missing columns {missing}: {edges_csv}")

    keys = target_keys["puma_uid_key"].astype(str).str.zfill(7).tolist()
    key_to_idx = {k: i for i, k in enumerate(keys)}
    pos = np.zeros((len(keys), len(keys)), dtype=bool)
    used_edges = 0
    for r in edges.itertuples(index=False):
        i = key_to_idx.get(str(getattr(r, "home_puma_uid")).zfill(7))
        j = key_to_idx.get(str(getattr(r, "work_puma_uid")).zfill(7))
        if i is None or j is None or i == j:
            continue
        pos[i, j] = True
        pos[j, i] = True
        used_edges += 1
    np.fill_diagonal(pos, False)

    train_idx = np.flatnonzero(train_mask)
    pos_train = pos[np.ix_(train_idx, train_idx)]
    anchors_with_pos = pos_train.any(axis=1)
    n_train_pos_edges = int(pos_train.sum())
    meta = {
        "functional_edges_csv": str(edges_csv),
        "n_edges_loaded": int(edges.shape[0]),
        "n_edges_mapped_to_target": int(used_edges),
        "n_train_positive_edges_directed": n_train_pos_edges,
        "n_train_anchors_with_positive": int(anchors_with_pos.sum()),
    }
    if n_train_pos_edges == 0 or int(anchors_with_pos.sum()) < 2:
        meta["skipped_reason"] = "no functional positive pairs within the training split"
        return {}, meta

    class Encoder(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed),
            )

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return F.normalize(self.net(x_in), dim=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_all = scaler.transform(x).astype(np.float32)
    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    pos_t = torch.tensor(pos_train, dtype=torch.bool, device=dev)
    nonself_t = ~torch.eye(pos_train.shape[0], dtype=torch.bool, device=dev)
    valid_anchor_t = pos_t.any(dim=1)
    temp = max(float(cfg.temperature), 1e-6)

    out: dict[str, np.ndarray] = {}
    graph_label = re.sub(r"[^0-9A-Za-z_]+", "_", str(graph_label).strip()) or "graph"
    for seed in cfg.seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        model = Encoder(d_in=x_train.shape[1], hidden=int(cfg.hidden_dim), embed=int(cfg.embed_dim)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
        for _ in range(int(cfg.epochs)):
            z = model(x_train_t)
            logits = z @ z.T / temp
            logits = logits.masked_fill(~nonself_t, -1e9)
            log_den = torch.logsumexp(logits, dim=1)
            log_pos = torch.logsumexp(logits.masked_fill(~pos_t, -1e9), dim=1)
            loss_vec = -(log_pos - log_den)
            loss = loss_vec[valid_anchor_t].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            emb = model(x_all_t).cpu().numpy().astype(np.float64)
        out[f"ssl_graph_{graph_label}_seed{seed}"] = emb
    return out, meta


def _masked_graph_contrastive_embeddings(
    x: np.ndarray,
    target_keys: pd.DataFrame,
    train_mask: np.ndarray,
    edges_csv: pathlib.Path,
    graph_label: str,
    input_label: str,
    cfg: SslConfig,
    device: str,
    graph_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not edges_csv.exists():
        raise SystemExit(f"functional_edges_csv not found: {edges_csv}")
    edges = pd.read_csv(edges_csv, dtype=str)
    required = {"home_puma_uid", "work_puma_uid"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise SystemExit(f"graph edges missing columns {missing}: {edges_csv}")

    keys = target_keys["puma_uid_key"].astype(str).str.zfill(7).tolist()
    key_to_idx = {k: i for i, k in enumerate(keys)}
    pos = np.zeros((len(keys), len(keys)), dtype=bool)
    used_edges = 0
    for r in edges.itertuples(index=False):
        i = key_to_idx.get(str(getattr(r, "home_puma_uid")).zfill(7))
        j = key_to_idx.get(str(getattr(r, "work_puma_uid")).zfill(7))
        if i is None or j is None or i == j:
            continue
        pos[i, j] = True
        pos[j, i] = True
        used_edges += 1

    train_idx = np.flatnonzero(train_mask)
    pos_train = pos[np.ix_(train_idx, train_idx)]
    anchors_with_pos = pos_train.any(axis=1)
    meta = {
        "functional_edges_csv": str(edges_csv),
        "n_edges_loaded": int(edges.shape[0]),
        "n_edges_mapped_to_target": int(used_edges),
        "n_train_positive_edges_directed": int(pos_train.sum()),
        "n_train_anchors_with_positive": int(anchors_with_pos.sum()),
        "graph_weight": float(graph_weight),
        "mode": "masked_graph",
    }
    if int(pos_train.sum()) == 0 or int(anchors_with_pos.sum()) < 2:
        meta["skipped_reason"] = "no graph positive pairs within the training split"
        return {}, meta

    class Model(nn.Module):
        def __init__(self, d_in: int, hidden: int, embed: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Linear(hidden, embed),
            )
            self.decoder = nn.Sequential(
                nn.ReLU(),
                nn.Linear(embed, hidden),
                nn.ReLU(),
                nn.Linear(hidden, d_in),
            )

        def embed(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.encoder(x_in)

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.embed(x_in))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_mask]).astype(np.float32)
    x_all = scaler.transform(x).astype(np.float32)
    dev = torch.device(device)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=dev)
    x_all_t = torch.tensor(x_all, dtype=torch.float32, device=dev)
    pos_t = torch.tensor(pos_train, dtype=torch.bool, device=dev)
    nonself_t = ~torch.eye(pos_train.shape[0], dtype=torch.bool, device=dev)
    valid_anchor_t = pos_t.any(dim=1)
    temp = max(float(cfg.temperature), 1e-6)
    graph_weight = float(graph_weight)

    out: dict[str, np.ndarray] = {}
    graph_label = re.sub(r"[^0-9A-Za-z_]+", "_", str(graph_label).strip()) or "graph"
    input_label = re.sub(r"[^0-9A-Za-z_]+", "_", str(input_label).strip()) or "acs_all"
    weight_label = str(graph_weight).replace(".", "p").replace("-", "m")
    for seed in cfg.seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        model = Model(d_in=x_train.shape[1], hidden=int(cfg.hidden_dim), embed=int(cfg.embed_dim)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
        for _ in range(int(cfg.epochs)):
            mask = (torch.rand_like(x_train_t) < float(cfg.mask_rate)).float()
            corrupted = x_train_t * (1.0 - mask)
            pred = model(corrupted)
            masked_loss = ((pred - x_train_t) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
            all_loss = 0.10 * torch.mean((pred - x_train_t) ** 2)

            z = F.normalize(model.embed(x_train_t), dim=1)
            logits = z @ z.T / temp
            logits = logits.masked_fill(~nonself_t, -1e9)
            log_den = torch.logsumexp(logits, dim=1)
            log_pos = torch.logsumexp(logits.masked_fill(~pos_t, -1e9), dim=1)
            graph_loss = (-(log_pos - log_den))[valid_anchor_t].mean()
            loss = masked_loss + all_loss + graph_weight * graph_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            emb = model.embed(x_all_t).cpu().numpy().astype(np.float64)
        out[f"ssl_masked_{input_label}_graph_{graph_label}_a{weight_label}_seed{seed}"] = emb
    return out, meta


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe whether SSL-style region representations predict copula residual structure.")
    data_root = pathlib.Path("/home/jinlin/data/geoexplicit_data/synthetic_city/data")
    ap.add_argument(
        "--target_wide_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_targets/exttarget_v1_full_earn_pums_2023_puma_us_joint_wide.csv",
    )
    ap.add_argument(
        "--condition_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/external_conditions/extcond_v1_agesex_earn_v1_acs5_2022_puma_us.csv",
    )
    ap.add_argument(
        "--spatial_csv",
        type=pathlib.Path,
        default=data_root / "us/processed/features/puma_spatial_features_5var_knn6.csv",
    )
    ap.add_argument(
        "--extra_feature_csv",
        type=pathlib.Path,
        default=None,
        help="Optional PUMA-level wide feature view, e.g. ACS survey-uncertainty features.",
    )
    ap.add_argument("--extra_feature_label", default="extra")
    ap.add_argument("--heldout_statefps", default="26,06,12,48,55")
    ap.add_argument("--reference_modes", default="target_marginals,acs_marginals")
    ap.add_argument("--joint_space", choices=["full", "coarse"], default="full")
    ap.add_argument("--max_pcs", type=int, default=40)
    ap.add_argument("--ridge_alphas", default="0.01,0.1,1,10,100,1000")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl_seeds", default="0,1,2")
    ap.add_argument("--ssl_modes", default="masked,multiview")
    ap.add_argument("--ssl_epochs", type=int, default=500)
    ap.add_argument("--ssl_embed_dim", type=int, default=24)
    ap.add_argument("--ssl_hidden_dim", type=int, default=96)
    ap.add_argument("--ssl_mask_rate", type=float, default=0.30)
    ap.add_argument("--ssl_temperature", type=float, default=0.10)
    ap.add_argument("--ssl_lr", type=float, default=1e-3)
    ap.add_argument("--ssl_batch_size", type=int, default=256)
    ap.add_argument("--functional_edges_csv", type=pathlib.Path, default=None)
    ap.add_argument("--functional_graph_label", default="functional")
    ap.add_argument("--graph_contrast_weight", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_ssl", action="store_true")
    ap.add_argument("--mlp_probe", action="store_true")
    ap.add_argument("--mlp_feature_sets", default="acs_1d,acs_all,acs_all_spatial")
    ap.add_argument("--hetero_mlp_probe", action="store_true")
    ap.add_argument("--hetero_mlp_feature_sets", default="acs_1d,acs_all,acs_all_spatial")
    ap.add_argument("--mlp_epochs", type=int, default=400)
    ap.add_argument("--mlp_hidden_dim", type=int, default=128)
    ap.add_argument("--mlp_embed_dim", type=int, default=32)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--output_dir", type=pathlib.Path, default=None)
    args = ap.parse_args()

    out_dir = args.output_dir or pathlib.Path(f"outputs/_ssl_copula_residual_probe_{_utc_ts()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    target_keys, p_true, p_eq_target, target_marginals = _load_target(args.target_wide_csv)
    acs_feature_df, acs_marginals, x_acs_1d, x_acs_all, x_acs_scale, one_d_cols, all_cols, scale_cols = _load_acs_conditions(args.condition_csv, target_keys)
    x_spatial, spatial_cols = _load_spatial(args.spatial_csv, target_keys)
    x_extra: np.ndarray | None = None
    extra_cols: list[str] = []
    extra_label = str(args.extra_feature_label).strip() or "extra"
    if args.extra_feature_csv is not None:
        x_extra, extra_cols_raw = _load_spatial(path=args.extra_feature_csv, target_keys=target_keys)
        extra_cols = [f"{extra_label}__{c}" for c in extra_cols_raw]
    p_eq_acs = _outer_joint([acs_marginals[v] for v in FULL_VARIABLE_ORDER])

    if str(args.joint_space) == "coarse":
        p_true = _aggregate_full_to_coarse(p_true)
        p_eq_target = _aggregate_full_to_coarse(p_eq_target)
        p_eq_acs = _aggregate_full_to_coarse(p_eq_acs)

    x_target_oracle = np.concatenate([target_marginals[v] for v in FULL_VARIABLE_ORDER], axis=1)
    features_base: dict[str, np.ndarray] = {
        "acs_1d": x_acs_1d,
        "acs_all": x_acs_all,
        "acs_1d_scale": np.concatenate([x_acs_1d, x_acs_scale], axis=1),
        "acs_all_scale": np.concatenate([x_acs_all, x_acs_scale], axis=1),
        "acs_all_spatial": np.concatenate([x_acs_all, x_spatial], axis=1),
        "acs_all_spatial_scale": np.concatenate([x_acs_all, x_spatial, x_acs_scale], axis=1),
        "target_1d_oracle": x_target_oracle,
    }
    if x_extra is not None:
        features_base[extra_label] = x_extra
        features_base[f"acs_all_{extra_label}"] = np.concatenate([x_acs_all, x_extra], axis=1)
        features_base[f"acs_all_scale_{extra_label}"] = np.concatenate([x_acs_all, x_acs_scale, x_extra], axis=1)
        features_base[f"acs_all_spatial_{extra_label}"] = np.concatenate([x_acs_all, x_spatial, x_extra], axis=1)
        features_base[f"acs_all_spatial_scale_{extra_label}"] = np.concatenate([x_acs_all, x_spatial, x_acs_scale, x_extra], axis=1)

    heldout_statefps = [str(x).zfill(2) for x in _parse_csv_ints(args.heldout_statefps)]
    reference_modes = [x.strip() for x in str(args.reference_modes).split(",") if x.strip()]
    ridge_alphas = _parse_csv_floats(args.ridge_alphas)
    ssl_cfg = SslConfig(
        seeds=_parse_csv_ints(args.ssl_seeds),
        epochs=int(args.ssl_epochs),
        embed_dim=int(args.ssl_embed_dim),
        hidden_dim=int(args.ssl_hidden_dim),
        mask_rate=float(args.ssl_mask_rate),
        temperature=float(args.ssl_temperature),
        lr=float(args.ssl_lr),
        batch_size=int(args.ssl_batch_size),
    )
    ssl_modes = [x.strip() for x in str(args.ssl_modes).split(",") if x.strip()]
    mlp_feature_sets = [x.strip() for x in str(args.mlp_feature_sets).split(",") if x.strip()]
    hetero_mlp_feature_sets = [x.strip() for x in str(args.hetero_mlp_feature_sets).split(",") if x.strip()]

    statefp = target_keys["statefp"].astype(str).str.zfill(2).to_numpy()
    puma_uid = target_keys["puma_uid_key"].astype(str).to_numpy()
    feature_summary: list[dict[str, Any]] = []
    pc_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    functional_ssl_meta: list[dict[str, Any]] = []

    references = {
        "target_marginals": p_eq_target,
        "acs_marginals": p_eq_acs,
    }

    for ref_name in reference_modes:
        if ref_name not in references:
            raise SystemExit(f"unknown reference mode: {ref_name}")
        tvd, kl, log_ratio = _residual_arrays(p_true, references[ref_name], eps=float(args.eps))
        for heldout in heldout_statefps:
            train_mask = statefp != heldout
            if int((~train_mask).sum()) == 0:
                print(f"[warn] heldout state {heldout} has no rows; skipped", file=sys.stderr)
                continue
            residual_rows.append(
                {
                    "reference_mode": ref_name,
                    "heldout_statefp": heldout,
                    "n_train": int(train_mask.sum()),
                    "n_test": int((~train_mask).sum()),
                    "tvd_train_mean": float(np.mean(tvd[train_mask])),
                    "tvd_test_mean": float(np.mean(tvd[~train_mask])),
                    "kl_train_mean": float(np.mean(kl[train_mask])),
                    "kl_test_mean": float(np.mean(kl[~train_mask])),
                }
            )

            scores, explained_ratio, cumulative = _fit_pc_scores(
                log_ratio,
                train_mask=train_mask,
                n_components=int(args.max_pcs),
                seed=int(args.seed),
            )

            feature_sets = dict(features_base)
            if not bool(args.skip_ssl):
                if "masked" in ssl_modes:
                    ssl_embeds = _masked_autoencoder_embeddings(
                        x=x_acs_all,
                        train_mask=train_mask,
                        cfg=ssl_cfg,
                        device=str(args.device),
                        label="acs_all",
                    )
                    feature_sets.update(ssl_embeds)
                if "masked_scale" in ssl_modes:
                    ssl_embeds = _masked_autoencoder_embeddings(
                        x=features_base["acs_all_scale"],
                        train_mask=train_mask,
                        cfg=ssl_cfg,
                        device=str(args.device),
                        label="acs_all_scale",
                    )
                    feature_sets.update(ssl_embeds)
                if "multiview" in ssl_modes:
                    ssl_views = _build_acs_views(acs_feature_df)
                    ssl_embeds = _multiview_contrastive_embeddings(
                        views=ssl_views,
                        train_mask=train_mask,
                        cfg=ssl_cfg,
                        device=str(args.device),
                    )
                    feature_sets.update(ssl_embeds)
                if "functional" in ssl_modes:
                    if args.functional_edges_csv is None:
                        print("[warn] ssl_modes includes functional but --functional_edges_csv is not set; skipped", file=sys.stderr)
                    else:
                        ssl_embeds, meta = _functional_contrastive_embeddings(
                            x=x_acs_all,
                            target_keys=target_keys,
                            train_mask=train_mask,
                            edges_csv=args.functional_edges_csv.expanduser().resolve(),
                            graph_label=str(args.functional_graph_label),
                            cfg=ssl_cfg,
                            device=str(args.device),
                        )
                        meta.update({"reference_mode": ref_name, "heldout_statefp": heldout})
                        functional_ssl_meta.append(meta)
                        feature_sets.update(ssl_embeds)
                if "masked_graph" in ssl_modes:
                    if args.functional_edges_csv is None:
                        print("[warn] ssl_modes includes masked_graph but --functional_edges_csv is not set; skipped", file=sys.stderr)
                    else:
                        ssl_embeds, meta = _masked_graph_contrastive_embeddings(
                            x=x_acs_all,
                            target_keys=target_keys,
                            train_mask=train_mask,
                            edges_csv=args.functional_edges_csv.expanduser().resolve(),
                            graph_label=str(args.functional_graph_label),
                            input_label="acs_all",
                            cfg=ssl_cfg,
                            device=str(args.device),
                            graph_weight=float(args.graph_contrast_weight),
                        )
                        meta.update({"reference_mode": ref_name, "heldout_statefp": heldout})
                        functional_ssl_meta.append(meta)
                        feature_sets.update(ssl_embeds)
                if "masked_graph_scale" in ssl_modes:
                    if args.functional_edges_csv is None:
                        print("[warn] ssl_modes includes masked_graph_scale but --functional_edges_csv is not set; skipped", file=sys.stderr)
                    else:
                        ssl_embeds, meta = _masked_graph_contrastive_embeddings(
                            x=features_base["acs_all_scale"],
                            target_keys=target_keys,
                            train_mask=train_mask,
                            edges_csv=args.functional_edges_csv.expanduser().resolve(),
                            graph_label=str(args.functional_graph_label),
                            input_label="acs_all_scale",
                            cfg=ssl_cfg,
                            device=str(args.device),
                            graph_weight=float(args.graph_contrast_weight),
                        )
                        meta.update({"reference_mode": ref_name, "heldout_statefp": heldout})
                        functional_ssl_meta.append(meta)
                        feature_sets.update(ssl_embeds)

            for feature_name, x in feature_sets.items():
                probe_summary, probe_rows = _ridge_probe(
                    x=x,
                    scores=scores,
                    train_mask=train_mask,
                    explained_ratio=explained_ratio,
                    alphas=ridge_alphas,
                )
                feature_summary.append(
                    {
                        "reference_mode": ref_name,
                        "heldout_statefp": heldout,
                        "feature_set": feature_name,
                        "n_train": int(train_mask.sum()),
                        "n_test": int((~train_mask).sum()),
                        "n_features": int(x.shape[1]),
                        "n_pcs": int(scores.shape[1]),
                        "pc_cumulative_explained": float(cumulative),
                        **probe_summary,
                    }
                )
                for row in probe_rows:
                    pc_rows.append(
                        {
                            "reference_mode": ref_name,
                            "heldout_statefp": heldout,
                            "feature_set": feature_name,
                            **row,
                        }
                    )

            if bool(args.mlp_probe):
                for base_name in mlp_feature_sets:
                    if base_name not in feature_sets:
                        continue
                    mlp_results = _mlp_direct_probe(
                        x=feature_sets[base_name],
                        scores=scores,
                        train_mask=train_mask,
                        explained_ratio=explained_ratio,
                        seeds=ssl_cfg.seeds,
                        epochs=int(args.mlp_epochs),
                        hidden_dim=int(args.mlp_hidden_dim),
                        embed_dim=int(args.mlp_embed_dim),
                        lr=float(args.mlp_lr),
                        batch_size=int(args.ssl_batch_size),
                        device=str(args.device),
                    )
                    for seed_name, probe_summary, probe_rows in mlp_results:
                        feature_name = f"mlp_residual_{base_name}_{seed_name}"
                        feature_summary.append(
                            {
                                "reference_mode": ref_name,
                                "heldout_statefp": heldout,
                                "feature_set": feature_name,
                                "n_train": int(train_mask.sum()),
                                "n_test": int((~train_mask).sum()),
                                "n_features": int(feature_sets[base_name].shape[1]),
                                "n_pcs": int(scores.shape[1]),
                                "pc_cumulative_explained": float(cumulative),
                                **probe_summary,
                            }
                        )
                        for row in probe_rows:
                            pc_rows.append(
                                {
                                    "reference_mode": ref_name,
                                    "heldout_statefp": heldout,
                                    "feature_set": feature_name,
                                    **row,
                                }
                            )

            if bool(args.hetero_mlp_probe):
                for base_name in hetero_mlp_feature_sets:
                    if base_name not in feature_sets:
                        continue
                    hetero_results = _mlp_heteroskedastic_probe(
                        x=feature_sets[base_name],
                        scores=scores,
                        train_mask=train_mask,
                        explained_ratio=explained_ratio,
                        seeds=ssl_cfg.seeds,
                        epochs=int(args.mlp_epochs),
                        hidden_dim=int(args.mlp_hidden_dim),
                        embed_dim=int(args.mlp_embed_dim),
                        lr=float(args.mlp_lr),
                        batch_size=int(args.ssl_batch_size),
                        device=str(args.device),
                    )
                    for seed_name, probe_summary, probe_rows in hetero_results:
                        feature_name = f"hetero_mlp_residual_{base_name}_{seed_name}"
                        feature_summary.append(
                            {
                                "reference_mode": ref_name,
                                "heldout_statefp": heldout,
                                "feature_set": feature_name,
                                "n_train": int(train_mask.sum()),
                                "n_test": int((~train_mask).sum()),
                                "n_features": int(feature_sets[base_name].shape[1]),
                                "n_pcs": int(scores.shape[1]),
                                "pc_cumulative_explained": float(cumulative),
                                **probe_summary,
                            }
                        )
                        for row in probe_rows:
                            pc_rows.append(
                                {
                                    "reference_mode": ref_name,
                                    "heldout_statefp": heldout,
                                    "feature_set": feature_name,
                                    **row,
                                }
                            )

    pd.DataFrame(feature_summary).to_csv(out_dir / "metrics" / "feature_probe_summary.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(out_dir / "metrics" / "pc_probe_long.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(out_dir / "metrics" / "residual_reference_summary.csv", index=False)
    if functional_ssl_meta:
        pd.DataFrame(functional_ssl_meta).to_csv(out_dir / "metrics" / "functional_ssl_meta.csv", index=False)

    run_summary = {
        "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "question": "Do inference-time census and spatial views encode predictable copula residual structure?",
        "target_wide_csv": str(args.target_wide_csv),
        "condition_csv": str(args.condition_csv),
        "spatial_csv": str(args.spatial_csv),
        "extra_feature_csv": str(args.extra_feature_csv) if args.extra_feature_csv is not None else "",
        "extra_feature_label": extra_label if args.extra_feature_csv is not None else "",
        "n_regions": int(p_true.shape[0]),
        "joint_k": int(p_true.shape[1]),
        "joint_space": str(args.joint_space),
        "variable_order": FULL_VARIABLE_ORDER,
        "shape": list(COARSE_SHAPE if str(args.joint_space) == "coarse" else FULL_SHAPE),
        "heldout_statefps": heldout_statefps,
        "reference_modes": reference_modes,
        "max_pcs": int(args.max_pcs),
        "ridge_alphas": ridge_alphas,
        "feature_dimensions": {k: int(v.shape[1]) for k, v in features_base.items()},
        "condition_columns": {
            "acs_1d": one_d_cols,
            "acs_all": all_cols,
            "scale": scale_cols,
            "spatial": spatial_cols,
            "extra": extra_cols,
        },
        "ssl": {
            "enabled": not bool(args.skip_ssl),
            "modes": ssl_modes,
            "seeds": ssl_cfg.seeds,
            "epochs": ssl_cfg.epochs,
            "embed_dim": ssl_cfg.embed_dim,
            "hidden_dim": ssl_cfg.hidden_dim,
            "mask_rate": ssl_cfg.mask_rate,
            "temperature": ssl_cfg.temperature,
            "lr": ssl_cfg.lr,
            "batch_size": ssl_cfg.batch_size,
            "device": str(args.device),
            "functional_edges_csv": str(args.functional_edges_csv) if args.functional_edges_csv is not None else "",
            "graph_contrast_weight": float(args.graph_contrast_weight),
        },
        "mlp_probe": {
            "enabled": bool(args.mlp_probe),
            "feature_sets": mlp_feature_sets,
            "epochs": int(args.mlp_epochs),
            "hidden_dim": int(args.mlp_hidden_dim),
            "embed_dim": int(args.mlp_embed_dim),
            "lr": float(args.mlp_lr),
        },
        "hetero_mlp_probe": {
            "enabled": bool(args.hetero_mlp_probe),
            "feature_sets": hetero_mlp_feature_sets,
            "epochs": int(args.mlp_epochs),
            "hidden_dim": int(args.mlp_hidden_dim),
            "embed_dim": int(args.mlp_embed_dim),
            "lr": float(args.mlp_lr),
            "loss": "weighted Gaussian negative log-likelihood on standardized residual PC scores",
        },
        "outputs": {
            "feature_probe_summary": str(out_dir / "metrics" / "feature_probe_summary.csv"),
            "pc_probe_long": str(out_dir / "metrics" / "pc_probe_long.csv"),
            "residual_reference_summary": str(out_dir / "metrics" / "residual_reference_summary.csv"),
            "functional_ssl_meta": str(out_dir / "metrics" / "functional_ssl_meta.csv") if functional_ssl_meta else "",
        },
    }
    _write_json(out_dir / "run_summary.json", run_summary)

    summary_df = pd.DataFrame(feature_summary)
    compact = (
        summary_df.sort_values(["reference_mode", "heldout_statefp", "weighted_nonnegative_r2"], ascending=[True, True, False])
        .groupby(["reference_mode", "heldout_statefp"])
        .head(3)
        .loc[:, ["reference_mode", "heldout_statefp", "feature_set", "weighted_nonnegative_r2", "weighted_raw_r2", "mean_r2_top5"]]
    )
    print(compact.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

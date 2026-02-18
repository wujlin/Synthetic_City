# Methods

> 面向 essay 的方法描述。按"读者需要理解什么才能评价结果"组织，而非"我们做了什么"的执行清单。

---

## Overview

本方法的核心思路：**社会经济属性的联合分布随空间变化（copula 异质性），应当通过空间条件驱动生成，而非从全局种子加权获得。**

方法分为两层：
- **Layer 1**：人口统计骨架（age × sex × race），使用 Block Group 级 DHC 精确计数 + IPF 求解，零误差。
- **Layer 2**：社会经济属性（income, education, employment），使用条件化表格扩散模型从 PUMS 微数据中学习空间异质联合结构，以 Layer 1 的属性和空间位置作为条件输入。

两层的逻辑关系：Layer 1 提供"这个人的 age / sex / race 是什么"以及"这个人在哪个 PUMA"，Layer 2 回答"给定这些条件，这个人的 income / education / employment 最可能的联合配置是什么"。

---

## 1 ｜ Study Area & Data

### 1.1 Study Area

Michigan, USA. 68 PUMAs, 8,386 Block Groups, ~2,900 Census Tracts.

选择 Michigan 的原因：PUMA 数量足够支撑 5-fold holdout CV（每折 10–17 个 PUMA）；城市–郊区–农村的梯度为 copula 异质性提供了充分的空间变异。

### 1.2 Data Sources

| Data | Source | Spatial Resolution | Role |
|------|--------|-------------------|------|
| Decennial Census (DHC 2020) | Census Bureau | Block Group | Layer 1 constraints: P12 (age×sex), P5 (race) |
| PUMS 2022 5-Year | Census Bureau | PUMA20 | Layer 2 training: individual-level microdata (N=499,880) |
| ACS 2022 5-Year | Census Bureau | Tract | External validation: tract-level marginals |
| TIGER/Line 2020 | Census Bureau | — | Geographic crosswalks (BG→Tract→PUMA) |

### 1.3 Variables

**Layer 1 (deterministic)**:
- AGEP: age (single-year → 23 groups per P12)
- SEX: sex (2 categories)
- RAC1P: race (7 categories)

**Layer 2 (generative)**:
- PINCP: personal income (continuous, log-transformed)
- SCHL: educational attainment (24 ordered categories → embedded)
- ESR: employment status (6 categories → embedded)

---

## 2 ｜ Layer 1: Demographic Skeleton via Constrained IPF

### 2.1 Problem

Given DHC 2020 cross-tabulation P12 (age × sex, 46 cells per BG) and marginal P5 (race, 7 cells per BG), reconstruct the 3-way joint count table age × sex × race for each of the 8,386 BGs.

### 2.2 Method

1. **Initial seed**: State-level PUMS joint distribution of age × sex × race, collapsed to P12 age groups.
2. **IPF**: Iterative Proportional Fitting to simultaneously match P12 and P5 marginals.
3. **Integerization**: Controlled rounding to obtain exact integer counts.

### 2.3 Validation

Maximum absolute error across all 8,386 BGs: **0** for both P12 and P5 constraints. This is by construction—IPF with exact marginals and a non-degenerate seed converges to the unique solution.

Layer 1 is a solved problem and serves as the deterministic foundation for Layer 2. Its contribution to the paper is context, not novelty.

---

## 3 ｜ Layer 2: Conditional Tabular Diffusion for Socioeconomic Attributes

### 3.1 Problem Formulation

Given a synthetic individual with known $(a, s, r, \text{puma})$ from Layer 1, generate socioeconomic attributes $(y, e, w) \sim P(Y, E, W \mid A=a, S=s, R=r, \text{PUMA})$, where the conditional distribution varies across PUMAs (copula heterogeneity).

### 3.2 Architecture: Gaussian TabDDPM

We adopt a Gaussian DDPM operating on encoded mixed-type vectors, following the TabDDPM framework (Kotelnikov et al., 2023).

**Encoding**: Each record $\mathbf{x} = (y, e, w)$ is encoded into a continuous vector:
- **Continuous variables** (PINCP): quantile-normalized to $[0, 1]$
- **Categorical variables** (SCHL, ESR): embedded via learned low-dimensional embeddings, concatenated with continuous features

**Forward process**: Gaussian noise is progressively added over $T = 200$ timesteps:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})$$

**Reverse process**: A denoising network $\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})$ predicts the noise at each step, conditioned on:
$$\mathbf{c} = [\text{age\_group},\; \text{sex},\; \text{race},\; \text{puma\_id}]$$

The condition vector $\mathbf{c}$ is injected via concatenation to each layer of the denoising MLP. PUMA ID is one-hot encoded (68 dimensions), providing the spatial signal that enables the model to learn PUMA-specific copula structure.

**Decoding**: After sampling $\mathbf{x}_0$, continuous components are inverse-quantile-transformed; categorical components are decoded via nearest-neighbor lookup in the embedding space followed by argmax.

### 3.3 Training

- **Dataset**: PUMS 2022 5-Year Michigan, 499,880 person records
- **Epochs**: 2,000
- **Batch size**: 4,096
- **Loss**: Standard $\epsilon$-prediction MSE loss
- **Optimizer**: Adam, learning rate schedule with warmup

### 3.4 Conditioning Ablation Design

To isolate the contribution of each conditioning signal, we train three model variants:

| Variant | Condition vector $\mathbf{c}$ | Purpose |
|---------|-------------------------------|---------|
| `demo_only` | age + sex | Baseline: does knowing demographics help? |
| `demo_race` | age + sex + race | Increment: does race add information beyond demographics? |
| `demo_race_puma` | age + sex + race + PUMA ID | Increment: does spatial location capture copula heterogeneity? |

### 3.5 Inference (Sampling)

For each synthetic individual from Layer 1 with attributes $(a, s, r)$ in PUMA $p$:
1. Sample $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. Run reverse diffusion for $T$ steps with condition $\mathbf{c} = [a, s, r, p]$
3. Decode $\mathbf{x}_0$ to obtain $(y, e, w)$

No post-hoc IPF alignment is applied. The rationale: post-processing via IPF enforces marginal constraints but disrupts the joint structure learned by diffusion — Exp5 quantifies this trade-off.

---

## 4 ｜ Baselines & Evaluation

### 4.1 Non-learning Baselines

Three reference points establish the interpretive scale:

| Baseline | Description |
|----------|-------------|
| **Independence** | $P(A, Y) = P(A) \cdot P(Y)$; no dependency information |
| **Global Seed IPF** | State-level PUMS joint distribution applied uniformly to all PUMAs (standard practice) |
| **Oracle IPF** | Target PUMA's own true marginals used in IPF (unachievable in practice) |

### 4.2 Evaluation Protocol

**Spatial holdout**: 5-fold cross-validation at the PUMA level. In each fold, 10–17 PUMAs are held out entirely—no records from these PUMAs appear in training. This tests whether the model generalizes to **unseen spatial units**, not just unseen individuals within the same distribution.

This is a critical distinction from Qian et al. (2026), whose self-similarity protocol (5% subsample → generate → compare to 95%) tests generalization within the same spatial distribution.

### 4.3 Metrics

| Metric | Definition | What it measures |
|--------|------------|-----------------|
| **Marginal TVD** | $\text{TVD}(P_v, Q_v) = \frac{1}{2}\sum_i \|P_{v,i} - Q_{v,i}\|$ per variable $v$ | Per-variable distribution accuracy |
| **Joint TVD** | TVD on the full age × income cross-tabulation | Multivariate dependency accuracy |
| **Copula TVD** | TVD on rank-transformed age × income (marginal effects removed) | Pure dependency structure accuracy |
| **Cosine Similarity** | $\cos(\mathbf{p}, \mathbf{q})$ on flattened joint distribution vectors | Shape similarity (robust to scale) |

Copula TVD is the key metric: it isolates the dependency structure from marginal accuracy, directly measuring whether the model captures how variables co-vary (not just their marginals).

---

## 5 ｜ Copula Heterogeneity Diagnosis (Exp0)

Before modeling, we establish whether copula heterogeneity is real or merely sampling noise.

**Method**: For each of 68 PUMAs, compute the empirical copula of age × income (via rank transformation), then measure TVD to the state-level pooled copula and pairwise across all PUMAs.

**Rationale**: If copula TVD to global is ≈ 0, the global-seed assumption is justified and conditional modeling is unnecessary. If copula TVD is substantially larger than expected from sampling noise, spatial heterogeneity is real and demands spatially-aware methods.

---

## 6 ｜ End-to-End Pipeline Integration (Exp4)

The full pipeline chains Layer 1 → Layer 2:

1. For each BG, generate age × sex × race counts from DHC constraints (Layer 1)
2. Map each BG to its PUMA via TIGER geographic crosswalk
3. For each individual, sample socioeconomic attributes from the best diffusion model (`demo_race_puma`) conditioned on their demographics + PUMA
4. Aggregate to PUMA level for internal validation against PUMS ground truth

This tests whether Layer 1 → Layer 2 introduces cascading errors beyond what is observed in holdout evaluation.

---

## 7 ｜ External Validation at Tract Level (Exp3)

Synthetic population is aggregated to tract level and compared against ACS 2022 5-Year marginals. ACS data is not used in training—this is a fully external check.

The key interpretive anchor: AGEP, a Layer 1 variable with zero BG-level error, establishes an **irreducible data-source baseline** at the tract level due to the DHC 2020 vs ACS 2022 temporal mismatch. Layer 2 variables with tract TVD near this baseline are performing well within the limits of available data.

---

## 8 ｜ Post-Alignment Experiment (Exp5)

**Design**: Apply tract-level IPF post-alignment to the diffusion-generated population, using ACS tract marginals as constraints. Compare joint structure before and after.

**Purpose**: Test the natural idea that IPF can "fix" tract-level precision without damaging higher-level structure. The experiment is designed to demonstrate whether generation and post-processing are compatible.

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Encoded individual record $(y, e, w)$ |
| $\mathbf{c}$ | Condition vector $[\text{age}, \text{sex}, \text{race}, \text{puma}]$ |
| $T$ | Number of diffusion timesteps (200) |
| $\beta_t$ | Noise schedule at step $t$ |
| $\epsilon_\theta$ | Denoising network (MLP) |
| TVD | Total Variation Distance |
| PUMA | Public Use Microdata Area (~100k population) |
| BG | Census Block Group (~1,000–4,000 population) |

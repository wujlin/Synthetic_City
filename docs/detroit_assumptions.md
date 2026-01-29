# Detroit v0：关键假设与硬编码清单（可复现/可审查）

> 目的：把“默认值/硬编码/口径选择”集中记录，避免散落在脚本与注释里导致团队叙事不一致、复现实验困难或评审质疑。
>
> 说明：本文件优先记录**事实**（在哪个文件、默认值是什么、影响什么）。如需对外引用来源，可由联网同事补充核实链接与出处。

---

## 1. 城市中心点（CBD）与距离度量

- **默认 CBD 坐标（经纬度）**：`(-83.0458, 42.3314)`
- **用途**：计算建筑到 CBD 的距离特征 `dist_cbd_km`
- **代码位置**：`tools/prepare_detroit_buildings_gba.py`（命令行参数默认值 `--cbd_lon/--cbd_lat`）
- **影响**：`dist_cbd_km` 的绝对值与排序；若用于 tract_context 或后续分析，会影响“中心性”相关指标

待核实（可由联网同事补充出处）：
- 该坐标对应的地标/选点逻辑（例如 Campus Martius / Hart Plaza 等）

---

## 2. 建筑容量代理（cap_proxy）与层高假设

- **层高假设**：`3.0 m`（用于把 `height_m` 近似换算为楼层数 `floors_proxy = height_m / 3.0`）
- **容量代理**：`cap_proxy = footprint_area_m2 * floors_proxy`
- **代码位置**：`tools/prepare_detroit_buildings_gba.py`
- **影响**：
  - `cap_proxy` 用作空间分配时的采样权重（capacity prior）
  - 如果商业/工业建筑层高显著不同，可能导致容量权重偏差

---

## 3. 坐标系（CRS）与几何处理假设

- **GBA tile CRS 猜测**：若坐标数值量级更像投影坐标，则假设为 `EPSG:3857`；否则 `EPSG:4326`
- **TIGER 边界 CRS**：若缺失则按 TIGER/NAD83 设定（脚本中有默认）
- **代码位置**：`tools/prepare_detroit_buildings_gba.py`
- **影响**：城市裁剪、空间叠加（PUMA/tract mapping）、距离计算与 centroid 输出

---

## 4. Parcel assessment → price_tier 的口径

- **经济条件代理**：使用 Wayne County parcel assessment（评估值）派生建筑 `price_per_sqft`
- **price_tier 默认分层**：
  - 默认 `n_tiers = 5`（五分位）
  - 默认在 **tract 内**做分位数分层（`group_for_tier=tract`）
- **同一 parcel 多建筑的评估值分摊**：默认按 `footprint_area_m2` 分摊（`allocate_within_parcel=area`）
- **代码位置**：`tools/join_detroit_buildings_parcel_assessment.py`
- **影响**：
  - `income_price_match` 分配策略对建筑“价格档位”的依赖
  - tract 切分/分位数边界会影响跨 tract 的可比性（但增强 tract 内相对一致性）

---

## 5. 扩散模型（TabDDPM 风格）的默认超参数

- **默认 timesteps**：`200`
  - `src/synthpop/model/diffusion_tabular.py`（`TabDDPMConfig.timesteps`）
  - `tools/poc_tabddpm_pums_buildingcond.py`（`--timesteps` 默认值）
- **默认网络宽度**：`hidden_dims=(256, 256)`（`TabDDPMConfig.hidden_dims`）
- **默认学习率/权重衰减**：`lr=1e-3`, `weight_decay=1e-4`（`TabDDPMConfig`）
- **影响**：训练耗时、采样质量、离散变量 one-hot 解码稳定性

---

## 6. Scheme B（属性生成 ↔ 空间分配分离）的实现口径

- **训练**：仅用 PUMS 属性，学习 `P(attrs | macro_geo_context)`（PoC 使用 PUMA one-hot）
- **采样**：两阶段
  1) 生成属性（不含 building 特征）
  2) 显式分配到建筑（后处理模块，便于审查与消融）
- **空间分配策略**（`src/synthpop/spatial/building_allocation.py`）：
  - `random`：组内均匀随机
  - `capacity_only`：组内按 `cap_proxy` 加权
  - `income_price_match`：收入分位数 → `price_tier` 匹配，tier 内按 `cap_proxy` 加权（缺 tier 时回退到近邻 tier/全体）

---

## 7. PoC 验证口径与数据独立性（当前阶段）

- **统计验证（P0）**：`src/synthpop/validation/stats.py`
  - group 级边际 TVD（连续变量分箱 + 分类变量）
  - 关键二阶关联（Pearson / Cramér's V）
  - 最小硬规则违规率（例如 `AGEP<16` 且 `ESR in (1,2,3)`）
- **参考数据（短期）**：PUMS holdout（在每个 PUMA 内按 `train_frac` 划分）
  - 代码位置：`tools/poc_tabddpm_pums_buildingcond.py`（`--train_frac` 默认 0.8）
- **后续目标**：接入独立的 ACS Summary Tables（`processed/marginals/marginals_long.parquet`）做真正独立验证

---

## 8. 统计验证的默认分箱边界（continuous → bins）

当前 `src/synthpop/validation/stats.py` 的默认连续变量分箱（用于 group 级边际 TVD）：

- `AGEP`（年龄，左闭右开，含最低边界）：
  - `[0, 5), [5, 18), [18, 25), [25, 35), [35, 45), [45, 55), [55, 65), [65, 75), [75, 85), [85, 1000)`
- `PINCP`（个人收入，美元，左闭右开，含最低边界）：
  - `[0, 10000), [10000, 25000), [25000, 50000), [50000, 75000), [75000, 100000), [100000, 150000), [150000, 250000), [250000, 10000000)`

说明：
- 这是一套 **PoC 级**的稳定默认值（避免过细导致稀疏），后续接入 ACS Summary 时应按表口径/分组需求调整或直接使用 ACS 自带分箱。

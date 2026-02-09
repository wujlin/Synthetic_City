# 实验推进计划 v2：从诊断到分层生成

> **背景**：前序实验（Scheme B PoC）发现，扩散模型在 PUMA 级别能学到属性边际分布（AGEP TVD ~0.06），但 tract 级别误差偏高（TVD ~0.24），且 Joint-LogP 在 tract 级别的 TVD(0.209) 未超越 Independence baseline(0.192)。本计划重新审视数据策略与训练框架，通过分阶段实验验证核心假设并构建可靠的生成 pipeline。
>
> **核心逻辑链**：诊断假设（Exp 0）→ 构建基本人口（Exp 1）→ 属性扩展（Exp 2）→ 多层验证（Exp 3）

---

## Exp 0：Copula 空间异质性诊断

### 为什么这是第一个实验

扩散模型相对于 IPF 的唯一价值在于：**能学到空间异质的变量关联结构（copula）**。如果不同地区的 copula 几乎相同，IPF + 全局 seed joint 就够了，diffusion 没有附加值。这个实验决定后续所有投入是否值得。

### 数据

- **全 Michigan PUMS 5-Year**（person file + housing file）
  - 路径：`data/detroit/raw/pums/pums_2022_5-Year/csv_pmi.zip`（2022 5-Year，全 Michigan 499,880 人）
  - 如果只有 Wayne County 子集，需要下载全州数据（statefp=26 的完整 person/housing file）
- 不需要其他数据源

### 方法

**Step 1：计算每个 PUMA 的经验 copula**

对 Michigan 全部 ~500 个 PUMA，分别计算 age-income 的 copula：

```python
# 伪代码
for puma in all_michigan_pumas:
    persons = pums[pums.PUMA == puma]
    
    # Rank transform：将 age 和 income 各自标准化为 [0, 1] 均匀分布
    age_rank = rankdata(persons.AGEP) / len(persons)
    inc_rank = rankdata(persons.PINCP) / len(persons)
    
    # 二维直方图作为经验 copula（例如 10x10 bin）
    copula, _, _ = np.histogram2d(age_rank, inc_rank, bins=10, 
                                   range=[[0,1],[0,1]], density=True)
    copula /= copula.sum()  # 归一化为概率分布
    
    copulas[puma] = copula
```

**Step 2：度量 copula 的跨 PUMA 变异**

```python
# 计算全局平均 copula
global_copula = np.mean([c for c in copulas.values()], axis=0)

# 计算每个 PUMA 与全局 copula 的 TVD
tvd_to_global = {}
for puma, cop in copulas.items():
    tvd_to_global[puma] = 0.5 * np.abs(cop - global_copula).sum()

# 计算所有 PUMA 两两之间的 copula TVD（可选，计算量较大时取随机子集）
pairwise_tvds = []
for (p1, c1), (p2, c2) in combinations(copulas.items(), 2):
    pairwise_tvds.append(0.5 * np.abs(c1 - c2).sum())
```

**Step 3：测试可预测性**

如果 Step 2 发现 copula 有变异（TVD > 0.05），测试这种变异能否从 PUMA 特征预测：

```python
# PUMA 特征（从 PUMS 聚合或从 ACS 获取）
puma_features = {
    puma: [median_age, median_income, pct_white, pct_urban, 
           pct_college, pct_elderly, pop_density]
    for puma in all_pumas
}

# 将 PUMA 按特征聚类（K=5 或 K=10）
from sklearn.cluster import KMeans
X = np.array([puma_features[p] for p in sorted_pumas])
labels = KMeans(n_clusters=5).fit_predict(X)

# 比较同 cluster 内的 copula 相似度 vs 跨 cluster 的 copula 相似度
intra_tvds = []  # 同 cluster 内两两 TVD
inter_tvds = []  # 跨 cluster 两两 TVD

# 如果 mean(intra_tvds) << mean(inter_tvds)，说明空间特征能预测 copula 差异
```

### 输出

```
outputs/exp0_copula_diagnostic/
  copula_by_puma.json          # 每个 PUMA 的经验 copula
  tvd_to_global.json           # 每个 PUMA 与全局 copula 的 TVD
  pairwise_tvd_summary.json    # 两两 TVD 的统计摘要
  cluster_analysis.json        # 聚类可预测性分析
  diagnostic_summary.json      # 结论摘要
```

### 判决标准

| 结果 | 含义 | 后续行动 |
|------|------|---------|
| mean TVD_to_global < 0.03 | copula 全局一致 | diffusion 对 copula 无附加值；Layer 2 改为 IPF + 全局 seed |
| mean TVD_to_global ∈ [0.03, 0.08] | copula 有轻微变异 | diffusion 可能有附加值，但需要大量训练数据 |
| mean TVD_to_global > 0.08 且 intra << inter | copula 有可预测的空间变异 | diffusion 有明确价值，继续 Layer 2 |

### 补充诊断

除了 age-income，对以下变量对也做同样分析（工作量不大，复用代码即可）：
- age-sex（预期变异小——作为 sanity check）
- age-education（如果 PUMS 有 SCHL 变量）
- income-race（预期变异较大）

---

## Exp 1：Layer 1 基本人口重建

### 为什么需要这个实验

Layer 2（属性扩展）的 diffusion 模型需要以个体的 (age, sex, race) 作为条件输入。这些基本属性必须先在 BG 级别准确重建。这一步用精确方法（IPF/优化），不用 diffusion。

### 数据

需要下载的新数据（**之前项目中没有的**）：

1. **2020 Decennial Census DHC (Demographic and Housing Characteristics)**
   - 内容：BG 级别的 age × sex × race 交叉表（精确计数，非估计值）
   - 获取方式：Census API 或 NHGIS
   - API 示例：`https://api.census.gov/data/2020/dec/dhc?get=group(P12)&for=block%20group:*&in=state:26`
   - 关键表：
     - P12: Sex by Age（全种族）
     - P12A-P12I: Sex by Age，按种族分别
     - P5: Hispanic or Latino Origin by Race
   - 地理范围：全 Michigan（state=26），BG 级别

2. **导师的合成人口数据**（可选，用于对比）
   - 来源：OSF https://doi.org/10.17605/OSF.IO/FPNC2
   - 下载 Michigan 的 gpkg 文件
   - 包含：age, gender, household_type, household_id, lat/long
   - 不包含：race（需要我们补充）

### 方法

**方案 A：优化方法（参考 Lin 2023）**

目标：从 DHC 的多张交叉表中重建 BG 级别的个体记录。

```
输入：
  - P12 (sex × age, 全种族): 给出每个 BG 的 sex × 23_age_groups 计数
  - P12A-P12I (sex × age, 按种族): 给出每个种族的 sex × age 计数
  - P5 (Hispanic × race): 给出每个 BG 的 Hispanic × 7_race 计数

决策变量：
  X[k, j] = BG j 中属于谓词 k 的人数
  谓词 k = (age_group, sex, race, Hispanic)
  维度：23 × 2 × 7 × 2 = 644 种谓词

目标函数：
  min Σ_p ||W^(p) X - Y^(p)||²
  
约束：
  X[k, j] ∈ Z+ (非负整数)

求解：
  gurobipy（与 Lin 2023 一致）
```

**方案 B：IPF（如果 Gurobi 不可用或计算量太大）**

```
输入：
  - 全局 seed joint：从全 Michigan PUMS 计算 P(age, sex, race, Hispanic) 的联合分布
  - BG 级别边际：DHC 的 P12（age × sex 边际）和 P5（race × Hispanic 边际）

IPF 迭代：
  1. 初始化：seed × BG_total_population
  2. 拟合 age × sex 边际（来自 P12）
  3. 拟合 race × Hispanic 边际（来自 P5）
  4. 重复 2-3 直到收敛
  5. 四舍五入为整数，得到个体列表
```

### 输出

```
outputs/exp1_base_population/
  base_pop_michigan_bg.parquet    # 每行一个个体，列：bg_geoid, age_group, sex, race, hispanic
  internal_validation.json         # 与 DHC 的边际对比（应为零误差或接近零）
  comparison_with_advisor.json     # 与导师数据在 tract 级别的对比（可选）
```

### 验证

- **内部验证**：将个体按 BG 聚合，与 DHC 表对比。优化方法应达到零误差；IPF 方法误差应极小（< 0.01 TVD）。
- **交叉对比**（可选）：与导师数据在 tract 级别按 age × sex 对比，量化差异。

### 计算资源估算

- Michigan 约 8,200+ 个 BG
- 如果用 Gurobi 优化：每个 BG 是一个独立的小规模整数规划（644 变量），预计几分钟到几小时完成全州
- 如果用 IPF：更快，预计几分钟

---

## Exp 2：Layer 2 属性扩展（Diffusion）

### 前置条件

- Exp 0 结果表明 copula 有空间异质性（TVD_to_global > 0.03）
- Exp 1 已产出 BG 级别的基本人口

### 数据

- **训练数据**：全 Michigan PUMS（person file，~50万个体）
  - 每个个体有：AGEP, SEX, RAC1P, HISP, PINCP, SCHL, ESR, PUMA, ...
  - 路径：`data/detroit/raw/pums/pums_2022_5-Year/csv_pmi.zip`（全 Michigan，已确认 499,880 人）
- **验证数据**（均不参与训练）：
  - ACS Summary Tables at Tract（B19001: income, B15003: education, B23025: employment）
  - PUMS holdout（按 PUMA 分折）
  - Exp 1 的基本人口（提供条件输入）

### 训练设计

**核心思路**：模型学习 `P(income, education, employment | age, sex, race, area_features)`

每个训练样本是一个 PUMS 个体：
- **条件（已知）**：
  - 个体属性：age_group, sex, race（离散，one-hot 编码）
  - 区域特征：该个体所在 PUMA 的聚合统计（可选，见下方消融设计）
- **目标（要生成的）**：
  - income (PINCP)：连续变量，log-transform + 标准化
  - education (SCHL)：有序离散，embedding 或 one-hot
  - employment (ESR)：离散，one-hot

**训练方式**：条件 TabDDPM

```python
# 训练数据构造
for person in michigan_pums:
    condition = encode_condition(
        age_group=bin_age(person.AGEP),    # 离散化为 23 组
        sex=person.SEX,                     # 1=Male, 2=Female
        race=person.RAC1P,                  # 7 类
        # 可选：area_features（PUMA 级别聚合统计）
    )
    target = encode_target(
        income=log_transform(person.PINCP),
        education=person.SCHL,
        employment=person.ESR,
    )
    # → (condition, target) 作为一个训练对
```

**消融设计**（4 个实验条件）：

| 条件 ID | 个体属性条件 | 区域特征 | 目的 |
|---------|------------|---------|------|
| `demo_only` | age, sex, race | 无 | Baseline：纯人口学条件 |
| `demo+puma_stats` | age, sex, race | PUMA 的 median_income, pct_elderly, pop_density 等 | 测试区域统计是否有附加信号 |
| `demo+spatial` | age, sex, race | BG 的建筑特征聚合（mean_area, pct_residential 等） | 测试空间特征是否有附加信号（注意：训练时用 PUMA 级聚合） |
| `global_copula` | — | — | IPF baseline：全局 seed + 目标 BG 边际 |

### 训练参数（初始）

```
model: TabDDPM (DiffusionTabularModel)
timesteps: 200
hidden_dims: (512, 512)   # 比之前大一些，因为训练数据多了
epochs: 2000
batch_size: 4096
lr: 1e-3
seed: 0
device: cuda
```

### 交叉验证设计

将 Michigan 的 ~500 个 PUMA 按地理聚类分为 **5 折**（每折 ~100 个 PUMA）：

```
Fold 0: PUMA group A (held out) — 训练在 B, C, D, E
Fold 1: PUMA group B (held out) — 训练在 A, C, D, E
...
```

每折的评估：
1. 在 held-out PUMA 的 PUMS 个体上生成 (income, education, employment)
2. 比较生成 vs 真实的联合分布

### 推理流程（在 BG 级别生成）

```python
# 对 Exp 1 产出的每个 BG 个体
for individual in exp1_base_population:
    condition = encode_condition(
        age_group=individual.age_group,
        sex=individual.sex,
        race=individual.race,
        area_features=bg_features[individual.bg_geoid],  # 如果消融条件包含
    )
    
    # 从扩散模型采样
    generated = model.sample(n=1, cond=condition)
    individual.income = decode_income(generated.income)
    individual.education = decode_education(generated.education)
    individual.employment = decode_employment(generated.employment)
```

### 输出

```
outputs/exp2_attribute_extension/
  <condition_id>/                     # 每个消融条件一个子目录
    fold_<k>/
      model.pt
      train_summary.json
    metrics/
      pums_holdout_joint_tvd.json     # 5折 CV 的联合分布 TVD
      pums_holdout_marginal_tvd.json  # 各变量边际 TVD
      acs_tract_income_tvd.json       # 聚合到 tract 后与 ACS B19001 对比
      acs_tract_education_tvd.json    # 与 ACS B15003 对比
      copula_preservation.json        # age-income copula 保持度
  ablation_summary.json               # 4 个条件的对比汇总
```

### 关键评估指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **Joint TVD @ PUMA** | 生成 vs PUMS holdout 的 (age, income, education) 联合分布 TVD | 模型是否学到了联合结构 |
| **Marginal TVD @ Tract** (income) | 聚合到 tract，与 ACS B19001 对比（MOE-aware） | 空间降尺度后边际是否合理 |
| **Copula TVD @ PUMA** | 生成 vs PUMS holdout 的 age-income copula TVD | 模型是否学到了关联结构（而不只是边际） |
| **Δ(diffusion - global_copula)** | diffusion 的 copula TVD - IPF baseline 的 copula TVD | **diffusion 是否超越了 IPF baseline** |

**核心判据**：`Δ(diffusion - global_copula) < 0`，即 diffusion 的 copula TVD 低于 IPF baseline。如果 Δ ≥ 0，说明 diffusion 在 copula 上没有超越全局 seed，需要回到 Exp 0 重新审视。

---

## Exp 3：多层验证框架

### 目的

建立从 Exp 1 + Exp 2 产出的完整合成人口的系统验证，为后续 Scheme C（联合生成 + 空间锚定）提供 baseline 数字。

### 验证层级

**L1：BG 级别边际验证（对标 DHC）**

```python
# Exp 1 保证了 age, sex, race 与 DHC 精确匹配
# 这里验证 Exp 2 补充的 income, education 在 BG 聚合后是否合理

for bg in all_michigan_bgs:
    synthetic_persons = get_persons_in_bg(bg)
    
    # income 边际：与 ACS B19001 at BG 对比（如果有）
    # 注意：ACS BG 估计有较大 MOE，需要 MOE-aware 比较
    income_dist = compute_income_distribution(synthetic_persons)
    acs_income = get_acs_b19001(bg)  # 可能不可用于所有 BG
    
    if acs_income.available:
        tvd = compute_tvd(income_dist, acs_income.estimate)
        excess_tvd = max(0, tvd - acs_income.moe_adjusted_threshold)
```

**L2：Tract 级别边际验证（对标 ACS Summary Tables）**

```python
# ACS tract 级别估计比 BG 更稳定，是主要验证层级

for tract in all_michigan_tracts:
    synthetic_persons = get_persons_in_tract(tract)
    
    # income: vs ACS B19001
    income_tvd = tvd(synthetic_income_dist, acs_b19001[tract])
    
    # education: vs ACS B15003
    edu_tvd = tvd(synthetic_edu_dist, acs_b15003[tract])
    
    # employment: vs ACS B23025
    emp_tvd = tvd(synthetic_emp_dist, acs_b23025[tract])
    
    # 交叉验证：age × income 交叉表 vs ACS B19037
    # 注意：B19037 是 household 级别，需谨慎处理
```

**L3：PUMA 级别联合分布验证（对标 PUMS holdout）**

```python
# PUMS 提供个体级联合分布，是联合结构验证的唯一来源

for puma in holdout_pumas:
    synthetic = get_persons_in_puma(puma)
    reference = pums_holdout[puma]
    
    # 全变量联合 TVD
    joint_tvd = compute_joint_tvd(
        synthetic[['age', 'sex', 'race', 'income', 'education', 'employment']],
        reference[['AGEP', 'SEX', 'RAC1P', 'PINCP', 'SCHL', 'ESR']],
        bins=...
    )
    
    # Copula 保持度
    copula_tvd = compute_copula_tvd(synthetic, reference, vars=['age', 'income'])
```

**L4：独立外部验证（对标 LODES / 其他行政数据）**

```python
# LODES 提供 block 级别的 居住地-工作地 关系
# 可以验证：我们生成的就业年龄人口（ESR=1,2）的空间分布是否合理

# 下载 LODES WAC (Workplace Area Characteristics) 和 RAC (Residence Area Characteristics)
# 比较：每个 tract 的 employed population 计数 vs LODES RAC
```

### 输出

```
outputs/exp3_validation/
  L1_bg_validation.json
  L2_tract_validation.json
  L3_puma_joint_validation.json
  L4_lodes_validation.json       # 如果 LODES 数据可用
  validation_summary.json         # 多层汇总
```

---

## 数据已就绪 — 传输到工作站指南

> **所有必须数据已在本地下载并验证**。存放在 `dataset/wsA_staging/` 目录下，按工作站约定结构组织好了。
> 约定：`export RAW_ROOT=/home/jinlin/data/geoexplicit_data && export DATA_ROOT="$RAW_ROOT/synthetic_city/data"`

### !! 重要变更：PUMS 年份 2023 → 2022

2023 5-Year PUMS 尚未发布。实际使用 **2022 5-Year**（覆盖 2018-2022）。  
**所有脚本运行时需加 `--pums_year 2022`**（代码默认值仍是 2023）。  
PUMA 列名为 `PUMA20`（不是 `PUMA`），代码的 fallback 逻辑已覆盖。

---

### 传输操作（向日葵 / scp / rsync）

**本地 staging 目录** `dataset/wsA_staging/` 的结构：

```
wsA_staging/
├── detroit/raw/pums/pums_2022_5-Year/
│   ├── csv_pmi.zip          # Person (66MB, 499,880 人)
│   └── csv_hmi.zip          # Housing (30MB, 254,227 户)
├── detroit/raw/census/dhc_2020/
│   └── dhc_2020_bg_michigan.csv.gz    # 8,386 BGs, 总人口 10,077,331
├── detroit/raw/census/acs/acs5_2022/
│   ├── acs5_2022_B01001_tract_michigan.csv.gz  # Sex by Age (3,017 tracts)
│   ├── acs5_2022_B15003_tract_michigan.csv.gz  # Education
│   ├── acs5_2022_B19001_tract_michigan.csv.gz  # HH Income
│   ├── acs5_2022_B20001_tract_michigan.csv.gz  # Person Earnings
│   └── acs5_2022_B23025_tract_michigan.csv.gz  # Employment
├── detroit/raw/census/lodes/
│   ├── mi_rac_S000_JT00_2020.csv.gz   # 居住地就业 (4.4MB)
│   └── mi_wac_S000_JT00_2020.csv.gz   # 工作地就业 (1.4MB)
└── reference/advisor_synthpop/
    └── mi.zip              # 导师合成人口 (~1.8GB, 单独传输)
```

**目标路径**：将 staging 里的 `detroit/` 和 `reference/` 两个文件夹合并到工作站的：

```
/home/jinlin/data/geoexplicit_data/synthetic_city/data/
```

**方式 1：向日葵文件传输**  
把 `wsA_staging/detroit/` 和 `wsA_staging/reference/` 分别拖拽到工作站上述目录下。**选合并，不要替换**。

**方式 2：rsync（更快，推荐大文件）**

```bash
# 在本地终端执行（wsA 是工作站 SSH 别名）
cd /path/to/Synthetic_City
rsync -avP --partial dataset/wsA_staging/ \
  wsA:/home/jinlin/data/geoexplicit_data/synthetic_city/data/
```

**导师数据（mi.zip, ~1.8GB）**：可在 **wsA 上直接下载**到目标目录，省去向日葵传输。在 wsA 终端执行：`export DATA_ROOT="$RAW_ROOT/synthetic_city/data"` 后运行 `bash tools/download_advisor_synthpop_mi.sh`（脚本会写到 `$DATA_ROOT/reference/advisor_synthpop/mi.zip`）。

---

### 传输后在工作站验证

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export DATA_ROOT="$RAW_ROOT/synthetic_city/data"

# 1. PUMS — 应看到 csv_pmi.zip (~66MB) + csv_hmi.zip (~30MB)
ls -lh "$DATA_ROOT/detroit/raw/pums/pums_2022_5-Year/"

# 2. DHC — 应看到 dhc_2020_bg_michigan.csv.gz (~685KB)
ls -lh "$DATA_ROOT/detroit/raw/census/dhc_2020/"

# 3. ACS — 应看到 5 个 csv.gz
ls -lh "$DATA_ROOT/detroit/raw/census/acs/acs5_2022/"

# 4. LODES — 应看到 2 个 csv.gz
ls -lh "$DATA_ROOT/detroit/raw/census/lodes/"

# 5. 快速验证 PUMS
python3 -c "
import zipfile, pandas as pd
with zipfile.ZipFile('$DATA_ROOT/detroit/raw/pums/pums_2022_5-Year/csv_pmi.zip') as z:
    with z.open('psam_p26.csv') as f:
        df = pd.read_csv(f, nrows=5)
        print('OK: Columns:', list(df.columns)[:10])
        print('PUMA20 present:', 'PUMA20' in df.columns)
        print('PINCP present:', 'PINCP' in df.columns)
"
```

---

### 数据详情与变量说明

#### 数据 1：PUMS 2022 5-Year（Exp 0 + Exp 2 训练）

**已下载验证**：Person 499,880 行 × 290 列，Housing 254,227 行 × 238 列。SERIALNO join 成功率 95.3%。

**关键列说明**（PUMS person file）：

| 列名 | 含义 | 类型 | Exp 中角色 |
|------|------|------|-----------|
| **PUMA20** | 2020 版 PUMA 编码（不是 `PUMA`！） | str | 空间标识（训练条件），69 个唯一值 |
| AGEP | 年龄 | int (0-99) | 基本属性 / Exp 0 copula 变量 |
| SEX | 性别 | int (1=M, 2=F) | 基本属性 |
| RAC1P | 种族 | int (1-9) | 基本属性 |
| HISP | Hispanic origin | int (1=Not, 2+=Yes) | 基本属性 |
| PINCP | 个人收入（past 12 months, inflation-adjusted） | float | **Exp 0 copula 变量 / Exp 2 目标变量**，76K 缺失（儿童等） |
| SCHL | 教育程度 | int (1-24 等级) | Exp 2 目标变量 |
| ESR | 就业状态 | int (1-6) | Exp 2 目标变量 |
| PWGTP | Person weight | int | 加权用（计算分布时使用） |

**关键列说明**（PUMS housing file，与 person 通过 SERIALNO 关联）：

| 列名 | 含义 | 类型 | 用途 |
|------|------|------|------|
| SERIALNO | 家庭唯一 ID | str | 与 person file 关联 |
| **PUMA20** | 2020 版 PUMA 编码 | str | 空间标识（与 person 一致） |
| HINCP | 家庭收入 | float | Exp 0 copula 诊断（household 级别分析） |
| WGTP | Household weight | int | 加权用 |

> **注意**：2022 5-Year PUMS 同时有 `PUMA10` 和 `PUMA20` 两列（过渡期），我们统一用 `PUMA20`。
> 代码中如果有 `group_col="PUMA"`，需改为 `group_col="PUMA20"`。

---

#### 数据 2：2020 DHC at Block Group（Exp 1）

**已下载验证**：8,386 个 Block Groups，P12 (Sex by Age) + P5 (Hispanic × Race)。  
总人口 P12_001N 求和 = **10,077,331**，与 2020 Census 精确一致。  
落盘路径：`$DATA_ROOT/detroit/raw/census/dhc_2020/dhc_2020_bg_michigan.csv.gz`  
下载脚本：`tools/download_census_api.py`（已调试通过）。

---

#### 数据 3：ACS 2022 5-Year Summary Tables at Tract（Exp 2 验证 + Exp 3）

**已下载验证**：全 Michigan 3,017 个 tracts，5 张表。

| 表名 | 内容 | 文件 |
|------|------|------|
| B01001 | Sex by Age | `acs5_2022_B01001_tract_michigan.csv.gz` |
| B15003 | Educational Attainment (25+) | `acs5_2022_B15003_tract_michigan.csv.gz` |
| B19001 | Household Income (16 bins) | `acs5_2022_B19001_tract_michigan.csv.gz` |
| B20001 | Person Earnings (与 PINCP 对应) | `acs5_2022_B20001_tract_michigan.csv.gz` |
| B23025 | Employment Status (16+) | `acs5_2022_B23025_tract_michigan.csv.gz` |

落盘路径：`$DATA_ROOT/detroit/raw/census/acs/acs5_2022/`

---

#### 数据 4：导师合成人口（可选，对比参考）

**来源**：OSF https://doi.org/10.17605/OSF.IO/FPNC2 → 子项目 "US-Synthetic-Population (M-Z)" → `mi.zip`  
**大小**：~1.8GB（zip 内含 gpkg）  
**推荐**：在 wsA 上直接下载到目标路径，免去本地下载再传输。在 wsA 终端：`export DATA_ROOT="$RAW_ROOT/synthetic_city/data"` 后执行 `bash tools/download_advisor_synthpop_mi.sh`。

```python
# 解压后校验
import geopandas as gpd
gdf = gpd.read_file("mi.gpkg")  # 解压 mi.zip 后的文件名待确认
print(gdf.columns.tolist())
# 预期列：id, age, gender, hhold, htype, wp, urban, assigned, long, lat, geometry
print(len(gdf))  # 预期 ~1000 万（Michigan 人口）
```

---

#### 数据 5：LODES（可选，Exp 3 L4 验证）

**已下载**：RAC (4.4MB) + WAC (1.4MB)  
落盘路径：`$DATA_ROOT/detroit/raw/census/lodes/`

---

### 数据汇总

```
[已就绪] 数据 1: PUMS 2022 5Y (csv_pmi.zip + csv_hmi.zip)  → Exp 0 + Exp 2
[已就绪] 数据 2: DHC 2020 at BG (dhc_2020_bg_michigan)      → Exp 1
[已就绪] 数据 3: ACS 2022 5Y Tract (5 张表)                  → Exp 2 验证 + Exp 3
[可选] 数据 4: 导师合成人口 (mi.zip, ~1.8GB)，建议在 wsA 上运行 tools/download_advisor_synthpop_mi.sh → 对比参考
[已就绪] 数据 5: LODES (RAC + WAC)                           → Exp 3 L4 验证
```

---

## 实施顺序与时间线

```
Week 1: Exp 0 (Copula 诊断)
  ├── 下载全 Michigan PUMS（如果还没有）
  ├── 实现 copula 计算 + 异质性分析
  ├── 产出 diagnostic_summary.json
  └── 决策点：diffusion 是否值得继续
  
Week 2: Exp 1 (基本人口)
  ├── 下载 2020 DHC at BG for Michigan
  ├── 实现 IPF 或优化方法
  ├── 产出 base_pop_michigan_bg.parquet
  └── 内部验证（应为零/近零误差）

Week 3-4: Exp 2 (属性扩展)
  ├── 构建训练数据（全 Michigan PUMS → 条件-目标对）
  ├── 4 个消融条件 × 5 折 CV = 20 次训练
  ├── 产出 ablation_summary.json
  └── 决策点：哪个条件组合最优 + diffusion 是否超越 IPF baseline

Week 5: Exp 3 (验证)
  ├── 下载 ACS tract tables + LODES（如果还没有）
  ├── 用最优条件生成全 Michigan 合成人口
  ├── 四层验证
  └── 产出 validation_summary.json → 写入下一阶段计划
```

---

## 代码组织建议

建议将实验代码放在 `tools/` 目录下，与现有 PoC 脚本风格一致：

```
tools/
  exp0_copula_diagnostic.py        # Exp 0：独立脚本，不依赖 src/synthpop
  exp1_base_population.py          # Exp 1：DHC 下载 + IPF/优化
  exp2_attribute_extension.py      # Exp 2：训练 + 推理 + 消融
  exp3_validation.py               # Exp 3：多层验证
```

每个脚本的输出遵循现有约定：
- 小文件（JSON metrics）提交到 `outputs/` 供 PI review
- 大文件（parquet, model.pt）保留在 workstation，不提交

---

## 关键注意事项

1. **Exp 0 是门控实验**：如果 copula 无空间变异，Exp 2 的 diffusion 方案需要调整为 IPF + 全局 seed。不要跳过 Exp 0 直接做 Exp 2。

2. **Exp 1 的 DHC 数据是 2020 年的**：而 ACS/PUMS 是 2018-2022 的 5 年估计。存在时间偏移，验证时需注意，但不影响方法论。

3. **Exp 2 的训练-验证隔离**：训练只用 PUMS 个体数据。ACS Summary Tables 和 DHC 完全不参与训练，全部留给验证。这确保了验证的独立性。

4. **收入变量口径已确认：使用个人收入（PINCP）**。PUMS 中 PINCP 是 person-level income（past 12 months, inflation-adjusted）。验证时**不能用 ACS B19001（household income）做对标**——口径不同。应使用 **ACS B20001（Earnings for full-time year-round workers）** 或 **B20002（Median earnings by sex）** 做 tract 级别验证。注意 B20001 只覆盖有收入的人群（不含无收入者），PINCP 包含所有人（含 0 收入），需要在验证时对齐筛选条件。

5. **计算资源**：Exp 2 的 20 次训练（4 条件 × 5 折），每次约 1000 epoch，在 4060 GPU 上预计每次 5-15 分钟（取决于数据量和模型大小）。总计 ~2-5 小时 GPU 时间。

6. **全 Michigan 数据量**：PUMS person file 499,880 行（已验证）。Exp 2 的 batch_size=4096 意味着每个 epoch 约 120 步。这是可控的。

7. **PUMS 年份变更**：代码默认 `--pums_year 2023`，但实际下载的是 2022 5-Year。所有脚本运行时请显式加 `--pums_year 2022`。PUMA 列名为 `PUMA20`（不是 `PUMA`）。

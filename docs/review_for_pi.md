# Review：关于 Copula 叙事的实验证伪与重新定位

> **背景**：本文档基于三组新增诊断实验（ExpA/B/C），对 `findings.md` 中 copula 核心叙事提出质疑。所有数据均来自仓库中的 JSON 输出文件，可逐条核实。目的不是否定整个工作，而是**确保论文的 claim 经得起审稿人的最严格检验**。

---

## 0 ｜ 分歧的根源：两套实验基于不同的数据文件

**这是所有数字分歧的起点，必须首先厘清。**

`findings.md` 中的全部实验结果（Exp0/Exp2/Exp3/Exp4/Exp5），来自以下数据文件：

- 路径：`pums_2023_5-Year/psam_p26.zip`
- metadata 标注：`pums_year: 2023`
- 实际行数：**503,042**
- PUMA 列：**`PUMA`**（2010 年代编码）
- PUMA 数量：**68 个**

Partner 核查后确认：该文件实为旧版本，PUMA 编码与 2022 5-Year PUMS 不一致。正确文件为：

- 路径：`pums_2022_5-Year/csv_pmi.zip`（即 `psam_p26.csv`）
- 实际行数：**499,880**
- PUMA 列：**`PUMA20`**（2020 年代新编码）
- PUMA 数量：**69 个**

**两套数据产生了截然不同的实验结果：**

| 指标 | 旧数据（findings.md 来源） | 新数据（re-run 后） | 差异 |
|------|--------------------------|-------------------|------|
| Exp0 Copula TVD to global | 0.090 | **0.136** | +51% |
| Exp2 Raw Seed Joint TVD | 0.123 | **0.172** | +40% |
| Exp2 Diffusion best Joint TVD | 0.102 | **0.170** | +67% |
| Exp2 Oracle IPF Joint TVD | 0.060 | **0.113** | +88% |
| **Diffusion vs Seed 优势** | **17%** | **1.2%** | **消失** |

**数据核实路径**：
- 旧数据 Exp0：`outputs/_exp0_copula_mi_20260209T151039Z/diagnostic_summary.json` → `inputs.pums_person_zip: psam_p26.zip`, `n_rows: 503042`
- 新数据 Exp0：`outputs/_exp0_copula_mi_20260210T022027Z/diagnostic_summary.json` → `inputs.pums_person_zip: csv_pmi.zip`, `n_rows: 499880`
- 旧数据 Exp2：`outputs/_exp2_attrdiff_mi_fixpuma20_20260210T034137Z/run.metadata.json` → `pums_year: 2023`
- 旧数据 Baselines（5 折均值）：seed = (0.110+0.134+0.118+0.132+0.120)/5 ≈ **0.123**，ipf = **0.060**（与 findings.md 完全一致）

**因此，findings.md 的 "Global Seed IPF = 0.123" 实为旧数据的 raw training-fold average；旧数据下 diffusion (0.102) 的 17% 优势是真实的，但它建立在一个已被确认有误的数据文件之上。**

正确数据（PUMS 2022, PUMA20）下，diffusion 的优势几乎消失（0.170 vs 0.172）。ExpA/B/C 是为了诊断"为什么换了正确数据后优势消失"而设计的。

---

## 1 ｜ 原叙事的核心 claim

`findings.md` 第 9 行的叙事主线：

> **"联合结构在空间上是异质的、可学习的；条件化扩散模型能够捕获这种异质性，而传统方法和全局预训练方法从根本上不能。"**

这条主线由四根证据柱支撑：

| # | Claim | 关键数字 | 出处 |
|---|-------|---------|------|
| C1 | Copula 异质性真实存在 | TVD to global = 0.136 | Exp0 |
| C2 | 扩散模型能学到异质 copula | Joint TVD 比 IPF 低 17% | Exp2 |
| C3 | IPF 后处理摧毁联合结构 | Copula TVD +206% | Exp5 |
| C4 | Tract 精度的天花板是数据粒度 | AGEP TVD = 0.183 | Exp3 |

**我对 C1、C3、C4 没有异议。** 异质性存在、后处理有害、粒度有限——这三条的证据是充分的。

**我的质疑集中在 C2："扩散模型能学到异质 copula"。** 以下三个诊断实验表明这条 claim 不成立。

---

## 2 ｜ 证据一：findings.md 中 Exp2 数字的标注存在错误

`findings.md` 第 43 行将 "Global Seed IPF" 标注为 Joint TVD = 0.123。但查看代码 `tools/exp2_baselines_age_income.py` 第 292–312 行，实际有**两个**不同的基线：

```
tvd_joint_seed  = 0.172  ← 训练折原始联合分布，没有做任何 IPF
tvd_joint_ipf   = 0.113  ← 训练折 seed + 测试 PUMA 的 oracle 边际做 IPF
```

**数据来源**：`outputs/_exp2_attrdiff_mi_puma20_20260218T095102Z/baselines_age_income_joint.json`

5 折平均值：

| 基线 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | 5 折平均 |
|------|--------|--------|--------|--------|--------|---------|
| tvd_joint_seed | 0.157 | 0.184 | 0.167 | 0.184 | 0.168 | **0.172** |
| tvd_joint_ipf | 0.107 | 0.117 | 0.111 | 0.120 | 0.110 | **0.113** |

`findings.md` 中标注的 "Global Seed IPF = 0.123" 既不是 0.172 也不是 0.113，且标签 "IPF" 在 seed 基线上不准确。

**这导致 C2 的核心比较出错**：

- findings.md 声称 "Diffusion (0.102) 比 IPF (0.123) 低 17%"
- 实际情况：
  - Diffusion (0.170) vs Raw Seed (0.172) → 差距 1.2%，无统计显著性
  - Diffusion (0.170) vs Oracle IPF (0.113) → Diffusion **更差** 50%

---

## 3 ｜ 证据二：ExpA — PUMA one-hot 诊断（瓶颈定位）

**实验设计**：给模型最强的空间条件——69 维 PUMA one-hot 编码，直接告诉模型"这是哪个 PUMA"。只在**训练 PUMA 上评估**（最容易的场景，因为模型训练时见过这些 PUMA 的数据）。

**数据来源**：`outputs/_expA_puma_id_diag_20260218T120735Z/ablation_summary.json`

| 条件 | 评估集 | Joint TVD | Copula TVD |
|------|-------|-----------|------------|
| demo_race_puma（5 个聚合特征） | 训练+测试 holdout | 0.170 | 0.159 |
| demo_race_puma_id（69 维 one-hot） | **仅训练集** | 0.163 | 0.160 |

**解读**：

- Joint TVD 小幅改善（0.170 → 0.163，约 4%），但这部分来自训练 vs holdout 评估的难度差异（训练集本身就更容易），所以实际增益更小
- **Copula TVD 没有改善**（0.159 → 0.160），甚至微幅恶化
- 即便在模型已经看过训练数据的、最有利的条件下，copula 依然无法被学到

**结论**：瓶颈不在"缺少 PUMA 条件信号"。给模型最精确的空间标识也不能改善 copula。问题在模型架构或训练目标本身。

---

## 4 ｜ 证据三：ExpC — Copula 基线直接对比

**实验设计**：直接比较扩散模型的 copula 与训练折平均 copula（即"什么都不做"的 copula 基线），在完全相同的 holdout PUMA 上评估。

**数据来源**：`outputs/_exp2_attrdiff_mi_puma20_20260218T095102Z/copula_baseline_demo_race_puma.json`

5 折 overall 结果：

| 指标 | 训练平均 Copula | 扩散模型 Copula | 改善？ |
|------|----------------|----------------|-------|
| Copula TVD | **0.137** | 0.159 | **否，更差 15.7%** |
| Joint TVD | 0.172 | **0.170** | 微幅好 0.9% |
| Copula TVD improvement | — | **-0.0217** | 负值 = 扩散更差 |
| Joint TVD improvement | — | **+0.0016** | 几乎为零 |

**逐折检验**——扩散模型的 copula 在所有 5 折中都劣于训练平均：

| Fold | Baseline Copula TVD | Diffusion Copula TVD | Δ |
|------|---------------------|---------------------|---|
| 0 | 0.128 | 0.147 | -0.019 |
| 1 | 0.144 | 0.165 | -0.022 |
| 2 | 0.130 | 0.148 | -0.018 |
| 3 | 0.148 | 0.175 | -0.028 |
| 4 | 0.137 | 0.159 | -0.022 |

**所有 5 折一致：扩散的 copula 都比训练平均差。** 这不是噪声，是系统性的。

**根因分析**：连续 Gaussian DDPM 的生成过程本身引入了噪声。MSE 损失优化的是逐样本重建质量（"每个生成的人合理吗？"），不是聚合分布匹配（"69 个 PUMA 的联合分布各自对吗？"）。训练平均 copula 基于 ~40 万样本的直接统计（~80% × 500k），天然比每个 PUMA ~7,000 样本的生成估计更稳定。生成过程的噪声扰动了 copula 的精细结构。

---

## 5 ｜ 证据四：ExpB — Tract 级评估（价值边界）

**实验设计**：比较扩散模型 vs 全局平均在 tract 和 PUMA 两个尺度上的表现差异。

**数据来源**：`outputs/_expB_tract_eval_20260218T121135Z/expb_summary.json`

**PUMA 级（delta = diffusion - global，TVD 越小越好）**：

| 指标 | Global | Diffusion | Δ | 方向 |
|------|--------|-----------|---|------|
| Income TVD | 0.070 | 0.062 | -0.008 | 好 |
| SCHL TVD | 0.124 | 0.112 | -0.011 | 好 |
| ESR TVD | 0.043 | 0.030 | -0.013 | 好 |
| Joint TVD | 0.173 | 0.172 | -0.002 | 微幅好 |
| **Copula TVD** | **0.142** | **0.145** | **+0.003** | **差** |

**Tract 级**：

| 指标 | Global | Diffusion | Δ | 方向 |
|------|--------|-----------|---|------|
| Income TVD | 0.196 | 0.192 | -0.003 | 略好 |
| ESR TVD | 0.102 | 0.097 | -0.005 | 略好 |
| SCHL TVD | 0.392 | 0.396 | +0.004 | 略差 |

**结论**：扩散模型的增益**全部来自边际分布**（income/SCHL/ESR 的单变量分布更准确），而**不是来自 copula**（依赖结构反而更差）。模型擅长的是 P(income | age, sex, race)，不是空间异质的 copula。

---

## 6 ｜ 对 findings.md 原叙事的逐条审计

### C1：Copula 异质性真实存在 → **成立**

Exp0 数据（PUMS 2022, PUMA20, 69 PUMAs）：TVD to global mean = 0.136，pairwise TVD mean = 0.192。异质性是统计显著的。无异议。

### C2：扩散模型能学到异质 copula → **不成立**

- ExpC 证明扩散的 copula **比训练平均更差**（0.159 vs 0.137，5 折全劣）
- ExpA 证明即便给 PUMA one-hot 也无法改善 copula
- ExpB 证明增益来自边际而非 copula
- findings.md 的 "Joint TVD 比 IPF 低 17%" 基于错误标注（0.123 不对应任何基线）

### C3：IPF 后处理摧毁联合结构 → **成立**

Exp5 的 copula TVD +133%（新数据）是实质性的、方向明确的。无异议。

### C4：Tract 精度的天花板是数据粒度 → **成立**

Exp3 的 AGEP TVD = 0.183 作为数据源基线是合理的锚点。无异议。

---

## 7 ｜ 哪些是对的，哪些需要修正

**无需改变的**：

1. 分层架构（Layer 1 IPF + Layer 2 条件生成）在设计上是合理的
2. 扩散模型在**边际分布**上有稳定增益（income -12%, SCHL -9%, ESR -30% 相对全局）
3. IPF 后处理不可取的结论是正确的
4. 数据粒度天花板的论断是正确的
5. 对 Qian (2026) A1 假设的质疑仍然成立（copula 异质性存在是事实）

**需要修正的**：

1. **核心叙事主线需要重写**。"可学习的"这个 claim 被 ExpA/B/C 证伪了。叙事不能再是"扩散模型捕获了空间异质 copula"
2. **Exp2 的基线标注和数字需要更正**。"Global Seed IPF = 0.123" 对应的实际是 Oracle IPF（使用测试 PUMA 精确边际），不是"传统方法基准"
3. **"信息恢复率 84%"需要重新计算或删除**。原来的计算基于错误的基线数字
4. 6.2 节的方法论推论第 2 条（"空间条件是联合结构的关键信号"）在 copula 层面不成立

---

## 8 ｜ 可能的反驳与我的回应

### 反驳 1："ExpA 用训练集评估不公平，应该和 holdout 比"

回应：**这正是 ExpA 的设计意图**——给模型最大优势（训练集 + 精确 PUMA ID），看它能否在最有利条件下学到 copula。如果在最容易的场景都不行，那在更难的 holdout 场景更不可能行。这是一个**必要条件检验**：如果 P(最容易的场景下成功) = 否，则 P(困难场景下成功) = 否。

### 反驳 2："Copula TVD 不是唯一的评估指标"

回应：同意。但 `findings.md` 的整个叙事主线是围绕 copula 构建的（标题是"Copula 空间异质性"，核心贡献是"copula 可学习"）。如果我们承认 copula 不是核心指标，那叙事主线本身就需要重构。

### 反驳 3："模型的价值在 joint TVD，不在 copula TVD"

回应：Joint TVD = f(age 边际, income 边际, copula)。扩散在 joint 上的微弱优势（0.172 → 0.170）来自 income 边际的改善（-12%），而非 copula。这在 ExpB 中已经确认：边际指标全面改善，copula 指标反向恶化。如果核心 claim 是"学到了更好的 joint"，需要诚实地说明改善来源是边际而非依赖结构——这和原叙事的"copula 驱动"论断矛盾。

### 反驳 4："样本量 7000/PUMA 不够学 copula，不代表方法不对"

回应：部分同意。这确实可能是数据量的问题而非方法的问题。但这恰恰说明 claim "能学到" 在**当前数据条件下**不成立。论文需要说清楚：copula 异质性存在（Exp0），但在 ~7k 样本/PUMA 的 PUMS 数据上，当前 DDPM 架构无法将其利用为生成优势。这本身是一个有价值的发现。

---

## 9 ｜ 建议的叙事重构方向

以下是我认为可以诚实地支撑的叙事：

> **人口合成的联合结构问题是真实的**：copula 异质性存在（Exp0），IPF 后处理会摧毁联合结构（Exp5），数据粒度有明确天花板（Exp3）。
>
> **条件化扩散架构提供了一种新范式**：通过 demographic-conditional generation 实现了比全局平均更好的边际分布（income -12%, SCHL -9%, ESR -30%），并产出了具有个体级内部一致性的合成记录。
>
> **但当前架构未能利用 copula 异质性信号**：即便给予最精确的空间标识，连续 DDPM 在 copula 层面不优于训练平均（ExpA/C）。这揭示了一个面向整个社区的问题：**copula 异质性信号（TVD ~0.14）在 PUMS 样本量（~7k/PUMA）下可能低于连续扩散模型的生成噪声底线**——是信息理论的约束，不只是工程问题。

这条叙事不依赖"扩散打败 IPF"的 claim，而是：
1. 证明问题存在（C1 ✓）
2. 提供一种合理的方法框架并展示其边际增益
3. 诚实地报告方法在 copula 维度的局限，并给出解释
4. 将局限性本身转化为对社区有价值的 finding

---

## 10 ｜ 数据索引

本文档引用的所有原始数据文件，均可在仓库中逐条核实：

| 实验 | 文件路径 |
|------|---------|
| Exp0 | `outputs/_exp0_copula_mi_20260210T022027Z/diagnostic_summary.json` |
| Exp2 Baselines | `outputs/_exp2_attrdiff_mi_puma20_20260218T095102Z/baselines_age_income_joint.json` |
| Exp2 Ablation | `outputs/_exp2_attrdiff_mi_puma20_20260218T095102Z/ablation_summary.json` |
| ExpA | `outputs/_expA_puma_id_diag_20260218T120735Z/ablation_summary.json` |
| ExpB | `outputs/_expB_tract_eval_20260218T121135Z/expb_summary.json` |
| ExpC | `outputs/_exp2_attrdiff_mi_puma20_20260218T095102Z/copula_baseline_demo_race_puma.json` |
| Baselines 代码 | `tools/exp2_baselines_age_income.py` 第 292–312 行 |

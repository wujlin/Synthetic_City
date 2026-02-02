# 文献梳理与调研（二）：人口合成 × 扩散模型 × 时空建模 × 建筑物尺度空间化

- **项目**：Population dynamics
- **交付**：Partner 文献梳理（对应你给的 6 个主题清单）
- **版本**：v0.1（可持续迭代）
- **日期**：2026-01-17

> 说明：这是一个“可扩展的工作底稿”。每个主题都包含：核心问题拆解 → 可复用的对照表/图谱 → 可落地的技术路线启发 → 参考文献入口。

---

## 目录

- [1. 现有方法的“硬伤”诊断（问题-方法对照表）](#1-现有方法的硬伤诊断问题-方法对照表)
- [2. 扩散模型在结构化/表格数据的应用（方法演进图谱）](#2-扩散模型在结构化表格数据的应用方法演进图谱)
- [3. 条件扩散与约束生成（技术方案对比）](#3-条件扩散与约束生成技术方案对比)
- [4. 时空扩散模型（架构参考列表）](#4-时空扩散模型架构参考列表)
- [5. 建筑物尺度人口空间化（精度-方法-数据对照）](#5-建筑物尺度人口空间化精度-方法-数据对照)
- [6. 多源数据融合的生成模型（融合策略汇总）](#6-多源数据融合的生成模型融合策略汇总)
- [附录A：建议阅读顺序](#附录a建议阅读顺序)
- [附录B：参考文献清单（初版，可扩展）](#附录b参考文献清单初版可扩展)

---

## 1. 现有方法的“硬伤”诊断（问题-方法对照表）

### 1.1 方法族与常见工作流（统一语言）

- **SR / Re-weighting（合成重构）**：IPF / IPU / GREGWT / raking 等；通常依赖**种子表（seed）+ 边际（margins）**，先拟合后抽样/分配。
- **CO（组合优化）**：把“选哪些微观样本、怎么拼家庭/个体”变成一个优化问题（拟合目标 + 约束 + 惩罚）。
- **ML/深度生成**：VAE/GAN/BN/扩散等，从样本学习联合分布，再生成个体与家庭；优势是能生成“样本没见过、但人口中存在”的组合，但存在可行性/约束难题。

在人口合成里，常见要同时满足：

1) **宏观统计一致性**（分区年龄结构、性别、收入、职业等边际/联合）
2) **微观可行性**（家庭内部逻辑、人-户一致性、结构性零组合不能出现）
3) **覆盖稀有组合**（sampling zeros：样本里没有但真实存在的组合）

这些目标之间存在**结构性的冲突与取舍**（尤其在高维离散空间）。

---

### 1.2 问题-方法对照表（失效场景与根因）

> 读表方式：先看“失效场景”，再看“为什么（根因）”，最后看“是否根本性（很难靠小修小补解决）”。

| 失效/痛点 | SR/IPF/IPU（含IPU） | CO | ML/深度生成（VAE/GAN/扩散等） | 根因与可修复性判断 |
|---|---|---|---|---|
| **零单元格（零权重）**：目标边际要求某类组合>0，但种子/样本该 cell 为 0 | IPF 只能在已有支持集上“调权”；cell=0 就永远 0（除非人为加 ε） | 若候选池里无该组合，CO 也无解，只能“换库/加合成候选” | 生成模型可“补齐” sampling zeros，但会引入结构性零风险 | **根本性**：属于“支持集缺失”问题；仅靠算法微调不能凭空创造合法组合。可修复路径通常是：引入更丰富的先验/样本、或用生成模型扩充候选，但必须做可行性控制。 |
| **维度灾难（curse of dimensionality）**：属性一多，联合空间爆炸，很多 cell 稀疏/缺失 | IPF/交叉表直接爆炸；IPU计算/收敛变差 | 优化维度/约束数激增，搜索空间膨胀 | 深度生成在高维更“能学”，但训练数据不足时会过拟合/崩坏 | **根本性倾向**：高维稀疏是数据与表示的客观限制。可缓解：分层建模、图模型、低秩/嵌入、先生成再校准。 |
| **属性逻辑不一致**：例如“年龄<16 却就业=全职”，“婴儿有驾照”等 | 若 seed/样本中存在噪声或抽样/整数化造成非法组合，SR无法自然排除 | CO可硬约束排除，但约束过多会导致可行解稀少/无解 | 生成模型很容易“发明”非法组合（结构性零） | **根本性**：需要显式的可行域约束/规则系统。仅靠拟合边际无法保证逻辑一致。 |
| **人-户/层级一致性**：户属性与成员属性必须协调（户规模=成员数、家庭关系等） | 基础IPF多在户层或人层，跨层一致性难 | CO可在户-人联合层面做，但复杂度高 | 生成模型若不建层级结构，极易不一致 | **根本性**：层级结构属于“组合约束”。需要层级生成/分步生成/图结构建模。 |
| **整数化误差（integerization）**：权重转整数后边际偏离 | 常见问题，需用Truncate/CS/CW等修正 | CO可以直接在整数空间优化，但代价高 | 生成模型如果输出离散个体，本身是整数，但边际可能漂移 | **可修复**：属于工程型问题；但当约束很多时，整数化会放大矛盾。 |
| **局部最优/收敛不稳定** | 有时IPF/IPU在复杂约束下收敛慢/敏感 | CO可能陷局部最优，需启发式/退火等 | 深度生成训练不稳、样本量不足时效果波动 | **可修复**：通过更强优化器/正则/多次重启等缓解；但“支持集缺失、规则冲突”仍无解。 |
| **公平性/隐私**：小群体被抹平、或泄露风险 | SR复制样本权重可能泄露；小群体易被边际吞没 | CO同样可能复制样本 | 生成模型可用于隐私合成，但也可能记忆训练样本 | **需额外机制**：差分隐私、去记忆、评估框架；不是单一算法能自动保证。 |

---

### 1.3 哪些是“根本性硬伤”？（建议优先在方案层面规避）

**A. 支持集缺失（sampling zeros） vs 结构性零（structural zeros）**
- SR/CO 对“样本/种子没有的合法组合”天生无能为力。
- 生成模型能补 sampling zeros，但**很容易同时生成 structural zeros**（非法组合），需要可行性控制。

**B. 仅靠边际匹配无法保证可行性/逻辑一致性**
- 这类问题必须引入：规则系统、可行域投影、或层级结构建模。

**C. 高维稀疏导致“看起来都能跑，但本质在猜”**
- 指标（如单变量边际）可能很好看，但真实联合分布、稀有组合、规则一致性会崩。

> 与“硬伤”相对的是“工程问题”：整数化误差、收敛速度、局部最优等，通常通过工程改进可显著缓解。

---

## 2. 扩散模型在结构化/表格数据的应用（方法演进图谱）

### 2.1 表格数据的难点：离散 + 混合类型 + 规则

表格/人口属性的典型特点：

- **强离散性**（年龄段、职业、教育、婚姻、户类型）
- **混合类型**（连续收入 + 离散职业）
- **强约束**（逻辑规则、层级一致性）
- **长尾组合**（稀有但真实存在）

扩散模型要做表格生成，核心是：**怎么定义“加噪/去噪过程”来适配离散与混合类型**。

---

### 2.2 方法演进图谱（从“连续扩散”到“表格扩散”）

```mermaid
flowchart LR
  A[连续DDPM/Score-based (图像/连续信号)] --> B[离散扩散: Multinomial diffusion]
  B --> C[D3PM: 更一般的离散转移矩阵]
  A --> D[表格扩散1: 连续化/嵌入 + DDPM]
  D --> E[TabDDPM: 混合类型表格建模]
  D --> F[STaSy: score-based表格合成 + 训练策略]
  E --> G[可控表格扩散: 控制器/条件注入（RelDDPM等）]
```

**解释：**
- **Multinomial Diffusion / D3PM**：从理论上解决“离散状态空间的扩散过程怎么定义”。
- **TabDDPM / STaSy**：把“离散+连续+混合”落地到表格任务上，并在评测上对比 GAN/CTGAN/TVAE 等。
- **可控表格扩散（SIGMOD 2024 等）**：把“条件/约束”从训练阶段或推断阶段注入，追求“可控生成”。

---

### 2.3 离散属性怎么处理？（策略对照）

| 策略 | 直觉 | 优点 | 风险/坑 | 适用在人口合成？ |
|---|---|---|---|---|
| **one-hot + 连续扩散** | 把 one-hot 当连续向量加噪去噪 | 实现简单 | 容易生成非 one-hot（需要投影/采样） | 可作为 baseline，但必须加投影/后处理 |
| **嵌入（embedding）后做连续扩散** | 把离散类别映射到连续嵌入 | 表达力强，可处理高维类别 | 逆映射会引入偏差；可能出现非法组合 | 适合高维离散，但要配合约束/校准 |
| **离散扩散（Multinomial/D3PM）** | 在类别上直接定义转移矩阵 | 理论更正宗，天然离散 | 训练/采样成本、工程复杂度高 | 若人口属性全离散，值得重点考虑 |
| **混合：连续列连续扩散、离散列离散扩散** | 每类变量用合适扩散 | 表格最自然 | 需要设计联合去噪网络、对齐时间步 | 是“最好但最复杂”的路线 |
| **后验校准/约束修正（后处理）** | 先生成，再调边际/规则 | 实用、便于插拔 | 可能破坏联合分布；会引入偏置 | 实务中常用，推荐作为工程兜底 |

---

### 2.4 表格扩散代表作（快速要点）

- **TabDDPM**：以 DDPM 框架适配 tabular；强调在多数据集上相对强的合成质量。
- **STaSy**：score-based 表格合成；强调训练策略（self-paced / fine-tune）与三难困境（质量/多样性/时间）。
- **可控表格扩散（如 SIGMOD 2024 RelDDPM）**：把“控制条件”做成轻量 controller，推断阶段控制生成。

> 对我们项目的启示：表格扩散模型本体解决的是“生成质量/多样性”，但**人口合成真正难点**在“规则+宏观约束+层级一致性”。因此第 3 节（约束生成）是关键。

---

## 3. 条件扩散与约束生成（技术方案对比）

### 3.1 我们要解决的“约束”到底是什么？

人口合成里常见的约束分三类：

1) **硬约束（Hard constraints）**：必须满足，否则样本无效（结构性零、户人一致性、逻辑规则）。
2) **软约束（Soft constraints）**：允许一定误差，但要尽量逼近（区县年龄结构、收入分布、行业结构）。
3) **分布级约束（Distributional constraints）**：不针对单条记录，而针对总体统计（多边际、相关结构、尾部）。

扩散模型的优势是“可把约束当成采样问题的一部分”，但不同注入方式代价和效果差异很大。

---

### 3.2 技术方案对比表（训练期 vs 推断期；软约束 vs 硬约束）

| 类别 | 代表思路 | 怎么做 | 对宏观边际（soft） | 对规则/结构性零（hard） | 代价与风险 | 适合度 |
|---|---|---|---|---|---|---|
| **条件扩散（Training-time conditioning）** | 把区域/人群特征当条件输入 | 条件编码 + 去噪网络 | ✅（容易学） | ❌（不保证绝对满足） | 需要覆盖条件空间；新条件泛化不稳 | 中高 |
| **Classifier-Free Guidance (CFG)** | 用“条件score−无条件score”调控采样 | 推断期调 guidance strength | ✅（可调强度） | ❌（仍可能违规） | 强 guidance 会牺牲多样性、产生分布漂移 | 中 |
| **Loss-Guided Diffusion（LGD）** | 定义可微损失，推断期沿梯度修正 | 约束损失 L(x) 反传到采样 | ✅（非常灵活） | ⚠️（若规则可微/可投影） | 代理损失不准会“跑偏” | 高（作为插件） |
| **Trust Sampling（训练无关的稳健guidance）** | 在每步做多次小梯度步，控制信任区间 | 以方差/噪声水平控制步长/停止 | ✅ | ⚠️（同上） | 工程复杂，但更稳 | 高（若约束复杂） |
| **Posterior Sampling / DPS（把约束当观测/后验）** | 将“满足统计/观测”作为后验采样 | 推断期做近似后验采样 | ✅ | ⚠️（对硬规则需额外投影） | 要能写出观测模型/似然 | 中高 |
| **投影/修复（Projection/Repair）** | 每步或最终把样本投影到可行域 | 规则检查 + 最近邻/ILP 修复 | ✅（作为兜底） | ✅（硬规则可保证） | 可能破坏联合分布；需设计修复器 | 最高（工程必备） |

---

### 3.3 面向“宏观统计约束”的可实现写法（建议）

将宏观统计写成一个可微目标：

- 例如年龄段边际：\(\hat{p}(age=k)\) 与目标 \(p^*(age=k)\) 的差
- 用 **soft histogram / differentiable counting**（或 Gumbel-Softmax / straight-through）构造可微统计
- 损失可用：KL、Chi-square、MSE、Earth Mover’s Distance（离散分布）

然后在扩散采样中注入：

- CFG：把“条件=区域统计向量”的信息作为条件编码
- LGD/Trust Sampling：把统计差异作为 loss guidance

> 关键提醒：只用“总体边际损失”很容易出现“投机解”（把统计做对，但微观逻辑/相关性坏掉）。因此必须配合：
> - 联合统计/相关性约束（至少二阶）
> - 硬规则投影/修复
> - 分层生成（户→人）

---

## 4. 时空扩散模型（架构参考列表）

时空扩散模型的共同点：

- 把对象从“单条样本”扩展成“序列/图/轨迹/视频”
- 去噪网络从 MLP/CNN 扩展到 Transformer、GraphNet、时序 U-Net 等
- 关键在于：**如何表达时空依赖 + 如何插入条件/约束**

---

### 4.1 架构参考列表（可借鉴点）

| 任务对象 | 代表工作 | 核心结构 | 条件注入方式 | 可借鉴到人口合成的点 |
|---|---|---|---|---|
| **时间序列预测** | TimeGrad | AR + diffusion（每步采样） | 过去窗口/外生变量 | 适合“人口属性随时间演化”的生成器（动态人口） |
| **时间序列插补** | CSDI | conditional score model（mask条件） | 观测mask + 已观测值 | 适合“多源数据缺失”的统一生成框架（先插补再生成） |
| **时空图预测** | DiffSTG | STGNN + diffusion（UGnet） | 图结构/历史序列 | 适合“区域之间耦合”（通勤/OD/邻接影响） |
| **轨迹生成（人类移动）** | TrajGDM / DiffTraj | 时空编码 + diffusion | POI/区域条件、上下文 | 适合“活动链/出行链”层面的个体行为生成 |
| **城市规模合成移动数据** | WorldMove | diffusion + 多源城市特征 | 人口网格、POI、OD 等 | 典型“多源条件→受控生成”范式，值得直接对标 |
| **视频/高维时空** | Video diffusion / Imagen Video | 时空U-Net/级联扩散 | 文本/条件 | 给我们提供“分层/级联生成”的工程范式（粗到细） |

---

### 4.2 对人口合成的启示：时空扩散“怎么用”

- **静态人口合成**：扩散模型生成“属性向量”即可。
- **动态人口（年度/季度/日内）**：借鉴 TimeGrad/CSDI 的“时间条件 + mask 条件”。
- **人口—出行耦合**：借鉴 DiffSTG/WorldMove，把“区域图 + OD”作为条件，生成个体轨迹或活动链，再反推人口属性一致性。

---

## 5. 建筑物尺度人口空间化（精度-方法-数据对照）

### 5.1 先统一“精度”口径

建筑物尺度空间化常见评价口径包括：

- **空间分辨率（resolution）**：30m/100m 网格，或“building-level（每栋/每入口）”。
- **误差指标（accuracy）**：MAE/RMSE/R²/NAE（normalized absolute error）等。
- **约束口径**：是否仅分配“总人口”，还是还能分到年龄/性别/职业。

> 注意：很多“高分辨率产品”并不等于“建筑物尺度的高精度”，因为训练/校准往往仍依赖较粗的统计单元。

---

### 5.2 精度-方法-数据对照（代表工作与可达上限）

| 研究/产品 | 空间单元 | 使用数据 | 方法概述 | 报告的精度/现象 | 主要瓶颈 |
|---|---|---|---|---|---|
| **Pajares et al. 2021（IJGI）** | building access（建筑入口） | 开放数据 + 过期普查 + 建筑 | “自底向上（新开发区）+ 自顶向下（dasymetric）”混合流程 | 强调标准化流程与可迁移性 | 建筑用途/入住率不确定；普查时效 |
| **Wang et al. 2022（Remote Sensing）** | building-level | 多源地理特征（RF） | 随机森林回归估计建筑人口 | R²中等（示例：0.44） | 特征可得性、泛化性、真值获取 |
| **Vergara 2024（ISPRS Archives）** | building-level | LiDAR/高程 + 建筑体积 + 用地等 | 用建筑体积等 3D 特征估计人口 | 报告 NAE≈0.133、R²≈0.976（特定城市/设置） | 高密度区系统性偏差；3D数据成本 |
| **Meta/Facebook HRSL** | 30m 网格 | 高分辨率影像 + 普查（建物检测/ML） | 在 30m tile 上估计人口密度 | 覆盖多国、分辨率高 | 对“建筑物用途/昼夜人口”不敏感；部分国家覆盖缺失 |
| **WorldPop（Unconstrained 100m）** | ~100m 网格（3 arc-second） | 多源 covariates + RF dasymetric | 在 100m 上分配人口 | 全球覆盖、公开 DOI | 仍是网格级；建筑级需进一步下推 |
| **LandScan HD** | ~90m（3 arc-second） | 多源地表与建筑提取 | bottom-up 细尺度人口分布 | 强调城市/区域级高分辨率 | 许可/可得性、方法黑箱、验证难 |
| **GHSL / GHS-POP + POP2G** | 100m/250m 等 | 建成区/建筑掩膜 + 普查 | 全球统一建模 + 工具化下推 | 公开、可复现 | 建筑掩膜在农村识别不足等系统偏差 |

---

### 5.3 “最高精度”通常卡在哪里？（国际共识型瓶颈）

1) **真值数据难**：建筑物实际居住人口很少有全面标注；很多研究只能用代理真值。
2) **建筑用途与入住率**：同体量建筑人口差异极大（住宅/办公/空置/短租）。
3) **昼夜人口/时变性**：静态分配很难同时解释日间与夜间分布。
4) **跨城/跨国泛化**：同一模型迁移到不同城市，误差会大幅漂移。

> 对项目落地：如果目标是“建筑物尺度的人口分配 + 可解释”，建议优先走“公开流程 + 可控特征 + 可修复约束”的路线，而不是追求单篇论文的最高 R²。

---

## 6. 多源数据融合的生成模型（融合策略汇总）

### 6.1 单一生成框架融合异构数据：三种主流范式

| 融合位置 | 策略 | 典型做法 | 优点 | 风险/坑 | 对人口合成的映射 |
|---|---|---|---|---|---|
| **输入侧融合（early fusion）** | 多模态编码后拼接条件 | 遥感CNN特征 + POI embedding + OD/图特征 → 条件向量 | 简单直接，工程易做 | 模态缺失时不稳；尺度对齐困难 | 适合“区域条件→个体属性” |
| **中间融合（cross-attention / co-attention）** | 去噪网络内部跨注意力融合 | 主模态token 与 辅助模态token 做 cross-attn | 表达力强，能学复杂交互 | 工程复杂，训练更难 | 适合“轨迹/时空 + 属性”的联合生成 |
| **输出侧融合（late fusion / calibration）** | 先生成，再用观测/统计校准 | 生成个体后，用边际/约束投影修正 | 灵活、可插拔 | 可能破坏联合分布；需权衡 | 非常适合“人口合成必须满足边际” |

---

### 6.2 多模态/多源扩散的典型例子（可借鉴结构）

- **WorldMove**：用人口网格、POI、OD 等多源公开数据作为条件，生成城市尺度合成轨迹数据（扩散生成）。
- **OD 网络扩散生成（Graph diffusion for OD）**：把 OD 矩阵当图生成对象，分两阶段生成拓扑与权重，适合“城市特征→OD结构”。
- **遥感场景（SAR+光学）扩散融合**：例如云移除任务，用 SAR 融合光学，体现“模态互补 + attention 融合”。

> 对项目启发：人口合成很可能也需要“多阶段/级联” —— 先生成可行的“人-户结构”，再生成属性细节，再做空间化（建筑/网格），最后在统计层面校准。

---

### 6.3 面向本项目的“融合路线”建议（可直接变成技术方案）

**建议把多源数据分成三层条件：**

1) **宏观统计层（必须满足）**：分区边际（年龄/性别/收入/行业…）
2) **空间环境层（解释差异）**：遥感/建成环境/建筑用途/道路可达性/POI
3) **流动与行为层（约束相关结构）**：OD/通勤矩阵/信令统计/出行链

对应生成框架：

- **Base diffusion**：学习“无条件的可行人口-户结构分布”（或弱条件）
- **Guidance/Controller**：推断期注入区域统计与多源条件（CFG/LGD/Trust sampling/控制器）
- **Projection/Repair**：把硬规则（结构性零、人-户一致性）做成投影或修复器
- **Macro calibration**：最后做一次边际校准（必要时）

---

## 附录A：建议阅读顺序

如果你希望后续把这 6 组内容“串成一条可实现路线”，推荐按以下顺序读与做：

1) **先读 1：硬伤诊断**（明确“支持集缺失/结构性零/层级一致性”是根问题）
2) **再读 2：表格扩散**（明确离散/混合类型怎么建模）
3) **核心读 3：约束生成**（决定可控生成路线：训练期条件？推断期guidance？投影修复？）
4) **补读 4：时空扩散**（为后续做动态人口/轨迹耦合做储备）
5) **补读 5：建筑尺度空间化**（明确“精度上限与瓶颈”，避免盲目追最高 R²）
6) **最后读 6：多源融合**（把遥感/POI/信令/OD纳入统一条件框架）

---

## 附录B：参考文献清单（初版，可扩展）

> 说明：这里列的是**本次梳理中最“承重”的入口文献/产品**，便于你后续按主题继续扩充。

### B.1 人口合成/采样零与结构性零

- Chapuis, K., Taillandier, P., & Drogoul, A. (2022). *Generation of Synthetic Populations in Social Simulations: A Review of Methods and Practices*. JASSS, 25(2). DOI: 10.18564/jasss.4762. PDF: https://www.jasss.org/25/2/6.html
- Ye, X., Konduri, K. C., Pendyala, R. M., Sana, B., & Waddell, P. (2009). *A Methodology to Match Distributions of Both Household and Person Attributes in the Generation of Synthetic Populations* (IPU). TRB Annual Meeting. (ResearchGate入口) https://www.researchgate.net/publication/228963837
- Kim, E.-J., & Bansal, P. (2022). *A Deep Generative Model for Feasible and Diverse Population Synthesis* (preprint). PDF: https://arxiv.org/pdf/2208.01403
- Kang, J., Kim, Y., Imran, M. M., Jung, G.-s., & Kim, Y. B. (2023). *Generating Population Synthesis Using a Diffusion Model*. Winter Simulation Conference. PDF: https://informs-sim.org/wsc23papers/247.pdf
- Garrido, S., Borysov, S. S., Pereira, F. C., & Rich, J. (2020). *Prediction of rare feature combinations in population synthesis: Application of deep generative modelling*. Transportation Research Part C, 120, 102787. DOI: 10.1016/j.trc.2020.102787. (arXiv入口) https://arxiv.org/abs/1909.07689

### B.2 离散/表格扩散模型

- Hoogeboom, E., Nielsen, D., Jaini, P., Forré, P., & Welling, M. (2021). *Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions*. arXiv:2102.05379. https://arxiv.org/abs/2102.05379
- Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM). NeurIPS 2021 / arXiv:2107.03006. https://arxiv.org/abs/2107.03006
- Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML 2023. arXiv:2209.15421. https://arxiv.org/abs/2209.15421
- Kim, J., Lee, C., & Park, N. (2023). *STaSy: Score-based Tabular data Synthesis*. ICLR 2023 Spotlight. arXiv:2210.04018. https://arxiv.org/abs/2210.04018
- Liu, T., Fan, J., Tang, N., Li, G., & Du, X. (2024). *Controllable Tabular Data Synthesis Using Diffusion Models*. Proc. ACM Manag. Data 2(1) (SIGMOD). DOI: 10.1145/3639283. PDF: https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/SIGMOD24-Gen.pdf

### B.3 可控/约束扩散（推断期guidance）

- Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. arXiv:2207.12598. https://arxiv.org/abs/2207.12598
- Song, J., Zhang, Q., Yin, H., Mardani, M., Liu, M.-Y., Kautz, J., Chen, Y., & Vahdat, A. (2023). *Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation*. ICML 2023. https://proceedings.mlr.press/v202/song23k.html
- Huang, W., Jiang, Y., Van Wouwe, T., & Liu, C. K. (2024). *Constrained Diffusion with Trust Sampling*. NeurIPS 2024. arXiv:2411.10932. https://arxiv.org/abs/2411.10932
- Chung, H., Kim, J., McCann, M. T., Klasky, M. L., & Ye, J. C. (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems* (DPS). ICLR 2023 / arXiv:2209.14687. https://arxiv.org/abs/2209.14687

### B.4 时空扩散（时间序列/图/轨迹/视频）

- Rasul, K., Seward, C., Schuster, I., & Vollgraf, R. (2021). *Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting* (TimeGrad). arXiv:2101.12072. https://arxiv.org/abs/2101.12072
- Tashiro, Y., Song, J., Song, Y., & Ermon, S. (2021). *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*. NeurIPS 2021 / arXiv:2107.03502. https://arxiv.org/abs/2107.03502
- Wen, H., Lin, Y., Xia, Y., Wan, H., Wen, Q., Zimmermann, R., & Liang, Y. (2023). *DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models*. SIGSPATIAL 2023. DOI: 10.1145/3589132.3625614. arXiv:2301.13629. https://arxiv.org/abs/2301.13629
- Chu, C., Zhang, Y., et al. (2024). *Simulating human mobility with a trajectory generation framework based on diffusion model (TrajGDM)*. IJGIS. PDF: https://giserwang.github.io/papers/IJGIS-2024-01.pdf
- Yuan, Y., Zhang, Y., Ding, J., & Li, Y. (2025). *WorldMove, a global open data for human mobility*. arXiv:2504.10506. https://arxiv.org/abs/2504.10506
- Ho, J., Saharia, C., Chan, W., Fleet, D. J., Norouzi, M., & Salimans, T. (2022). *Video Diffusion Models*. (ACM入口) https://dl.acm.org/doi/10.5555/3600270.3600898
- Ho, J., et al. (2022). *High Definition Video Generation with Diffusion Models* (Imagen Video). arXiv:2210.02303. https://arxiv.org/abs/2210.02303

### B.5 建筑物尺度人口空间化与数据产品

- Pajares, E., Muñoz Nieto, R., & others. (2021). *Population Disaggregation on the Building Level Based on Outdated Census Data*. IJGI 10(10):662. https://www.mdpi.com/2220-9964/10/10/662
- Vergara, K. A. (2024). *BUILDING-LEVEL POPULATION ESTIMATION USING LIDAR-DERIVED BUILDING VOLUME DATA*. ISPRS Archives. https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/453/2024/
- Meta / Facebook HRSL: High Resolution Population Density Maps docs. https://ai.meta.com/ai-for-good/docs/high-resolution-population-density-maps-demographic-estimates-documentation/
- WorldPop (100m, unconstrained): https://hub.worldpop.org/geodata/summary?id=6324
- LandScan HD（~90m）：ORNL LandScan data sets介绍（PDF）https://www.ornl.gov/file/landscan-data-sets/display
- GHSL POP2G tool（人口下推工具）：https://ghsl.jrc.ec.europa.eu/download.php?ds=pop

### B.6 多源融合扩散（例子）

- Li, Z., et al. (2023). *Complexity-aware Large Scale Origin-Destination Network Generation via Diffusion Model*. arXiv:2306.04873. https://arxiv.org/abs/2306.04873
- (遥感示例) *Multimodal Diffusion Bridge with Attention-Based SAR Fusion for Satellite Image Cloud Removal*. arXiv:2504.03607. https://arxiv.org/abs/2504.03607


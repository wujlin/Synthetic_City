# Core Narrative & Insights

> 本文件梳理研究的核心叙事主线——从问题识别到实验发现到理论推论——供组织 essay 时参照。不是实验报告的重复，而是提炼"发现了什么，这意味着什么"。

---

## 叙事主线（一句话）

**人口合成不是边际拟合问题，而是联合结构重建问题；联合结构在空间上是异质的、可学习的；条件化扩散模型能够捕获这种异质性，而传统方法和全局预训练方法从根本上不能。**

---

## 1 ｜ 被忽视的鸿沟：社会经济属性的联合分布

既有方法（Lin 2024, Jiang et al. 2024）已能以零误差生成 Block Group 级的人口统计学骨架（age × sex × race）。但这是一个 tautology——用 DHC 约束求解，再验证约束是否满足。

真正的空白在**社会经济属性**（income, education, employment）。这些变量的公开数据仅到 PUMA 级（~100k 人），而目标粒度是 tract / BG（~4k 人）。传统 IPF 的做法是用全州 PUMS 的联合分布作为 seed，隐含假设：**所有区域共享同一联合结构**。

这个假设是否成立？没有人检验过。

## 2 ｜ 第一个发现：Copula 空间异质性是真实的

**Exp0** 直接度量了这个假设。对 Michigan 68 个 PUMA 的 age × income copula（秩变换后的联合结构，剥离边际效应）做 pairwise TVD：

| 指标 | 值 | 含义 |
|------|------|------|
| TVD to global mean | **0.090** | 平均 9% 的概率质量需要重新分配 |
| TVD to global P90 | 0.142 | 10% 的 PUMA 偏离超过 14% |
| Pairwise TVD mean | 0.128 | 任意两个 PUMA 之间平均差 13% |

这不是抽样噪声——每个 PUMA 约 7,000 个样本，bootstrap 置信区间远小于 0.090。

**这个发现的意义**：它直接否定了传统 IPF 和 Qian et al. (2026) TL-VAE 的共同隐含假设——copula 全局不变（Qian 论文的 Assumption A1 显式写出了这一假设）。也就是说，大学城 PUMA 里"年轻+低收入"的高频组合，和郊区 PUMA 里"中年+高收入"的高频组合，不是边际不同造成的，而是**配对方式本身就不一样**。

## 3 ｜ 第二个发现：扩散模型能学到异质联合结构

**Exp2** 的消融实验揭示了条件信息的价值层级：

| 条件 | Joint TVD | 对比 |
|------|-----------|------|
| Independence（零信息） | 0.316 | — |
| Global Seed IPF（传统方法） | 0.123 | — |
| Diffusion: demo_only | 0.113 | 比 IPF 好，但有限 |
| Diffusion: demo_race | 0.112 | Race 贡献温和 |
| **Diffusion: demo_race_puma** | **0.102** | **比 IPF 低 17%** |
| Oracle IPF（不可达上界） | 0.060 | — |

关键 insight 不是"-17%"这个数字本身，而是它的**来源**：

- demo_only → demo_race 的收益微弱（-2%），因为种族信息在 PUMA 尺度被空间位置部分吸收
- demo_race → demo_race_puma 的收益巨大（-8%），**因为 PUMA ID 直接编码了空间位置，让模型有机会学到局部 copula**

换算到信息恢复率：扩散模型在"零信息 → 完美信息"的路上走了 **84%**，传统 IPF 走了 75%。差距来自 copula 异质性——正是 Exp0 发现的那 9%。

## 4 ｜ 第三个发现：IPF 后处理摧毁联合结构

一个自然的想法：既然 tract 级精度受限于 PUMA 分辨率，能否用 ACS 的 tract 边际对合成人口做 IPF post-alignment？

**Exp5** 给出了毁灭性的答案：

| 指标 | 后对齐前 | 后对齐后 |
|------|---------|---------|
| Copula TVD | 0.109 | **0.333**（+206%） |
| Joint TVD | 0.125 | **0.296**（+137%） |
| Cosine Sim | 0.976 | 0.902 |

更致命的是：**后对齐后，扩散模型和全局 seed 的 cosine similarity 仅差 0.004**——扩散的优势被完全抹平。

这个发现揭示了一个深层的方法论冲突：**IPF 保证边际正确，代价是打碎联合结构；扩散模型的核心价值恰恰是联合结构。二者在逻辑上不兼容。** 这不是工程调参能解决的，而是问题本身的数学结构决定的——给定边际的联合分布有无穷多个（non-identifiability），IPF 总是收敛到信息最少的那一个。

## 5 ｜ 第四个发现：Tract 级精度的结构性天花板

**Exp3** 将合成人口聚合到 tract，与 ACS 2022 做外部对比：

| 变量 | Tract TVD | 诊断 |
|------|-----------|------|
| AGEP | 0.183 | **数据源基线**（DHC 2020 vs ACS 2022 时间差） |
| PINCP | 0.211 | 仅比基线高 0.028——扩散几乎没有引入额外误差 |
| SCHL | **0.403** | 教育的极端空间聚类，PUMA 分辨率无法区分 |

AGEP 的 0.183 是最关键的数字——它是 Layer 1 中 BG 级**精确**的变量，聚合到 tract 后仍有这么大的 TVD，完全是数据源差异设定的噪声底线。在这个基准下，PINCP 仅高 0.028，说明扩散模型在其可控范围内表现良好。

SCHL 的 0.403 则暴露了 **PUMA→tract 分辨率落差**的结构性限制：同一 PUMA 内大学城 tract 和工业区 tract 的教育分布截然不同，PUMA 级模型无法区分。这不是模型缺陷，而是数据粒度的天花板。

## 6 ｜ 这些发现共同指向什么

### 6.1 对现有文献的诊断

| 方法 | 核心假设 | 本研究的证据 |
|------|---------|-------------|
| 传统 IPF | 全局 seed copula 适用于所有区域 | Exp0: TVD=0.090，异质性真实存在 |
| Qian 2026 TL-VAE | Assumption A1: copula 全局不变 | 同上；且 Qian 无空间 holdout 验证 |
| Tract-level post-alignment | IPF 可改善 tract 精度 | Exp5: 联合结构崩溃，copula TVD +206% |

### 6.2 方法论推论

1. **联合结构 > 边际精度**：评估合成人口不应仅看边际 TVD，而应重点关注联合分布（copula TVD, joint TVD）。边际正确但联合错误的人口在下游应用（ABM、通勤模拟、政策评估）中会产生系统性偏差。

2. **空间条件是联合结构的关键信号**：PUMA ID 在消融中贡献了最大收益，说明"这个人住在哪"是理解其社会经济属性配对模式的最重要信息。这为下一步引入更丰富的空间条件向量（sub-PUMA 级）提供了实验依据。

3. **生成与后处理不可共存**：扩散模型和 IPF 后处理在数学上冲突。正确的路径是将约束前移到生成过程中（guidance / projection），而非生成后做边际修正。

4. **精度天花板是数据粒度的函数**：PUMA 级是当前公开数据的天花板。突破需要 sub-PUMA 级的条件信号——这正是多源数据融合（LODES、property data、mobile data）的切入点。

### 6.3 与竞争方法的差异化定位

**vs. Qian et al. (2026) TL-VAE**：

Qian 的方法先在全州 PUMS 上预训练 VAE，再 freeze decoder、优化 latent matrix 匹配 tract 边际。其 Assumption A1 显式假设 copula 全局不变。我们的 Exp0 直接否定了这个假设（TVD=0.090），而条件化扩散通过 PUMA ID 让 copula 随空间变化——这是**结构性的方法差异**，不是"用扩散替代 VAE"的技术替换。

此外，Qian 的验证是 self-similarity protocol（5% subsample → 生成 → 与 95% 比较），没有空间 holdout。我们的 5-fold PUMA holdout 验证了模型在**未见过的空间单元**上的泛化能力。

**vs. MIGRATE (Agostini et al. 2025)**：

MIGRATE 用 IPF 融合 Infutor（有偏但个体级）+ Census（可靠但聚合级）。其 IPF 成功的前提是 Infutor 提供了真实的个体级空间行为信号——IPF 只做 de-biasing。我们没有 SES 维度的"Infutor"，但移动数据可以提供 CBG/POI 级的人群活动特征作为空间条件信号，进入扩散模型训练。思路从"IPF de-biasing"变为"conditional generation"。

---

## 7 ｜ 论文的核心叙事弧

```
[问题]
既有方法能生成人口统计骨架（零误差），
但社会经济属性的联合分布是空白——
不是没人尝试，是数据粒度不匹配。

         ↓ 那联合结构在空间上是均匀的吗？

[发现 1] Copula 异质性真实存在（TVD=0.090）
         → 否定全局不变假设

         ↓ 能学到吗？

[发现 2] 条件扩散能捕获异质 copula
         → Joint TVD 0.102, 比 IPF 低 17%
         → PUMA ID 是最关键条件（84% 信息恢复）

         ↓ 能用后处理改善 tract 精度吗？

[发现 3] IPF 后处理摧毁联合结构
         → Copula TVD +206%, 扩散优势归零
         → 生成与后处理不兼容

         ↓ 那 tract 精度的上界在哪？

[发现 4] 数据源基线 = 0.183
         → PINCP 仅高 0.028, 扩散本身表现好
         → SCHL 天花板来自空间分辨率，非模型

         ↓ 突破天花板的路径？

[展望] Sub-PUMA 条件向量
       → 多源数据（LODES / property / mobile）
       → 更丰富的空间语境驱动联合结构生成
```

## 8 ｜ 核心数字速查表

| 量 | 值 | 出处 | 角色 |
|----|-----|------|------|
| Copula 异质性 | TVD = 0.090 (mean) | Exp0 | 问题的存在性证明 |
| Independence baseline | Joint TVD = 0.316 | Exp2 | 零信息下界 |
| Global Seed IPF | Joint TVD = 0.123 | Exp2 | 传统方法基准 |
| **Diffusion (best)** | **Joint TVD = 0.102** | Exp2 | 本方法 |
| Oracle IPF | Joint TVD = 0.060 | Exp2 | 不可达上界 |
| 信息恢复率 | Diffusion 84% vs IPF 75% | Exp2 | 方法定位 |
| Cosine similarity | 0.976 (mean, 68 PUMAs) | Exp4 | 端到端精度 |
| Post-alignment 代价 | Copula TVD +206% | Exp5 | 方法论结论 |
| 数据源基线 | Tract AGEP TVD = 0.183 | Exp3 | 精度天花板锚点 |
| PINCP 额外误差 | 0.028 above baseline | Exp3 | 模型表现诊断 |
| SCHL tract TVD | 0.403 | Exp3 | 分辨率天花板 |
| Layer 1 精度 | BG age×sex×race = 0 error | Exp1 | 准入门槛 |

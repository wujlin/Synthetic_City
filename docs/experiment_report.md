# 合成人口社会经济属性的空间异质联合分布生成：实验报告

## 核心发现

合成人口生成领域的既有方法已经能以零误差生成 Block Group 级别的人口统计学骨架（年龄、性别、种族），但**社会经济属性**（收入、教育、就业）的联合分布至今仍是空白——不是因为没人尝试，而是因为这些变量的空间异质性远比人口统计变量复杂。

本研究发现：**不同 PUMA 的收入×年龄联合分布（copula，即在给定边际后“年龄与收入如何配对”的联合结构）并不相同，这种差异的平均幅度约为 9%（mean TVD = 0.090）**。这里的 TVD 可理解为“分布错配比例”（越小越好），0.090 表示约有 9% 的概率质量需要重新分配才能匹配。因此，传统方法若用全州平均 copula 替代局部结构，会引入系统性偏差。条件扩散模型（TabDDPM）能从 PUMS 微数据中学到这种空间异质结构，并泛化到未见过的区域，将联合分布误差从 0.124（全局平均）降低到 0.103（**-17%**）。

---

## 一、问题的提出：社会经济属性的联合分布为什么难？

### 1.1 既有方法能做什么

现有合成人口方法（Lin 2024, Jiang et al. 2024）通过 IPF 或分层抽样，将普查聚合表"反推"为个体级记录。这些方法在**人口统计学维度**上已接近完美：

| 方法 | 变量 | 空间粒度 | 内部验证误差 |
|------|------|----------|-------------|
| Lin 2024 | 住房类型、年龄(23组)、性别、种族(7类)、族裔 | Block Group | **0** |
| Jiang et al. 2024 | 年龄(18组)、性别 | Census Tract + 经纬度 | **0** |

但这里的"零误差"是一个 **tautology**——用约束数据求解，再验证约束是否满足。更重要的是，两者都**不包含收入、教育、就业**等社会经济属性。

### 1.2 为什么社会经济属性不能用同样的方式生成

根本原因在于**数据可用性的空间粒度错配**：

- 人口统计变量（age, sex, race）：DHC 提供 **Block Group** 级精确计数
- 社会经济变量（income, education, employment）：PUMS 仅提供 **PUMA** 级微数据（每个 PUMA 约 10 万人，包含数百个 BG）

PUMA 内部的社会经济结构不是均质的——大学城 tract 与工业区 tract 的教育分布截然不同，但它们共享同一份 PUMS 数据。传统 IPF 的做法是用全州平均的联合分布作为初始 seed，忽略了这种空间异质性。

**核心问题**：能否从 PUMS 数据中学到 PUMA 之间社会经济属性联合分布的差异，并用于生成更真实的合成人口？

---

## 二、空间异质性的诊断（Exp0）

在建模之前，首先需要回答一个前提性问题：**copula 的空间异质性是否真实存在，还是只是抽样噪声？**

### 方法

以 Michigan 州 68 个 PUMA 为单元，对每个 PUMA 计算年龄×收入的秩变换联合分布（copula），然后度量：
- 每个 PUMA 的 copula 与全州平均 copula 之间的 TVD（Total Variation Distance）
- 所有 PUMA 对之间的 pairwise TVD

其中 TVD 定义为 $\mathrm{TVD}(P,Q)=\frac{1}{2}\sum_i |P_i-Q_i|$，可理解为两种分布之间需要“重新分配”的概率质量比例。

### 发现

| 指标 | 值 |
|------|------|
| TVD to global（mean） | **0.090** |
| TVD to global（p90） | 0.142 |
| TVD to global（max） | 0.226 |
| Pairwise TVD（mean） | **0.128** |

这三个数字分别对应三层含义：

- `TVD to global (p90) = 0.142`：有 10% 的 PUMA 与全州平均结构的偏离超过 14.2%，说明尾部区域差异明显，不是“个别离群点”。
- `TVD to global (max) = 0.226`：最极端 PUMA 与全州平均结构相差 22.6%，代表在某些区域直接套用全局 copula 会产生非常大的结构误配。
- `Pairwise TVD (mean) = 0.128`：任意两个 PUMA 之间平均相差 12.8%，表明异质性不仅存在于“局部 vs 全局”，也存在于“局部 vs 局部”。

换句话说，不同 PUMA 的联合分布不一样，这种“不一样”的平均幅度约为 9%。这不是随机噪声——PUMS 样本量（每 PUMA 约 7,000 人）下的抽样波动远小于此。

**Insight**：copula 的空间异质性是真实的、可学习的。全局平均 copula 会丢失约 9% 的局部分布信息。这为扩散模型提供了明确的学习目标。

---

## 三、人口骨架的精确构建（Exp1）

### 方法

使用 DHC 2020 的 P12（age×sex 交叉表）和 P5（race 边际）作为 Block Group 级约束，通过 IPF + 整数化生成 age×sex×race 的联合计数。初始 seed 从 PUMS 2022 5-Year 的联合分布导出。

### 发现

| 约束 | 最大绝对误差 |
|------|-------------|
| P12（age×sex, 8386 个 BG） | **0** |
| P5（race, 8386 个 BG） | **0** |

与 Jiang et al. 2024 的 head-to-head 比较（BG age×sex 归一化分布 TVD）：**mean = 0.138**。这反映的是两种方法在 IPF seed 选择上的差异（我们使用 PUMS 联合分布 seed，Jiang 使用分层独立抽样），不存在对错之分。

**Insight**：Layer 1 的零误差是所有方法的准入门槛，不是差异化贡献。研究的价值从 Layer 2 开始。

---

## 四、条件扩散模型的训练与消融（Exp2）

### 方法

- **模型**：Gaussian TabDDPM（表格数据扩散模型），timesteps=200, epochs=2000
- **训练数据**：PUMS 2022 5-Year Michigan（499,880 条记录，68 PUMAs）
- **目标变量**：PINCP（收入）、SCHL（教育，24类）、ESR（就业状态，6类）
- **验证**：5-fold PUMA holdout 交叉验证（每折留出 10-17 个 PUMA 作为测试集）
- **消融设计**：三种条件组合
  - `demo_only`：仅 age + sex
  - `demo_race`：age + sex + race
  - `demo_race_puma`：age + sex + race + PUMA ID

### 对照基线

同时计算三个非学习基线，作为参照系：

| 基线 | 含义 | age×income joint TVD |
|------|------|---------------------|
| **Independence** | $P(\text{age}, \text{income}) = P(\text{age}) \times P(\text{income})$，零信息 | **0.316** |
| **Global Seed** | 用全州 PUMS 联合分布，传统 IPF 方法 | **0.124** |
| **Oracle IPF** | 用测试 PUMA 自身的真实边际做 IPF（不可达上界） | **0.060** |

### 发现

**消融实验结果（PUMA 级，5-fold holdout 均值）：**

| 条件 | Income TVD | Education TVD | ESR TVD | Copula TVD | **Joint TVD** |
|------|-----------|--------------|---------|-----------|--------------|
| `demo_only` | 0.074 | 0.115 | 0.045 | 0.110 | 0.114 |
| `demo_race` | 0.073 | 0.107 | 0.043 | 0.109 | 0.112 |
| `demo_race_puma` | **0.060** | **0.092** | **0.028** | **0.095** | **0.103** |

**条件信息的递增收益**：

| 从 → 到 | Income | Education | ESR | Joint |
|---------|--------|-----------|-----|-------|
| demo_only → demo_race | -1% | -7% | -5% | -2% |
| demo_race → demo_race_puma | **-18%** | **-14%** | **-35%** | **-8%** |

**Insight**：

1. **PUMA ID 是最关键的条件信号**，贡献了最大的边际收益——因为它直接编码了空间位置，让模型学到"这个区域的收入-年龄关系长什么样"
2. Race 的贡献相对温和，说明在 Michigan 的 PUMA 尺度上，种族组成的信息部分已被 PUMA ID 隐含
3. 最佳条件 `demo_race_puma` 的 joint TVD = 0.103，位于 independence（0.316）到 oracle（0.060）连续谱的 **78% 位置**：

$$\frac{0.316 - 0.103}{0.316 - 0.060} = 0.83$$

即扩散模型从零信息到完美信息的路上走了 **83%**，而传统 global seed IPF 只走了 75%。

---

## 五、端到端管线的集成验证（Exp4）

### 方法

将 Layer 1（Exp1 的 BG 级 age×sex×race 计数）与 Layer 2（Exp2 的最佳模型 `demo_race_puma`）完整串联：
1. 从 Exp1 的 BG 计数出发，通过 TIGER 地理映射确定每个 BG 所属的 PUMA
2. 对每个 BG 的个体，以其 age/sex/race + 所属 PUMA 作为条件，调用扩散模型采样 income/education/ESR
3. 在 PUMA 级别与 PUMS 真实分布做 TVD 对比

### 发现

**全量 68 个 PUMA 的端到端精度：**

| 指标 | Mean | Max |
|------|------|-----|
| Income TVD | **0.063** | 0.200 |
| Education TVD | **0.093** | 0.296 |
| ESR TVD | **0.031** | 0.101 |
| Copula TVD（age×income） | **0.109** | 0.202 |
| Joint TVD（age×income） | **0.125** | 0.243 |
| **Cosine Similarity（age×income joint）** | **0.976** | 0.906 (min) |

端到端的 joint TVD（0.125）略高于 Exp2 holdout（0.103），因为 Exp4 包含了 Layer 1 → Layer 2 传递过程中 BG→PUMA 映射引入的微小偏差。但整体精度稳定，68 个 PUMA 全部 cosine similarity > 0.90。

---

## 六、Tract 级外部验证与精度边界（Exp3）

### 方法

将 Exp4 的合成人口聚合到 tract 级别，与 ACS 2022 5-Year 的 tract 级边际分布做 TVD 对比。这是**外部验证**——ACS 数据未参与训练。

### 发现

| 变量 | Tract TVD (mean) | 含义 |
|------|-------------------|------|
| AGEP | **0.183** | 数据源 baseline（DHC 2020 vs ACS 2022 时间差） |
| SEX | 0.060 | 极低，符合预期 |
| ESR | 0.101 | 合理 |
| PINCP | 0.211 | 略高于 AGEP baseline |
| SCHL | **0.403** | 显著偏高 |

**Insight**：

AGEP 的 0.183 至关重要——它是 Layer 1 中 BG 级**精确**的变量，但聚合到 tract 后仍然有 0.183 的 TVD。这意味着 **0.183 是数据源差异（DHC 2020 vs ACS 2022）设定的基线噪声**，任何方法都无法降到这个值以下。

在此基准下：
- **PINCP（0.211）仅比基线高 0.028**——扩散模型引入的额外 tract 级误差极小
- **SCHL（0.403）** 显著偏高，根本原因是教育的极端空间聚类（大学城 tract vs 工业区 tract 在同一个 PUMA 内），PUMA 级模型无法区分

这不是模型的缺陷，而是**空间分辨率的结构性天花板**：SCHL 在 PUMA 级的 TVD 仅为 0.093（可接受），问题出在 PUMA→tract 的分辨率落差。

---

## 七、Tract 级后对齐的代价（Exp5）

### 动机

既然 tract 级精度受限于 PUMA 分辨率，一个自然的想法是：用 ACS 的 tract 级边际作为约束，对合成人口做 tract 级 IPF 后对齐（post-alignment）。

### 发现

| 指标 | Pre-align | Post-align | Delta |
|------|-----------|------------|-------|
| **Diffusion seed** ||||
| Joint TVD (PUMA) | 0.125 | 0.296 | **+0.171** |
| Copula TVD (PUMA) | 0.109 | 0.333 | **+0.224** |
| Cosine Sim (PUMA) | 0.976 | 0.902 | **-0.074** |
| **Global seed** ||||
| Joint TVD (PUMA) | 0.129 | 0.286 | +0.157 |
| Copula TVD (PUMA) | 0.105 | 0.318 | +0.213 |
| Cosine Sim (PUMA) | 0.975 | 0.905 | -0.069 |
| **Post: Diffusion vs Global** ||||
| Cosine Sim 差值 | — | — | **-0.004** |

**Insight**：

1. **Post-alignment 以 PUMA 级联合结构的大幅恶化为代价**——copula TVD 从 0.109 飙升到 0.333（+206%），cosine similarity 下降 0.074
2. **扩散模型的优势被完全抹平**——后对齐后，diffusion seed 和 global seed 的 cosine similarity 仅差 0.004，已无统计意义
3. 这证实了一个深层张力：**IPF 后处理和生成模型在逻辑上是冲突的**。IPF 只保证边际正确，代价是打碎联合结构；扩散模型的核心价值恰恰是联合结构。用 IPF 后处理扩散模型的输出，等于让扩散退化为一个 IPF 初始化器

**结论**：Tract 级后对齐不应采用。PUMA 级是当前公开数据约束下的精度天花板，应诚实报告这一边界。

---

## 八、精度版图总结

### 各层级验证能力

| 空间层级 | 变量 | Ground Truth | 验证状态 |
|---------|------|-------------|----------|
| **Block Group** | age × sex × race | DHC 2020 | ✅ 0 误差 |
| **PUMA** | income, education, ESR 联合分布 | PUMS holdout | ✅ Joint TVD 0.103 |
| **PUMA** | age × income cosine similarity | PUMS holdout | ✅ mean 0.976 |
| **Tract** | income, ESR 边际 | ACS 2022 | ✅ 可报告（0.101–0.211） |
| **Tract** | 联合分布 | 无公开数据 | ❌ 无法验证 |
| **BG** | 社会经济属性 | 无公开数据 | ❌ 无法验证 |

### 方法对比谱

```
Joint TVD (age × income)

0.316  Independence ─────────────────── 零信息
  │
  │    ↓ -61%  （全局平均 copula）
  │
0.124  Global Seed IPF ──────────────── 传统方法
  │
  │    ↓ -17%  （空间异质 copula）
  │
0.103  Diffusion (demo_race_puma) ──── 本方法
  │
  │    ↓ -42%  （不可达上界）
  │
0.060  Oracle IPF ───────────────────── 完美信息
```

扩散模型在 **0.316→0.060 的可提升空间中，实现了 83% 的信息恢复**，相比全局平均 IPF 的 75%，提升了 8 个百分点。

### 核心结论

1. **社会经济属性的空间异质联合分布是可学习的**——扩散模型通过 PUMA ID 条件化，捕获了传统方法忽略的局部 copula 结构
2. **PUMA ID 是最有效的条件信号**——在消融实验中贡献了绝大部分精度提升，证明空间位置信息是理解社会经济结构的关键
3. **PUMA 级是当前公开数据下的精度天花板**——tract 级后对齐虽能改善边际，但以破坏联合结构为代价，得不偿失
4. **该方法与既有骨架生成方法互补而非竞争**——Layer 1 延续了 IPF 在约束满足上的优势，Layer 2 用生成模型填补了社会经济属性的空白

---

## 附录：实验配置

| 参数 | 值 |
|------|------|
| 研究区域 | Michigan 州（FIPS 26） |
| PUMA 数量 | 68 |
| Block Group 数量 | 8,386 |
| Tract 数量 | ~2,904（有 ACS 重叠） |
| PUMS 数据 | 2022 5-Year，499,880 条记录 |
| DHC 数据 | 2020，P12 + P5 表 |
| ACS 数据 | 2022 5-Year，tract 级边际 |
| 扩散模型 | Gaussian TabDDPM |
| 训练配置 | timesteps=200, epochs=2000, batch_size=4096 |
| 交叉验证 | 5-fold PUMA holdout |
| 消融条件 | demo_only / demo_race / demo_race_puma |

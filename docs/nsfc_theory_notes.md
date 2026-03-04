# 国自然理论线索备忘：Langevin-FP 对偶性与 Population Synthesis

> 状态：草稿思路整理，供国自然申请书撰写参考
> 更新：2026-02-09

---

## 1. 核心物理类比

### 1.1 Langevin 方程 ↔ Person-level 生成

Langevin 方程描述单个粒子在外力和随机扰动下的运动：

$$dx = -\nabla U(x)\,dt + \sqrt{2D}\,dW$$

在 population synthesis 的语境中，这对应**逐个生成**合成个体：模型每次输出一条个人记录 $x = (\text{age, sex, income, education, employment})$，优化的是个体重建质量。

### 1.2 Fokker-Planck 方程 ↔ Distribution-level 生成

Fokker-Planck 方程描述粒子集合的概率密度演化：

$$\frac{\partial p}{\partial t} = \nabla \cdot \big[p\,\nabla U\big] + D\,\nabla^2 p$$

在我们的方法中，这对应**直接建模区域联合概率向量** $p \in \Delta^{K-1}$：每个训练样本是一个区域的完整分布，diffusion 模型在分布空间中操作。

### 1.3 经典物理中的一致性

在物理中，Langevin 和 FP 是**严格对偶的**：
- 给定正确的 Langevin 方程，无穷多粒子的经验分布精确收敛到 FP 的平稳解
- 二者通过 Itô 公式建立的精确数学关系连接
- 个体层面的动力学一定"涌现"出正确的总体统计量

---

## 2. 为什么一致性在 ML 中被打破

### 2.1 Score-based 生成模型中的理论一致性

Score-based 生成模型（Song et al. 2021）中：
- 前向过程是一个 SDE：$dx = f(x,t)dt + g(t)dW$
- 对应的 FP 方程描述 $p_t(x)$ 的演化
- 反向生成过程使用学到的 score $\nabla_x \log p_t(x)$
- **若 score 完美学到**，反向采样的个体分布精确等于 $p_0(x)$

**理论上，person-level 和 distribution-level 是一致的。**

### 2.2 实际中一致性崩溃的三个机制

| 条件 | 物理中 | ML 中 |
|------|--------|-------|
| 运动方程正确 | 由物理定律保证 | Score function 是神经网络近似，在低密度区域不准确 |
| 无穷多粒子 | 理论极限 | 有限生成样本 + 需要按区域聚合 |
| 遍历性 | 平衡态假设 | 有限步采样，模式坍缩 |

**关键机制：训练目标错位**

Score matching 优化的是：
$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

这是一个**逐点**目标——它惩罚的是每个样本的重建误差。但 copula TVD 是一个**总体**统计量：

$$\text{TVD}(p, q) = \frac{1}{2}\sum_{k=1}^K |p_k - q_k|$$

二者之间没有直接的梯度连接。即使每个生成的个体"看起来合理"（边际正确），它们聚合后的联合分布可以系统性偏差——因为属性间的高阶关联没有被训练信号约束。

**类比：用正确的能量函数做 Langevin 采样，但扩散系数（temperature）不对——个体轨迹合理，但平衡态分布偏了。**

### 2.3 有限样本的放大效应

要从 person-level 生成恢复一个 $K$ 维联合分布，需要：
- 生成 $N$ 个个体
- 按区域聚合成经验分布
- 经验分布的采样噪声 $\sim O(1/\sqrt{N \cdot p_k})$

对于 $K=512$ 的稀疏分布，很多 cell 的 $p_k$ 极小，采样噪声远大于信号——这是 person-level 在高维下尤其脆弱的原因。

---

## 3. 我们的方法用物理语言重新表达

> **核心：绕过 Langevin→FP 的涌现瓶颈，直接在 FP 层面建模。**

- 不去学习个体的"运动方程"（Langevin），而是直接学习概率密度的条件映射
- 训练目标与评估目标**对齐**：模型预测的就是 $p \in \Delta^{K-1}$，评估的也是 $\text{TVD}(p_{\text{pred}}, p_{\text{true}})$
- 用条件信息（marginals, pairwise）作为"边界条件"，约束 FP 解的可行域

---

## 4. 国自然可展开的理论方向

### 4.1 信息论视角：Fisher 信息与 Copula 恢复

Score matching 的目标函数与 Fisher 信息直接相关：
$$J(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\|\nabla_x \log p_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|^2\right]$$

但对 copula 的 Fisher 信息不在这个目标中被直接优化。可以量化：
- person-level score matching 关于 copula 参数的 Fisher 信息效率
- distribution-level 直接预测的 Fisher 信息效率
- 证明后者在有限样本下具有更高的 copula 恢复效率

### 4.2 Mean-field 连接

人口合成可以形式化为 mean-field 问题：
- 个体 $i$ 有属性 $x_i \in \{1, \ldots, K\}$
- 区域的"场"是经验分布 $\hat{p} = \frac{1}{N}\sum_i \delta_{x_i}$
- 我们关心的是场 $\hat{p}$ 的性质（copula），不是个体 $x_i$

在 Mean-field game (MFG) 理论中：
- 个体优化和总体分布通过一致性条件耦合
- 当耦合弱（训练信号不直接约束 copula）时，分别建模更高效
- 我们的工作 = 直接在 mean-field 层面建模

**可以写成**：当 population synthesis 的评估指标定义在 mean-field 层面（TVD of joint distribution）而非个体层面时，直接在 mean-field 层面训练是更高效的策略。

### 4.3 单纯形上的连续 FP 过程

当前用离散的 DDPM。理论上可以定义单纯形 $\Delta^{K-1}$ 上的连续 SDE/FP：
- 前向过程：在单纯形上做 Brownian motion（需要反射边界或对数坐标）
- 反向过程：学习单纯形上的 score
- 与 Dirichlet diffusion (Avdeyev 2023)、simplex diffusion (Floto 2023) 对接

这比我们目前的 log-probability + Z-score 规范化更数学地严谨。国自然中可以把"**单纯形上的条件 FP 过程**"作为一个理论贡献方向。

### 4.4 Langevin-FP 不一致性的"相变"

一个有理论深度的问题：**在什么条件下 person-level 训练能涌现正确的 copula（一致性恢复），在什么条件下不能（一致性崩溃）？**

关键参数：
- $K$：联合分布维度
- $N$：每区域样本量（或生成量）
- $d_c$：条件信息维度（marginal, pairwise）

猜想：存在一个相变边界 $K^*(N, d_c)$：
- $K < K^*$：person-level 可以涌现正确的 copula（一致性区）
- $K > K^*$：person-level 失败，必须在 distribution-level 建模（不一致区）

**我们的 K=512 scaling boundary 可能就是这个相变点的经验证据。**

这可以通过以下方式验证：
1. 固定 $N$，扫描 $K$，测量 person-level 和 distribution-level 的 TVD 差距
2. 固定 $K$，扫描 $N$（生成量），看 person-level 在什么 $N$ 下追上 distribution-level
3. 用随机矩阵理论或高维统计的工具推导 $K^*(N)$ 的 scaling law

---

## 5. 与现有理论的连接

| 我们的概念 | 物理/数学对应 | 关键文献 |
|-----------|-------------|---------|
| Person-level generation | Langevin dynamics | Langevin 1908; Song et al. 2021 |
| Distribution-level generation | Fokker-Planck equation | Fokker 1914; Planck 1917 |
| Copula 作为目标统计量 | Mean-field order parameter | McKean 1966; Lasry & Lions 2007 |
| K=512 scaling boundary | Phase transition | 高维统计中的 curse of dimensionality |
| Score matching ↔ Fisher info | Score function = ∇log p | Hyvärinen 2005 |
| 单纯形上的扩散 | Dirichlet process | Avdeyev et al. 2023; Floto et al. 2023 |

---

## 6. 一句话总结（适合国自然摘要）

> 人口合成中"从个体生成涌现总体依赖结构"的失败，本质上是 Langevin-Fokker-Planck 对偶性在有限样本、不完美 score 学习条件下的崩溃。本项目提出直接在 Fokker-Planck 层面（概率分布空间）建模，绕过涌现瓶颈，并从信息论和 mean-field 理论两个角度为这一策略提供理论基础。

# Methods

> 本文档为路径 A 论文的方法部分草稿，面向 CEUS / EPB 级别期刊。
> 核心命题：条件扩散模型从边际分布中学习空间异质性 copula，替代 IPF 中的全局同质 copula 假设。

---

## 1 问题定义

### 1.1 人口合成中的"边际→联合"问题

人口合成（population synthesis）的核心任务是为每个地理区域（geographic zone）生成一组微观个体（synthetic agents），使其在统计上与已知的宏观约束一致。宏观约束通常以**边际分布**（marginal distributions）的形式获得——例如某区域的年龄分布、收入分布——但区域内多个属性的**联合分布**（joint distribution）不可直接观测。

形式化地，设一个区域的人口由 $D$ 个离散属性刻画，第 $d$ 个属性有 $B_d$ 个分箱（bin），联合分布为概率单纯形上的向量：

$$\mathbf{p} \in \Delta^{K-1}, \quad K = \prod_{d=1}^{D} B_d$$

已知条件是 $D$ 个一维边际 $\{\mathbf{m}^{(d)}\}_{d=1}^{D}$，其中 $\mathbf{m}^{(d)} \in \Delta^{B_d - 1}$。目标是从边际恢复联合分布 $\mathbf{p}$，再从 $\mathbf{p}$ 中采样得到微观个体。

这是一个**欠定反问题**：$K - 1$ 个自由参数需要从 $\sum_d (B_d - 1)$ 个边际约束中恢复。例如，5 个变量各取 4 个分箱时，$K = 1024$，但边际仅提供 15 个约束。缺失的信息——属性之间的**关联结构**（copula）——必须从某处获得。该 1024 例子用于说明自由度规模，不对应本文主实验的固定配置。

### 1.2 IPF 的全局同质 copula 假设

迭代比例拟合（Iterative Proportional Fitting, IPF）是目前最广泛使用的方法。IPF 需要一个**种子联合分布**（seed joint）$\mathbf{s}$，然后通过交替缩放使边际对齐到目标值：

$$\mathbf{p}^{\text{IPF}} = \text{IPF}(\mathbf{s};\, \mathbf{m}^{(1)}, \dots, \mathbf{m}^{(D)})$$

IPF 的收敛性已有理论保证（I-projection），但其质量完全取决于种子 $\mathbf{s}$ 的选择。实践中，种子通常取自**全局微观样本**（如美国的 PUMS）或**全域平均联合分布**。这意味着 IPF 对所有区域施加了**同一个 copula 结构**——它假设底特律与安阿伯、城市核心与郊区的属性关联模式完全相同，区别仅体现在边际分布的差异上。

**这一假设在实证上不成立。** 大学城中"年轻 + 低收入 + 高学历"的三阶关联与制造业城市中"中年 + 中收入 + 低学历"的关联截然不同。当空间异质性 copula 真实存在时，IPF 使用全局种子会系统性地偏离当地的真实联合分布。

### 1.3 本文的核心思路

我们提出用**条件扩散模型**替代 IPF 中的全局种子，从数据中学习"边际 → copula"的映射：

$$\mathbf{p} = f_\theta(\mathbf{c}), \quad \mathbf{c} = [\mathbf{m}^{(1)};\, \dots;\, \mathbf{m}^{(D)};\, \text{optional pairwise marginals}]$$

其中 $f_\theta$ 是条件 DDPM（Denoising Diffusion Probabilistic Model）学到的生成器。与 IPF 的关键区别是：**模型根据不同的边际条件 $\mathbf{c}$ 生成不同的 copula**，而非对所有区域施加相同的种子。

### 1.4 在完整 pipeline 中的位置

本文的方法替代的是人口合成 pipeline 中的**第一步**——从边际恢复区域联合概率表。后续步骤（从联合分布中采样离散个体，以及在各分箱内生成连续属性值如具体收入金额）仍沿用标准做法，不依赖扩散模型。具体而言：

1. 扩散模型 → 区域联合概率表 $\hat{\mathbf{p}} \in \Delta^{K-1}$
2. 多项式采样 → $N$ 个离散属性组合（如 "25–44岁 / 男 / \$35k–\$75k / 本科 / 在业"）
3. 箱内连续化 → 在每个分箱内从微观样本的经验分布（或均匀分布）中采样连续值

我们的贡献在第 1 步：更准确的联合分布使第 2 步采样出的属性**组合**更符合当地人口的真实关联模式。

---

## 2 Distribution-Level Conditional Diffusion

### 2.1 建模对象：区域联合分布向量

与传统的 person-level 扩散（每个训练样本是一个人）不同，我们的建模对象是**区域级别的联合分布向量**。每个训练样本 $\mathbf{x}_0 \in \mathbb{R}^K$ 对应一个地理区域（PUMA）的联合概率分布经过对数变换后的表示：

$$\mathbf{x}_0 = \log(\mathbf{p} + \epsilon), \quad \epsilon = 10^{-6}$$

对数变换将概率单纯形映射到实数空间，使 DDPM 的高斯噪声假设成立。加 $\epsilon$ 避免零概率 cell 产生的 $-\infty$。

**为什么不直接在 person level 建模？** 我们的早期实验（见 Findings §1）证明，person-level 的 $\epsilon$-MSE 损失是**样本级别**的损失，无法学习**总体级别**的 copula 结构。即使模型完美预测每个个体的属性，也不保证采样总体的联合分布正确——因为 copula 是总体的涌现属性，不等于个体的求和。

### 2.2 Z-score 标准化

DDPM 的噪声调度（$\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$，线性）隐式假设 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。但对数概率向量的实际分布偏离这一假设——尤其当 $K$ 较大时，大量 cell 的概率趋近零，$\log(\epsilon) \approx -13.8$，导致数据均值远低于 0、方差远大于 1。

我们在每个训练折（fold）内对 $\mathbf{x}_0$ 执行逐维 z-score 标准化：

$$\mathbf{z} = \frac{\mathbf{x}_0 - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}, \quad \sigma_j = \max(\hat{\sigma}_j,\, 10^{-6})$$

采样时反标准化回对数概率空间，再通过 softmax 投影回概率单纯形。

这一步的必要性随 $K$ 急剧上升。$K=64$ 时每个 cell 平均约 110 人，零 cell 很少，对数值集中在 $[-6, -2]$（均值约 $-4$，标准差约 $1.5$），尺度偏移不致命——不做标准化时 TVD=0.076 vs 做了 TVD=0.069，差异仅 9%。但 $K=512$ 时大量 cell 的概率趋近零，$\log(10^{-6}) = -13.8$ 成为主导值，数据均值约 $-11$、标准差约 $4$。反向去噪需要将均值从 0 推移到 $-11$、方差从 1 放大到 16——完全超出噪声调度的有效范围。不做标准化时 TVD≈0.98（随机噪声），做了之后恢复到 0.15（有意义的生成），差距是**生死级别**的。

### 2.3 条件向量设计

模型以条件向量 $\mathbf{c}$ 引导生成。我们设计了四种条件配置，以系统性地考察不同信息量对 copula 恢复质量的影响：

| 条件 | 组成 | 维度（以 5-var $B=(4,2,4,4,4)$ 为例） |
|---|---|---|
| **none** | 无条件生成 | 0 |
| **marginal** | $D$ 个一维边际 $\mathbf{m}^{(d)}$ 的拼接 | $\sum_d B_d = 18$ |
| **pairwise** | $\binom{D}{2}$ 个二维边际的展平拼接 | $\sum_{i<j} B_i B_j = 128$ |
| **marginal + pairwise** | 一维 + 二维边际 | 146 |

一维边际携带各属性的频率信息；二维边际（pairwise）额外携带两两属性的**二阶关联**信息。当 $D$ 较大时，联合分布的高阶结构（三阶及以上交互）无法从 pairwise 中恢复，模型必须从训练数据中学习这些高阶模式。

### 2.4 模型架构

去噪网络 $\epsilon_\theta(\mathbf{z}_t, t, \mathbf{c})$ 采用全连接 MLP（multi-layer perceptron）：
- 时间步 $t$ 通过 128 维正弦位置编码嵌入
- 条件向量 $\mathbf{c}$、噪声输入 $\mathbf{z}_t$ 与时间嵌入在输入层拼接（concat injection）
- 隐藏层：$[1024, 1024]$，SiLU 激活
- 输出层：$K$ 维（与输入同维）
- 优化器：AdamW（$\text{lr} = 10^{-3}$，$\text{weight\_decay} = 10^{-4}$），梯度裁剪至 1.0

训练目标为标准 $\epsilon$-prediction MSE：

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{z}_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{c})\|^2\right]$$

其中 $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$，$T = 1000$。

### 2.5 采样与后处理

对每个测试区域，给定其条件向量 $\mathbf{c}$：
1. 从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样 $\mathbf{z}_T$
2. 执行 $T$ 步 DDPM 反向去噪，得到 $\hat{\mathbf{z}}_0$
3. 反标准化：$\hat{\mathbf{x}}_0 = \hat{\mathbf{z}}_0 \cdot \boldsymbol{\sigma}_{\text{train}} + \boldsymbol{\mu}_{\text{train}}$
4. Softmax 投影到概率单纯形：$\hat{\mathbf{p}}_{\text{raw}} = \text{softmax}(\hat{\mathbf{x}}_0)$
5. 重复 $N_{\text{sample}}$ 次（本文取 128），对 $\hat{\mathbf{p}}_{\text{raw}}$ 取均值以降低采样方差
6. （可选）对 $\hat{\mathbf{p}}_{\text{raw}}$ 执行 post-hoc IPF 对齐边际，得到 $\hat{\mathbf{p}}_{\text{post}}$

Post-hoc IPF 仅对含一维边际条件的配置启用（marginal、marginal+pairwise），将扩散输出作为 IPF 的种子，对齐到真实边际。这对扩散模型**不是优势**——对照组 IPF(global seed) 使用的也是同样的 IPF 后处理，差别仅在于种子不同。

---

## 3 实验设计

### 3.1 数据

使用美国社区调查（American Community Survey, ACS）2023 年 5 年估计的公共微观样本（PUMS），覆盖全美 50 州 2,456 个 PUMA（Public Use Microdata Area）。

本文实际使用两类统计单元：
- **2-var（household-level）**：来自 person+household 文件联接，变量为 `householder_age × household_income`，与 ACS B19037 口径对齐（4×16）。
- **3-var/5-var（person-level）**：来自 person 文件，变量为 `AGEP/SEX/PINCP/SCHL/ESR`。

3-var/5-var 实验的实际分箱如下：

| 属性 | 变量名 | bins=2（coarse） | bins=4（fine） |
|---|---|---|---|
| 年龄 | AGEP | 0–44 / 45+ | 0–24 / 25–44 / 45–64 / 65+ |
| 性别 | SEX | Male / Female | Male / Female |
| 个人收入 | PINCP | <$50k / >=$50k | <$25k / $25k–$50k / $50k–$100k / >=$100k |
| 受教育程度 | SCHL | <SomeCollege / SomeCollege+ | <HS / HS-GED / SomeCollege / BA+ |
| 就业状态 | ESR | LaborForce / NILF | Employed / Unemployed / Armed / NILF |

其中，3-var 的收入在细粒度配置下另有 16 档（`<10k` 到 `200k+`）。

由此构建不同变量数量和分辨率的联合分布：

| 配置 | 变量 | 分箱 | $K$ |
|---|---|---|---|
| 2-var | age × income | 4 × 16 | 64 |
| 3-var K=64 | age × income × education | 4 × 4 × 4 | 64 |
| 3-var K=256 | age × income × education | 4 × 16 × 4 | 256 |
| 5-var K=32 | age × sex × income × education × employment | 2 × 2 × 2 × 2 × 2 | 32 |
| 5-var K=128 | 同上 | 4 × 2 × 4 × 2 × 2 | 128 |
| 5-var K=512 | 同上 | 4 × 2 × 4 × 4 × 4 | 512 |

对联合计数施加 Laplace 平滑（$\alpha = 1$）后计算概率。

需要强调：2-var 与 3-var/5-var 的统计单元不同（household vs person）。本文将其共同纳入 scaling 分析，用于比较“条件信息与联合维度”对方法性能的影响；不将二者解读为完全同口径的绝对误差对比。

### 3.2 评估协议：Leave-Michigan-Out

采用空间留出验证（spatial hold-out）：以 Michigan 州的 68 个 PUMA 作为测试集，其余 2,388 个 PUMA 作为训练集。模型从未见过 Michigan 的联合分布，仅在测试时接收其边际作为条件。

对 2-var 配置额外执行 **Michigan 5-fold 交叉验证**：将 68 个 MI PUMA 分为 5 折，每折约 13–14 个 PUMA 作为测试，其余 54–55 个 MI PUMA 并入训练集。这产生 5 组独立的 Diffusion vs IPF 对比，用于统计显著性检验。

### 3.3 评估指标

**主指标：总变差距离（Total Variation Distance, TVD）**

$$\text{TVD}(\hat{\mathbf{p}}, \mathbf{p}) = \frac{1}{2} \sum_{k=1}^{K} |\hat{p}_k - p_k|$$

TVD ∈ [0, 1]，越低越好。对所有测试 PUMA 取平均。

**辅助指标：余弦相似度（Cosine Similarity）**，衡量分布形状的方向一致性。

### 3.4 对照基线

| 基线 | 描述 |
|---|---|
| **Independence** | 测试 PUMA 的各一维边际的外积 $\mathbf{p}^{\text{ind}} = \mathbf{m}^{(1)} \otimes \cdots \otimes \mathbf{m}^{(D)}$，假设所有属性完全独立 |
| **IPF(global seed)** | 以训练集加权平均联合分布为种子，用 IPF 对齐到测试 PUMA 的真实边际。这是实践中最常用的 IPF 配置。 |

Independence 是下界（完全忽略 copula），IPF(global seed) 是**强基线**（使用全局 copula + 精确边际对齐）。我们的方法需要超越后者，才能证明学到了超越全局平均的空间异质性 copula。

---

## 4 技术细节备注

### 4.1 N 维 IPF 实现

对 $D > 2$ 的联合分布，IPF 在每轮迭代中依次沿每个轴（axis）做边际对齐缩放。设当前估计为 $\hat{P} \in \mathbb{R}^{B_1 \times \cdots \times B_D}$：

$$\hat{P}^{(d)}_{i_1 \cdots i_D} \leftarrow \hat{P}_{i_1 \cdots i_D} \cdot \frac{m^{(d)}_{i_d}}{\sum_{j \neq d} \hat{P}_{i_1 \cdots i_D}}$$

迭代直至收敛（最大边际误差 < $10^{-10}$）或达到 200 轮。

### 4.2 Pairwise 边际的提取

对 $D$ 个变量，$\binom{D}{2}$ 个 pairwise 边际直接从联合分布的多维张量中沿相应轴求和得到，展平后拼接为条件向量的一部分。在测试时，pairwise 边际从测试 PUMA 的真实联合分布中计算——这在实践中可通过微观样本的交叉列联表获得，与一维边际的可得性相当。

### 4.3 训练超参数

主实验统一设置：batch_size = 512，timesteps $T$ = 1000，hidden_dims = [1024, 1024]，AdamW optimizer，seed = 0。训练时长按配置调整：2-var（MI 5-fold）使用 4,000 epochs；3-var 与 5-var（$K \leq 256$）使用 10,000 epochs；5-var（$K = 512$）使用 30,000 epochs。

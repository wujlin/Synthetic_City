# Equilibrium-Residual World Model

## 1. 这份 note 要回答什么问题

这份 note 不是再解释“为什么 diffusion 有时比 IPF 好”，而是要把 `Paper 1` 和 `Paper 2` 放进同一个问题框架：

- 给定一组观测约束，最大熵解提供一个结构中性的平衡态参考；
- 真实区域分布偏离这个参考的部分，是一个具有空间异质性的残差场；
- 学习模型真正需要恢复的，不是约束本身，而是**在约束之上仍然可预测的残差结构**。

这条线的核心问题因此不是“模型够不够复杂”，而是：

\[
\text{在给定约束 } c \text{ 的前提下，真实分布对平衡态的偏离 } \delta(c) \text{ 有多少是可预测的？}
\]

---

## 2. 正确的状态对象：不是 marginals，而是 constraints

若只写 marginals `m`，这个框架只能覆盖最干净的 `marginals-only` 情形。但仓库里的真实实验已经包含：

- 一维 marginals
- pairwise marginals
- hierarchical coarse-to-fine constraints

因此统一记号必须写成观测约束 `c`：

\[
\mathcal P(c) = \{p \in \Delta^{K-1}: p \text{ satisfies } c\}
\]

\[
p_{\mathrm{eq}}(c) = \arg\max_{p \in \mathcal P(c)} H(p)
\]

\[
p_{\mathrm{true}} = p_{\mathrm{eq}}(c) + \delta(c)
\]

这里：

- `p_true`：区域真实 joint distribution
- `p_eq(c)`：在当前约束集 `c` 下的最大熵参考解
- `δ(c)`：超出约束所能解释的残差结构

这一步很关键。因为：

- `marginals-only` 时，`p_eq(c)` 就是 marginals 的独立乘积，等价于 uniform-seed IPF；
- `pairwise` 时，`p_eq(c)` 已经不再是普通 IPF，而是 pairwise max-ent / log-linear reference；
- `hierarchical` 时，参考系还需要跟着 coarse/fine 约束一起改变。

所以这个 world model 的核心对象不是某一种具体算法，而是**在给定约束集下的平衡态参考**。

---

## 3. Paper 1 的物理图像

在 `Paper 1` 里，每个 PUMA 都对应单纯形上的一个点：

\[
p_r \in \Delta^{K-1}, \quad r = 1,\dots,2456
\]

观测约束是对 joint distribution 的粗粒化投影。可行域中的每一个点都满足这些宏观约束，但 copula 或更高阶 interaction 不同。

因此：

- 约束 `c` 决定了一个凸可行域；
- 最大熵解 `p_eq(c)` 选出其中最中性的那个点；
- 真实分布 `p_true` 与 `p_eq(c)` 的差，就是局部社会经济过程留下的结构残差。

对 `marginals-only` 的最干净情况，这个图像最简单：

- `p_eq` 由一维 marginals 的独立乘积给出；
- `δ` 就是“在已知 marginals 之后，区域仍然保留的 joint dependence”。

这时 diffusion 的角色可以写成：

\[
\hat p_{\mathrm{diff}}(c) \approx p_{\mathrm{eq}}(c) + \hat\delta(c)
\]

也就是说，diffusion 若优于 MaxEnt，不是因为它更会满足 marginals，而是因为它恢复了**可由约束 profile 推断的残差场**。

---

## 4. Paper 2 的映射

同一个框架也适用于 destination problem，只是状态从 demographic joint 变成 destination flow。

\[
P^{OD}_{\mathrm{true}} = P^{OD}_{\mathrm{eq}}(c_{\mathrm{struct}}) + \delta_{\mathrm{mobility}}
\]

这里：

- `c_struct` 可以是 home/work mass、distance impedance、LODES skeleton；
- `P_eq^{OD}` 是结构性 transport baseline；
- `δ_mobility` 是行为层的偏离场；
- mobility 不是直接提供 dense tract-OD supervision，而是在某些 coarse state 上给出残差信号。

这正是为什么 `Paper 2` 当前走向 `coarse-to-fine destination model` 是合理的：  
如果细尺度残差不可识别，就应该先在更可观测的 coarse state 上恢复它。

---

## 5. 这套 world model 的可检验预测

### 5.1 核心预测

diffusion 相对 MaxEnt 的优势，应该与残差的可预测性一致。

更具体地说，如果定义：

\[
\Delta = \mathrm{TVD}_{\mathrm{MaxEnt}} - \mathrm{TVD}_{\mathrm{diffusion}}
\]

那么跨 `K` 或跨约束配置的 `\Delta`，应当随着 `R^2(\delta \mid c)` 上升而上升。

这意味着：

- 若 `δ` 存在但不可预测，diffusion 不该稳定优于 MaxEnt；
- 若 `δ` 有显著可预测部分，diffusion 才有学习基础。

### 5.2 层级预测

如果 coarse residual 比 fine residual 更可预测，那么层级模型的优势就不是工程偶然，而是来自：

- Stage 1 先恢复高可预测残差；
- Stage 2 再恢复条件残余。

### 5.3 Breakdown 预测

当 `K` 继续增大而约束维度增长更慢时，残差场的可预测比例应该下降。  
这会导致 diffusion 对 MaxEnt 的优势收缩，甚至消失。

这里不需要假装存在一个 sharp phase transition。更稳的表述是：

- 存在一个 **data-compatibility ceiling**
- 一旦 `δ` 的大部分方差对给定约束变得不可预测，复杂生成模型的优势就不再可识别

---

## 6. 当前实验策略：先做最干净的 P0

第一步不直接碰 pairwise，也不碰 hierarchical。  
先固定一个最干净、定义不歧义的参考系：

- 数据对象：PUMA-level 5-variable joint distribution
- 约束：一维 marginals only
- 平衡态：marginals 的独立乘积，即 pure MaxEnt
- 残差：`p_true - p_eq` 的概率残差，以及 `log(p_true) - log(p_eq)` 的结构残差

这样做的原因很直接：

1. 参考系最清楚，不会混入 seed prior；
2. 可以直接回答“非平衡偏差是否实质存在”；
3. 可以先验证 `δ` 的可预测性，再决定是否值得推广到 richer constraints。

---

## 7. P0 的最小可交付

P0 只回答两个问题：

### Q1. `δ` 是否显著存在？

对每个区域计算：

- `TVD(p_true, p_eq)`
- `KL(p_true || p_eq)`

如果这些量接近零，说明问题本身接近平衡态，没必要继续讲 residual learning。  
如果这些量显著大于零且跨区域存在变异，说明 world model 的第一步成立。

### Q2. `δ` 是否能从约束 profile 预测？

做法：

- 用 marginals 作为特征 `c`
- 用 `log(p_true) - log(p_eq)` 作为残差表示
- 在训练区域上做 PCA 提取主残差模式
- 用 ridge regression 在 held-out 区域上预测 PC score

输出至少包括：

- 每个 PC 的解释方差比例
- held-out `R^2(PC_i | c)`
- 加权的可预测残差方差占比

如果 held-out `R^2` 普遍为零或负，则 diffusion 的优势不应被解释为“学到了可迁移的 residual law”。  
如果存在一批稳定可预测的主模式，这个 world model 就有实证支撑。

---

## 8. 当前不做什么

当前不做以下扩展：

- 不把全国平均 seed IPF 混入 equilibrium baseline
- 不把 pairwise / hierarchical 与 marginals-only 一起混算
- 不先做 diffusion vs MaxEnt 的跨 `K` 最终关联图
- 不先把 `Paper 2` 的 mobility residual 塞进同一个脚本

这些都属于下一阶段。  
P0 的任务只是把参考系和最小证据立住。

---

## 9. 一句话结论

这套 world model 的核心不是“diffusion 学 joint”，而是：

> 在给定观测约束之后，真实区域对平衡态参考的偏离只要仍然具有可预测部分，生成模型就有机会稳定超越最大熵；若这种偏离在当前尺度上不可识别，方法复杂度再增加也不会带来可靠增益。

这句话同时给 `Paper 1` 和 `Paper 2` 提供了统一的理论骨架。

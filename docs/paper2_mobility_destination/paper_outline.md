# 第二篇文章大纲：Coarse-to-Fine Work Destination

## 0. 核心主张

现有 synthetic population 方法已经能生成 **demographically consistent** 的人口，但并不会自动生成 **behaviorally plausible** 的 home-work interaction structure。  
`work` 的关键缺口不是 tract 内点位怎么放，而是上游缺少：

\[
P(g_w \mid g_h, x)
\]

因此，这篇文章的目标不是继续修 demographic marginals，而是单独建模 **destination layer**。  
在这个 layer 中：

- `ACS / Census` 负责人口与区域结构锚点
- `LODES` 负责 work mass 与 OD skeleton
- `mobility` 负责 coarse destination scale 上的行为约束

---

## 1. Introduction

### 1.1 现有 synthetic population 的主要成功

- 可以恢复小区域人口总量与 demographic structure
- 可以生成 household-aware 的 home 语义
- 也可以把人口落到显式空间位置

### 1.2 当前仍未解决的问题

- `work` 常被简化成 home tract 内的白天 support point
- 这并不等价于真实工作地
- 因而 demographic consistency 并不自动推出 commuting realism

### 1.3 本文的问题定义

本文关心的是：

- 在已知 home location 与 worker attributes 的条件下
- 如何生成可信的 destination state 与 destination tract
- 并进一步恢复合理的 commuting pattern

### 1.4 本文的核心发现应当压成三句

- 现有方法的主要瓶颈不是人口属性，而是 interaction structure
- gravity / LODES 只能提供 destination skeleton，不能充分解释 subgroup-conditioned residuals
- mobility 在 tract-OD 尺度过稀，不能直接取代 skeleton，但在 coarse destination scale 上可以提供稳定的行为约束

---

## 2. Problem Diagnosis

这一节只做一件事：证明为什么 destination model 是必要的。

### 2.1 当前 pipeline 的 work 对象定义有问题

- 原有 `work` 更接近 `within-home-tract daytime support`
- 不是显式的 destination object

### 2.2 失败证据

- work tract 指标弱
- OD 指标弱
- same-tract share 畸高
- commute distance 不合理

### 2.3 结论

当前 work 的主要误差来自 **destination generation 缺失**，而不是 point placement 本身失败。

---

## 3. Data Roles

这一节必须把不同数据源的角色分开，避免“多源融合”说得含糊。

### 3.1 PUMS / ACS / Census

- 负责人口属性与 small-area structure
- 不直接给 work destination

### 3.2 LODES

- 提供 destination mass 与 OD skeleton
- 负责低阶 commuting structure

### 3.3 Mobility

- 提供 day-anchor / night-anchor / stay pattern
- 提供 origin 与 coarse destination state 的活动画像
- 提供 gravity 无法解释的行为残差
- 不应在 tract-OD 尺度上被当作 dense supervision

### 3.4 POI

- 可作为 destination-side refinement
- 不是第一轮主轴

---

## 4. Method

### 4.1 Coarse-to-fine destination model

推荐主模型写成：

\[
P(g_w \mid g_h, x)
=
P(z_w \mid g_h, x)
\times
P(g_w \mid g_h, x, z_w)
\]

其中：

- `z_w`：coarse destination state，例如 `work county / job center / regime`
- 上层 `P(z_w \mid g_h, x)` 用结构骨架加 mobility 行为约束建模
- 下层 `P(g_w \mid g_h, x, z_w)` 用 LODES / gravity / accessibility / local context 细化

### 4.2 结构分解

方法上应强调：

- `gravity / LODES` 是 backbone，不是 baseline-only
- `mobility` 只在 coverage 足够稳定的 coarse layer 进入模型
- `destination model` 和 `explicit placement` 是两个层次
- tract-OD residual 只能作为局部 refinement，而不适合作为主约束

### 4.3 上层状态选择

这一节需要用 coverage diagnostics 支撑 state 选择：

- tract-OD：过稀，不适合作为主监督
- county：覆盖稳定，但空间分辨率偏粗
- center：预计是最优折中
- top-K tract candidates：适合作为局部 reranking 区域

### 4.4 下游显式点位

- destination tract 先生成
- tract 内 work point 再生成
- 不把两步揉成一个黑盒

---

## 5. Experiments

### 5.1 实验主问题

mobility 是否在 `gravity + LODES` 已存在时，仍能在 **正确的 destination scale** 上带来可验证的 structure 增益？

### 5.2 Baseline ladder

按信息增量展开：

1. current / same-tract baseline
2. distance-only
3. gravity-only
4. gravity + LODES
5. gravity + LODES + worker/home context
6. gravity + LODES + worker/home context + tract-OD residual
7. gravity + LODES + worker/home context + coarse-state mobility
8. optional: `+ POI`

### 5.3 指标

主指标围绕 interaction structure，而不是只看点图：

- tract share
- OD share
- same-tract vs cross-tract share
- commute distance distribution
- destination hotspot overlap
- subgroup-conditioned commute pattern

同时单独报告：

- county / center 层面的 share alignment
- top-K candidate coverage

---

## 6. Results

结果部分建议按三层问题顺序展开：

### 6.1 骨架层

- gravity / LODES 能解释多少 destination structure

### 6.2 状态层

- mobility 在 `center / county / top-K` 层面是否解释了骨架模型剩余误差

### 6.3 下游层

- destination 改善后，显式 work placement 是否同步改善

---

## 7. Discussion

### 7.1 本文不声称什么

- 不声称 mobility 是个体 SES 真值
- 不声称 mobility 可以覆盖 census 约束
- 不声称 tract-OD 稀疏观测可以直接替代 LODES skeleton

### 7.2 本文真正回答了什么

- demographic consistency 与 interaction realism 是两个层次
- destination model 是 synthetic population 的独立识别层
- 当行为数据在细尺度稀疏、在粗尺度稳定时，destination model 应采用 coarse-to-fine 分解

### 7.3 后续扩展

- industry-aware destination model
- POI-aware destination refinement
- home micro placement 与 activity chain consistency

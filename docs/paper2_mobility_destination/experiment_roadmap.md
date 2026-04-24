# 实验路线：Coarse-to-Fine Destination Model

## 1. 实验总目标

本轮实验不再问“mobility 能不能替代 ACS”，而是问：

**在 demographic structure 已由 Census / ACS 锚定、destination mass 已由 LODES 提供骨架的前提下，mobility 应该在什么空间尺度进入 destination model，才能提供可验证的行为增量？**

---

## 2. 实验逻辑顺序

### Stage A. 缺口诊断

**问题**

当前 pipeline 的失败到底来自哪里？

**要回答的点**

- 当前 `work` 是不是 destination object
- same-tract bias 有多强
- tract / OD / commute 哪一层先坏掉

**产出**

- 一页式诊断总结
- 作为新文章问题提出部分的直接证据

---

### Stage B. Backbone baseline

**问题**

只用透明可解释结构，destination 能做到什么程度？

**baseline**

1. `B0`: current / same-tract
2. `B1`: distance-only
3. `B2`: gravity-only
4. `B3`: gravity + LODES

**目的**

- 确认 classic destination skeleton 的上限
- 识别 mobility 真正有无新增信息

---

### Stage C. Context-aware baseline

**问题**

只加 worker/home context，不加 mobility，能否已经解释剩余误差？

**baseline**

5. `B4`: gravity + LODES + worker/home context

**目的**

- 把“属性效应”和“mobility 效应”分开

---

### Stage D. Proof-of-signal

**问题**

mobility 是否真的含有 OD 增量信息？

**模型**

6. `M1`: gravity + LODES + worker/home context + tract-OD residual

**目的**

- 先证明 mobility 不是纯重复 destination mass
- 不把这一步误写成最终主模型

---

### Stage E. Coverage diagnostics and state selection

**问题**

mobility 在哪一层最可信？

**必须诊断**

- tract-OD pair 覆盖率
- home tract -> work county 覆盖率与 share alignment
- home tract -> work center 覆盖率与 share alignment
- LODES top-K tract candidates 中的 mobility 命中率

**目的**

- 不再凭直觉选 state
- 用数据决定主模型该落在 `county / center / top-K reranking` 的哪一层

---

### Stage F. Coarse-to-fine main model

**问题**

当 tract-OD 过稀时，如何把 mobility 重新放回一个结构稳定的位置？

**模型**

7. `M2`: gravity + LODES + worker/home context + county-state mobility
8. `M3`: raw center-state diagnostic branch
9. `M4`: center hierarchy + top-K tract reranking within center

**推荐结构**

\[
P(g_w \mid g_h, x)
=
P(z_w \mid g_h, x)
\times
P(g_w \mid g_h, x, z_w)
\]

其中 `z_w` 优先测试：

- `job center`
- `work county`
- `destination regime`

**目的**

- 让 mobility 在可观测尺度上发声
- 同时保住 tract-level completeness 与 commute realism

当前根据 Detroit 首轮结果：

- `M2 county-state` 是稳健主结果候选
- `M3 raw center-state` 只保留为诊断，不作为最终主模型
- `M4 center + top-K reranking` 是下一步最值得推进的方法增强版本

---

### Stage G. Optional refinement

**问题**

当 destination tract 已经基本成立时，POI 或更细 destination-side features 是否还能提升 tract 内 work realism？

**模型**

10. `M5`: `M2/M4 + POI`

**定位**

- 可选增强项
- 不作为第一轮主结果依赖项

---

## 3. 指标体系

### 3.1 主指标

必须围绕 interaction structure：

- tract-level work share Spearman / cosine / TVD
- OD share Spearman / cosine / TVD
- commute distance distribution cosine / TVD
- same-tract share
- same-county share

### 3.2 状态层指标

- county share alignment
- center share alignment
- top-K candidate hit rate
- top-K candidate LODES mass coverage

### 3.3 下游指标

- explicit work-point support hit rate
- tract 内 placement hotspot alignment

---

## 4. 首轮实验设计

### 4.1 数据范围

首轮固定在 Detroit study area。

原因：

- 已有现成 synthetic persons
- 已有 LODES tract OD 预处理脚本
- 已有 mobility anchor validation 脚本
- 已有一轮 destination pilot，可作为起点

### 4.2 首轮不做的事

- 不直接做 statewide 泛化
- 不先做 POI-heavy refinement
- 不把 mobility 接回 tract demographic allocation
- 不从个体轨迹直接预测 SES

### 4.3 首轮最小交付

至少形成以下逻辑链：

- `B0-B4`：结构骨架与 context 上限
- `M1`：证明 mobility 有真实信号
- coverage diagnostics：判定 tract / county / center / top-K 哪一层最可信
- `M2`：把主模型收敛到 coarse-to-fine 结构
- `M4`：在不破坏 commute realism 的前提下尝试更强的 center-level signal
- layered evaluation：把 destination 误差拆成 county 选择误差与条件 tract 细化误差
- ceiling analysis：刻画 mobility 在 tract / center / county 三层的 compatibility 与 support boundary

---

## 5. 风险与判别标准

### 5.1 最大风险

mobility 只是在重复 destination mass，而没有提供新的结构信息。

**判别方式**

如果 `B4 -> M1` 的 OD 提升很弱，说明 mobility 贡献有限，需要收缩主张。

### 5.2 第二风险

mobility 在 tract-OD 尺度过稀，无法支撑主模型。

**解释**

这不等于 mobility 无用，而是说明它必须被上移到更稳定的 coarse state。

### 5.3 第三风险

mobility 提升了 OD，但破坏了 commute realism。

**应对**

- 不把它作为全局 dense prior
- 改成 coarse-state constraint 或 top-K reranking

---

## 6. 当前结论与立场

当前最稳的论文主线不是：

- “一个 tract-OD residual trick”

而是：

- “把 synthetic population 从 demographic consistency 推进到 interaction realism，并通过 coarse-to-fine destination model 在结构完整性与行为真实性之间建立分层耦合”

因此实验路线必须围绕这个主张服务，而不是变成一次 feature stacking。

### 6.1 现阶段的直接执行优先级

- `P0`：固定 `M2` 为稳健主结果，突出 county-level destination correction
- `P0`：保留 `M4` 作为 finer refinement 证据，但停止无约束参数扫
- `P0`：所有核心表述切换到分层评估，不再单独依赖全矩阵 tract-OD 指标
- `P1`：把 ceiling / compatibility 结果写成信息边界，而不是方法失败
- `P1`：若后续继续推 `M4`，必须以更明确的层级目标为前提，而不是继续在噪声内找最优 `K` 与 `weight`

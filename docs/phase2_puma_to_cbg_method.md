# 二期方法方案稿：从 PUMA 级 Full-Joint 到 CBG 级空间异质分解

> 状态：方案稿，待审核  
> 角色：二期方法文档，不替代一期 `full-earn` 主线  
> 对应关系：一期回答 `P(attrs | PUMA)`；二期回答 `P(CBG | attrs, PUMA)` 的空间分解问题

---

## 0. 核心结论

二期不应被定义成“给每个合成人预测 home CBG 的回归或分类问题”，而应被定义成一个**数据驱动的、受约束的聚合分解问题**：

- 一期已经给出 `PUMA` 内的 5 维 full-joint 分布
- 二期要做的是把这个 `PUMA` 级 joint mass 合理地下推到 `CBG`
- `mobility` 在这里承担的是**空间异质性先验**，不是个体级真值标签
- `building capacity / affordability` 不作为二期主锚点，而留到下一步 `CBG -> building`

因此，二期的核心对象不是某个 individual 的 `home_CBG`，而是：

\[
M_{p,k,g}
\]

它表示：

- 在 `PUMA p` 内
- 属于类型 `k` 的人口
- 有多少人被分配到 `CBG g`

---

## 1. 二期回答的问题

一期 full-earn 的方法主线是：

\[
P(\text{AGE}, \text{SEX}, \text{SCHL}, \text{ESR}, \text{EARN} \mid \text{PUMA})
\]

它恢复的是 `PUMA` 尺度的人口属性联合结构，但还没有回答：

> 同一个 `PUMA` 内，这些不同类型的人应该如何在更细的 `CBG` 空间中分布？

二期的任务就是回答：

\[
P(\text{CBG} \mid \text{attrs}, \text{PUMA})
\]

但这个条件分布不是通过 person-level 标签监督直接训练出来的，因为我们没有真实的 `(person, home_CBG)` 训练集。  
二期真正能做的是：

1. 利用一期生成的 `PUMA` 级 type counts  
2. 利用 `ACS` 的 `CBG` 级边际目标  
3. 利用 `mobility` 提供的 `home_CBG -> behavior profile` 弱监督  
4. 在这些条件下求解一个可解释的 `type -> CBG` 分解矩阵

---

## 2. 为什么这不是普通回归

把二期写成普通回归或分类会在问题定义上犯三个错误：

### 2.1 我们没有 person-level 的 home CBG 真值

目前可直接使用的 mobility 关键信号来自 `merged_poi_201902.csv` 中的：

- `visitor_home_cbgs`
- `visitor_work_cbgs`

这类字段是 `POI × month` 级别的聚合来源分布，不是 individual 级 home 标签。  
因此它可以告诉我们：

- 哪些 `CBG` 是更强的 residential origin
- 这些 `CBG` 的行为画像是什么

但它不能直接告诉我们：

- 这一个 synthetic person 应该住进哪一个 `CBG`

### 2.2 这是 ecological inference，不是 fully supervised prediction

从 `PUMA` 微观样本与 `CBG` 聚合边际恢复 `sub-PUMA` 联合结构，本质上属于 ecological inference / disaggregation。  
这类问题一般不是点识别的，而是需要通过额外先验来收紧可行解空间。

对本项目而言，额外先验来自：

- `ACS` 小区域边际
- `mobility` 所提供的行为异质性
- 最大熵 / 熵正则耦合

### 2.3 mobility 更适合作为空间异质性注入，而不是 SES 真值标签

现有文献与内部笔记都提醒过两点：

- `mobility -> sociodemographics` 的关系不是稳定强监督，存在明显偏差与泛化风险
- 更稳的 framing 是用移动数据把**空间异质性**注入合成过程，再用 census/ACS 边际拉回人口属性分布

因此，二期最合适的 framing 不是：

- “从 mobility 精准预测收入、教育、年龄”

而是：

- “用 mobility 识别不同 `CBG` 的行为画像差异，并将这种差异注入 `PUMA -> CBG` 分解”

---

## 3. 核心对象与记号

固定一个 `PUMA = p`。

### 3.1 类型 `k`

定义：

\[
k \in \mathcal{K}
\]

表示一期 full-earn 输出空间中的一个 joint cell：

\[
k = (\text{AGE bin}, \text{SEX}, \text{SCHL}, \text{ESR}, \text{EARN bin})
\]

例如：

- `25-34 / male / college / employed / earn_bin_4`
- `65-74 / female / HS / NILF / earn_bin_1`

记：

\[
n_{p,k}
\]

为 `PUMA p` 内类型 `k` 的人数。

### 3.2 CBG `g`

定义：

\[
g \in \mathcal{G}_p
\]

为 `PUMA p` 内的候选 `CBG` 集合。

### 3.3 分配矩阵 `M`

定义：

\[
M_{p,k,g} \in \mathbb{Z}_{\ge 0}
\]

含义是：

- `PUMA p` 内
- 类型 `k`
- 分到 `CBG g` 的人数

二期的目标就是求解 `M_{p,k,g}`。

### 3.4 CBG 级目标 `T`

记：

\[
T_{p,g,c}
\]

为 `CBG g` 上、约束类型 `c` 的外部目标统计。  
`c` 的候选包括：

- total population
- age
- sex
- age × sex
- education
- employment

不是每个 `c` 都必须可得；二期只使用稳定可拿到的目标。

### 3.5 CBG 行为画像 `b_g`

定义：

\[
b_{p,g}
\]

为 `CBG g` 的 mobility profile / behavior embedding。  
它来自 `merged_poi_201902.csv` 的 `visitor_home_cbgs` 聚合信号。

---

## 4. 输入数据与二期边界

### 4.1 一期输入

二期直接承接一期的产物：

- `PUMA` 级 5 维 full-joint distribution
- 或其离散展开后的 type counts `n_{p,k}`

### 4.2 CBG 级外部输入

二期需要的 `CBG` 级输入包括：

- `CBG` 总人口
- 稳定可得的 `ACS` 小区域边际
- `PUMA -> CBG` 的隶属关系

### 4.3 mobility 输入

二期使用 `merged_poi_201902.csv`，而不是把 `place20190206.csv` 当主桥。

原因是：

- `merged_poi_201902.csv` 直接提供 `visitor_home_cbgs`
- `place20190206.csv` 没有 `safegraph_place_id`
- `cluster_id` 当前不能可靠当作跨用户、跨表共享的 place bridge

因此二期的 mobility 主信号是：

- `POI <- home_CBG` 的来源分布
- `POI` 的类别、dwell、distance、hour/day popularity

### 4.4 明确不在二期主目标中的信号

以下信号不作为二期主锚点：

- building-level affordability
- building-level capacity
- building-level nearest-neighbor matching

这些更适合下一步 `CBG -> building`，不应提前混入 `PUMA -> CBG`。

---

## 5. cell 空间如何处理

一期理论上可能有约 3000 个 cell，但这并不意味着二期应该直接把 3000 个 cell 全部当成独立、同权、同可信度的对象来学习。

二期建议把 cell 分成三类：

### 5.1 结构性不可能 cell

这类 cell 由规则决定，不由样本频数决定。  
例如明显违反可行性逻辑的组合。

处理方式：

- 直接设为 hard zero
- 进入可行域 mask

### 5.2 逻辑上可能、但支持度极低的 cell

这类 cell 不能因为稀疏就删掉。  
删除会把“没观测到”误写成“不可能”。

处理方式：

- 输出空间保留
- 学习时做层级收缩

### 5.3 支持度充足的 cell

这类 cell 正常建模。

### 5.4 默认策略：保留 full space，学习时层级收缩

二期默认不粗暴合并掉 full-earn 的 5D 输出空间，而是在 `q(g|k)` 的学习中做 shrinkage：

\[
q(g \mid k) \approx
\lambda_0 q(g \mid k_{\text{full}})
 + \lambda_1 q(g \mid k_{\text{drop earn}})
 + \lambda_2 q(g \mid k_{\text{age,sex}})
 + \lambda_3 q(g \mid puma\ baseline)
\]

这意味着：

- `full 5D cell` 仍然是最终分解对象
- 但对稀疏 cell，不强迫它从零学一个独立的 `CBG` 偏好
- 它可以向更粗粒度的父层级借力

这比“每个 cell 独立 scorer”更稳，也比“先粗暴合并 cell”更忠实于一期主线。

---

## 6. 软约束与硬约束

二期最容易混淆的地方，是把所有“重要目标”都当成 hard constraint。  
正确做法不是按重要性分，而是按**可信度与可行性角色**分。

### 6.1 硬约束的定义

硬约束定义的是 feasible set。违反硬约束，结果在定义上就是无效的。

二期默认硬约束：

1. 类型人数守恒

\[
\sum_{g \in \mathcal{G}_p} M_{p,k,g} = n_{p,k}, \quad \forall k
\]

2. 非负性

\[
M_{p,k,g} \ge 0
\]

3. 结构性不可能 cell 的 hard zero mask

4. `CBG total population`

这是二期默认唯一建议硬化的小区域聚合量，但前提是：

- 它和一期总人数属于同一统计 universe
- 已经在 `PUMA` 内做了总量协调

### 6.2 软约束的定义

软约束定义的是 preferred solution。违反它不表示结果无效，而表示结果不够好。

二期默认软约束：

- `age × sex` 等小区域边际
- `education`
- `employment`
- mobility-derived prior
- 任何 proxy 型变量

### 6.3 为什么 mobility 必须是软约束

因为 mobility 本身受以下因素影响：

- 设备渗透率偏差
- 采样偏差
- POI 覆盖偏差
- 月度聚合噪声

因此它适合作为“哪类解更合理”的偏好项，不适合作为“唯一真实答案”。

---

## 7. mobility profile 如何构造

二期真正 data-driven 的核心，不是直接拿 `visitor_home_cbgs` 当 resident count，而是从它构造 `CBG` 的行为画像 `b_{p,g}`。

### 7.1 原始聚合逻辑

对每个 `POI j`，`merged_poi_201902.csv` 提供：

- `top_category`
- `sub_category`
- `median_dwell`
- `distance_from_home`
- `popularity_by_hour`
- `popularity_by_day`
- `visitor_home_cbgs`

如果记 `w_{j,g}` 为 `POI j` 的访客中来自 `CBG g` 的权重，则：

\[
b_{p,g} = \sum_j w_{j,g} x_j
\]

其中 `x_j` 是 POI 特征向量。

### 7.2 第一版不建议直接喂原始 one-hot

第一版不建议把所有 `top_category / sub_category` 原始 one-hot 直接丢给黑箱 scorer。  
那样虽然 data-driven，但解释性太弱，且更容易过拟合。

### 7.3 第一版建议的语义轴

建议先把 `POI` 聚合成若干语义轴，再构造 `h(g)`：

- childcare / school intensity
- healthcare / eldercare intensity
- office / weekday daytime intensity
- retail / service intensity
- leisure / nightlife intensity
- median dwell regime
- mean distance-from-home regime
- weekday-weekend contrast

这样做的好处是：

- 保留行为差异
- 便于写作解释
- 便于和人口类型发生可解释耦合

必要时，可在这些语义轴基础上再做低维投影，但不建议第一版直接用完全不可解释的 latent factor。

---

## 8. `q(g|k)` 的设计

二期最关键的 data-driven 对象不是 `M` 本身，而是：

\[
q_{p,k,g} = P(g \mid k, p)
\]

它表示：

- 在 `PUMA p` 内
- 类型 `k`
- 更倾向于落到哪个 `CBG g`

### 8.1 不建议的第一版

第一版不建议：

- 大型 black-box MLP classifier
- 直接做 person-level 回归到 `CBG`
- 用 raw `POI` one-hot 暴力拟合

这些做法不利于写作，也不利于解释失败模式。

### 8.2 建议的第一版：低秩 compatibility model

定义：

\[
\text{score}_{p,k,g}
=
\alpha_{p,g}
 + \phi(k)^\top W h(p,g)
 + u(\text{parent}(k))^\top v_{p,g}
\]

然后：

\[
q_{p,k,g}
=
\frac{\exp(\text{score}_{p,k,g})}
{\sum_{g' \in \mathcal{G}_p}\exp(\text{score}_{p,k,g'})}
\]

其中：

- `\alpha_{p,g}`：`CBG` baseline mass / base attractiveness
- `\phi(k)`：cell `k` 的属性表示
- `h(p,g)`：`CBG` 的 mobility + ACS profile
- `W`：学习 “什么类型的人更偏向什么样的 CBG 画像”
- `u(parent(k))`：用于稀疏 cell 层级收缩
- `v_{p,g}`：`CBG` 的低维表征

### 8.3 为什么这个形式更有洞见

这个形式有三个优点：

1. 它是 data-driven 的  
   `W` 是从数据里学出来的 compatibility，而不是手工规则。

2. 它是受控的  
   不会像大黑箱一样失去解释性。

3. 它允许层级收缩  
   稀疏 cell 不需要独立估计全部参数。

换句话说，二期要学的不是“这个人属于哪个 `CBG`”，而是：

> 哪一类人更容易落到哪一种行为画像的 `CBG`

这就是二期最值得写进论文的方法洞见。

---

## 9. 优化目标

在固定 `PUMA p` 后，二期求解：

\[
\min_{M \ge 0}
\quad
\lambda_{\text{acs}}
\sum_{g,c} w_c \, \ell(\hat T_{p,g,c}, T_{p,g,c})
\;-\;
\lambda_q \sum_{k,g} M_{p,k,g}\log q_{p,k,g}
\;+\;
\tau \sum_{k,g} M_{p,k,g}\log M_{p,k,g}
\]

其中：

\[
\hat T_{p,g,c} = \sum_k A_{c,k} M_{p,k,g}
\]

`A_{c,k}` 表示类型 `k` 是否计入约束 `c`。

约束条件为：

\[
\sum_g M_{p,k,g} = n_{p,k}, \quad \forall k
\]

并在需要时加入：

\[
\sum_k M_{p,k,g} = T_{p,g,\text{pop}}, \quad \forall g
\]

三项分别表示：

- 第一项：匹配 `ACS` 小区域目标
- 第二项：鼓励分配符合 mobility 驱动的 compatibility prior
- 第三项：最大熵正则，避免塌缩与过硬匹配

### 9.1 数学口径

这个目标本质上属于：

- maximum entropy coupling
- entropic OT / entropic matching
- ecological inference 下的 prior-regularized disaggregation

这比普通回归更贴近问题结构，因为我们真正知道的是：

- 行边际 `n_{p,k}`
- 部分列边际 `T_{p,g,c}`
- 以及一个 data-driven 的 preference `q_{p,k,g}`

---

## 10. 求解策略

二期的求解器不需要一上来就做复杂端到端神经网络。  
第一版建议：

### 10.1 v0 求解器

1. 先构造 `q_{p,k,g}`
2. 用

\[
M^{(0)}_{p,k,g} = n_{p,k} \cdot q_{p,k,g}
\]

初始化

3. 在 `PUMA` 内做熵正则迭代修正：
   - 满足 hard constraints
   - 逐步减少 soft residuals

4. 末尾做小规模 local swap / refinement

### 10.2 为什么不直接上 end-to-end

因为二期当前最需要的是：

- 明确问题结构
- 分辨哪些偏差来自 prior，哪些来自约束
- 让失败模式可诊断

直接端到端会过早掩盖这些结构问题。

---

## 11. 输出与下游接口

二期的主要输出有两层。

### 11.1 聚合输出

- `M_{p,k,g}`：type-to-CBG 分解矩阵
- `cbg_assignment_summary.json`

### 11.2 individual 展开输出

把 `M_{p,k,g}` 展开回一期生成的 individual persons，得到：

- `synthetic/persons_geo.parquet`

建议字段：

- `person_id`
- `puma`
- `cbg_geoid`
- `cell_id`
- `assignment_score`
- `assignment_source`
- `shrinkage_level`

然后下一步再进入：

\[
\text{CBG} \to \text{building}
\]

由第三阶段显式完成 `CBG -> building` 分配。

---

## 12. 默认决策

如果当前就要为二期定一个默认方法版本，建议如下：

1. 保留 full-earn 的 5D full cell，不粗暴预合并
2. 对稀疏 cell 做层级收缩，而不是独立拟合
3. 把 `CBG total population` 设为默认硬约束
4. 把其余 `ACS` 小区域边际设为软约束
5. 把 mobility 只当空间异质性 prior，不当个体真值
6. `q(g|k)` 使用低秩 compatibility model，而不是黑箱大网
7. 外层求解用 maximum-entropy / entropic coupling，不写成普通回归问题

---

## 13. 这套方法真正回答了什么

二期的核心贡献不是“从手机数据预测 individual income / education”，而是：

> 在没有 person-level home CBG 真值的情况下，利用一期 full-joint、ACS 小区域边际与 mobility 弱监督，恢复 `PUMA` 内不同人口类型的 `CBG` 级空间异质分布。

更具体地说，二期试图回答：

- 哪类人更可能落到哪类行为画像的 `CBG`
- 这种空间异质性如何在 `PUMA` 内部分解一期学到的 full-joint mass

它不试图回答：

- 某个真实 individual 的 home CBG 是什么
- mobility 是否能精确回归个体 SES

这条边界必须在文档和后续论文中保持清楚。

---

## 14. 审核点

这份方案稿落地前，需要你拍板的点有四个：

1. `CBG total population` 是否在二期中默认硬化
2. 第一版 `ACS` 软约束优先包含哪些维度
3. `mobility profile` 的语义轴是否接受“先做可解释聚合，再考虑低维投影”
4. 二期是否先做 Detroit case study，再扩到更大范围

---

## 15. 与现有文档的关系

这份文档与现有材料的分工如下：

- 一期 full-earn 主线：负责 `P(attrs | PUMA)` 的学习
- 本文档：负责 `P(CBG | attrs, PUMA)` 的空间分解
- 后续三期：负责 `CBG -> building`

相关内部文档：

- `docs/framework_layout_draft.md`
- `docs/research_framework_v1.md`
- `docs/synthpop_architecture.md`
- `../Mobility_Population/docs/data_overview.md`
- `../Mobility_Population/docs/data_fusion.md`

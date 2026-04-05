# 三期方法方案稿：从 tract/CBG 级人口到基于路网的显式 home/work 位置

> 状态：方案稿，待审核  
> 角色：三期方法文档，对应 `small-area -> explicit locations`  
> 对应关系：二期回答“人在哪个 tract/CBG”；三期回答“人在 tract/CBG 内落到哪里”

---

## 0. 核心结论

三期不应继续被表述成“`CBG -> building allocation`”，至少在当前数据口径下不应该。

原因很简单：

- 当前可直接复用的是 `road` shapefile，而不是 `building polygon`
- `MTFCC` 是道路功能类别，不是严格的土地利用或建筑类型标签
- 参考实现也是在道路线上插值采样 `home/work` 候选点，而不是把人放进真实建筑物内部

因此，三期更准确的定义应当是：

> **从 tract/CBG 级 synthetic population 到基于路网支持集的显式 home/work 位置分配**

在这个定义下，三类信息的角色需要明确分层：

- `ACS / 二期输出`：决定每个 tract/CBG 里有多少什么样的人
- `MTFCC road mask`：决定这些人“可以落在哪里”
- `mobility`：默认**不进入** home/work 支持集定义；若以后使用，也只能在候选点集合固定后做软重权，而不能与道路支持定义混在一起

这一步的关键不是再注入一层 demographic prior，而是把二期已得到的 small-area population 显式地落到可解释、可复现的空间支持集上。

---

## 1. 三期回答的问题

二期已经回答了：

\[
P(\text{small-area} \mid \text{attrs}, \text{PUMA})
\]

也就是：

- 同一个 `PUMA` 里的不同类型人口
- 应该如何分解到 tract/CBG

但二期没有回答：

> 在同一个 tract/CBG 内，这些人具体应该落到什么样的空间位置上？

三期回答的是一个**小区域内部显式落位**问题：

\[
P(\text{location} \mid \text{small-area}, \text{role})
\]

其中 `role` 至少包括：

- `home`
- `work`

但这里的 `location` 当前不是“真实建筑物 ID”，而是：

- 道路支持集上的候选点
- 或由这些候选点采样得到的最终显式位置

所以三期真正输出的是：

- `home_location`
- `work_location`
- 以及这些位置的生成来源和 fallback 标记

---

## 2. 为什么这一步不应继续叫 building allocation

如果当前数据和实现都还是以道路为支持，那么继续使用 `building allocation` 这个名字会带来两个概念混淆。

### 2.1 当前支持集不是建筑，而是道路

参考实现对 `home/work` 的处理是：

- `home`：在 `S1400|S1740` 道路上插值采样
- `work`：在 `S1100 + S1200` 道路上插值采样

这说明当前方法的物理对象是：

- `road-supported candidate points`

而不是：

- `residential building polygons`

### 2.2 `MTFCC` 是道路功能代理，不是严格 land-use 标签

例如：

- `S1400` 可以作为较强的 residential proxy
- `S1740` 可以作为较弱但可操作的补充支持
- `S1200` 更像工作/通勤相关道路的 proxy
- `S1100` 可作为更保守的主干工作支持补充

但这些都不是“住宅真值”或“办公楼真值”。

因此，如果方法层面仍然把这一步写成“building placement”，读者会自然误以为：

- 我们已经拿到了真实 building footprints
- 并且已经做了 building-level home/work assignment

这与当前数据条件不符，也不利于解释。

### 2.3 更准确的命名

建议统一改成以下口径之一：

- `road-constrained location assignment`
- `road-supported home/work location generation`
- `small-area explicit location allocation`

本方案稿统一采用：

> **road-constrained home/work location assignment**

---

## 3. 输入数据与边界

三期默认承接二期输出，不再重新决定人口在哪个 tract/CBG。

### 3.1 人口输入

三期输入应至少包含：

- `person_id`
- `tract_geoid` 或 `cbg_geoid`
- `AGE / SEX / SCHL / ESR / EARN` 等属性
- 若已存在则包含 `household_id`

这里有一个非常重要的边界：

- `home` 从概念上更适合按 `household` 分配
- `work` 从概念上更适合按 `worker/person` 分配

因此：

- **优选模式**：有 `household_id` 时，home 在 household 级分配，再广播给成员
- **v0 兼容模式**：若当前只有 person-level population，则先按 person 生成 home point，并在输出中显式标记这是 `person-home proxy`

### 3.2 空间输入

三期使用 Michigan cleaned road network：

- 路径：`/home/jinlin/data/geoexplicit_data/synthetic_city/data/reference/geo_synthetic_pop_usa/road/MI_road_cleaned.shp.zip`
- 关键字段：
  - `LINEARID`
  - `MTFCC`
  - `component`
  - `geometry`

还需要与之配套的小区域边界：

- tract polygon 或 CBG polygon

### 3.3 mobility 的边界

在三期的 v0 版本中，mobility **不参与**：

- home road mask 定义
- work road mask 定义
- 候选点支持集构建

mobility 如果以后进入三期，只允许出现在：

- 候选点集合已经固定之后的软重权
- 或下游 `POI / activity consistency` 模块

这样做的目的是避免把：

- 道路功能代理
- demographic prior
- 行为先验

混在同一层一起解释。

---

## 4. 小区域内部的道路支持集

三期的核心对象不是 building set，而是每个 tract/CBG 上的两类道路支持集：

\[
\mathcal{H}^{home}_g,\quad \mathcal{H}^{work}_g
\]

其中 `g` 表示 tract 或 CBG。

### 4.1 home 支持集

定义：

\[
\mathcal{H}^{home}_g
=
\{r \in \text{Roads} : r \cap g \neq \emptyset,\ \text{MTFCC}(r)\in \mathcal{M}_{home}\}
\]

这里 `\mathcal{M}_{home}` 有两种可解释口径。

#### 口径 A：Compatibility 模式

\[
\mathcal{M}_{home} = \{S1400, S1740\}
\]

优点：

- 与参考实现完全一致
- 支持集更稳，不容易出现候选点过少
- 更适合作为产品 v0 默认口径

缺点：

- `S1740` 解释上更像“私有服务道路补充”，语义略弱

#### 口径 B：Conservative 模式

\[
\mathcal{M}_{home} = \{S1400\}
\]

优点：

- 语义更干净
- 更贴近“住宅道路 proxy”的直觉

缺点：

- 会减少候选点覆盖
- 与参考仓库当前实现不完全一致

**建议**：

- 产品默认使用 `Compatibility`：`S1400 + S1740`
- 论文或敏感性分析额外报告 `S1400 only`

### 4.2 work 支持集

定义：

\[
\mathcal{H}^{work}_g
=
\{r \in \text{Roads} : r \cap g \neq \emptyset,\ \text{MTFCC}(r)\in \mathcal{M}_{work}\}
\]

当前参考实现实际使用：

\[
\mathcal{M}_{work} = \{S1100, S1200\}
\]

虽然主 work support 的核心口径是：

- `work = S1100 + S1200`

但真实 10-PUMA zero-fallback 诊断进一步暴露出一个更细的结构：

- 有一批 tract 确实存在 worker
- 但 tract 内完全没有 `S1100/S1200`
- 这些 tract 同时具有 `S1400`

因此，三期产品版默认不应再停留在单一的全局 `S1100 + S1200`，而应采用一个**显式、预声明的 tract-type 例外规则**：

- 常规 tract：`work = S1100 + S1200`
- `arterial-missing` tract：若 tract 内 `S1100 + S1200 = 0`，则显式改用 `S1400`

这个规则不是 runtime fallback，而是 support definition 的一部分。  
它的作用不是偷偷补洞，而是承认这样一类 tract 的道路功能结构本来就不提供 arterial-style work support。

### 4.3 几何裁剪与去重

对每个 tract/CBG：

1. 先筛选与 polygon 相交的道路
2. 再按 `MTFCC` 做支持集过滤
3. 将道路 geometry 裁剪到 tract/CBG 内部
4. 仅保留 `LineString / MultiLineString`
5. 按几何哈希去重

这一步与参考实现保持一致。

---

## 5. 从道路支持集到候选点集合

在得到 `\mathcal{H}^{home}_g` 和 `\mathcal{H}^{work}_g` 之后，三期并不直接把人贴到整条 road 上，而是先把道路离散成可采样的候选点集合。

### 5.1 插值采样

对支持集中的每条 road segment，沿线按固定密度插值：

- `home`：插值密度沿用参考实现 `0.0005`
- `work`：插值密度沿用参考实现 `0.0002`

这样得到：

\[
\mathcal{C}^{home}_g,\quad \mathcal{C}^{work}_g
\]

即 tract/CBG `g` 内的 home/work 候选点集合。

### 5.2 work 的补充候选点

参考实现还会把 home-road 的交点或端点补入 `work` 候选集合。  
但在当前版本里，这个逻辑更适合作为**诊断性对照**，而不是产品默认。

- `work` 的主支持来自 `S1100 + S1200`
- 如果这套口径仍覆盖不足，首先应该把 failure 暴露出来，而不是默认切换到另一套支持语义

因此，三期主线应采用：

- 常规 tract：`work = S1100 + S1200 interpolation only`
- `arterial-missing` tract：`work = S1400 interpolation only`
- `home-road intersection` 只保留为可选对照，不进入默认 baseline

---

## 6. 分配对象与分配规则

### 6.1 home 分配

home 的概念最自然的分配对象是 household，而不是 person。

因此：

- **优选模式**：对每个 tract/CBG 的 synthetic households 采样 `|\mathcal{C}^{home}_g|` 上的位置
- 再把同一 household 的 home point 赋给全部 household members

如果当前阶段尚无 household layer，则采用兼容模式：

- 对每个 person 在 `\mathcal{C}^{home}_g` 上独立采样
- 并在输出中写入 `home_assignment_mode = person_proxy`

### 6.2 work 分配

work 是 person-level 或 worker-level 对象。

建议先定义 eligible workers，例如：

- 根据 `ESR` 过滤 employed persons

然后仅对 eligible workers 在 `\mathcal{C}^{work}_g` 上采样。

非 worker 人群：

- `work_location = null`

### 6.3 采样方式

三期 v0 不需要复杂打分器，先采用显式、可审查的采样规则：

- tract/CBG 内均匀采样
- 允许 replacement
- 显式输出 `no_candidates` 与 `unassigned_no_candidates`

后续若要加入道路长度、component 或局部密度权重，可在此基础上扩展，但不改变“道路支持集优先”的主逻辑。

---

## 7. zero-fallback 基线与诊断性对照

三期产品版默认不应依赖 fallback。  
一旦需要 fallback，含义就变成：

- 当前支持集定义不足
- 或当前 tract/CBG 在该角色下并没有可解释的显式支持

因此，fallback 最多只能是诊断性对照，不能算“成功覆盖”。

### 7.1 产品默认：zero-fallback

产品版 baseline 应采用：

1. `home` 只在选定的 home mask 上生成候选  
2. `work` 只在预定义的 work support 上生成候选  
   即：常规 tract 用 `S1100 + S1200`，`arterial-missing` tract 用 `S1400`
3. 若候选为空，则保留 `no_candidates`  
4. 分配阶段若无候选，则显式输出 `unassigned_no_candidates`

这种设计的好处是：

- 支持语义始终一致
- failure 会被真实暴露，而不是被后处理掩盖
- 不会把“能落点”和“偷偷换定义之后能落点”混为一谈

### 7.2 可选诊断对照

如果确实要做 fallback，对照口径也应显式写清，而不能混进默认结果：

1. `home`: `S1400 only` 与 `S1400 + S1740` 做并列比较，而不是运行时自动升级  
2. `work`: `S1100 + S1200`、`S1100 + S1200 + S1400`、以及 `arterial-missing -> S1400` 显式例外做并列比较，而不是运行时自动补洞

这类对照的作用是诊断支持集是否不足，而不是生成最终结果。

---

## 8. ACS、MTFCC 与 mobility 的角色分离

这是三期最需要讲清楚的部分。

### 8.1 ACS 的角色

ACS 在三期之前已经完成了它的主要工作：

- 决定 tract/CBG 级人口总量与 demographic composition

因此，三期不再让 ACS 决定“落在哪条路上”。

### 8.2 MTFCC 的角色

`MTFCC` 在三期中承担的是：

> **hard spatial support definition**

也就是：

- 哪些道路可以作为 home support
- 哪些道路可以作为 work support

### 8.3 mobility 的角色

在三期 v0 中，mobility 不参与 support 定义。

原因不是 mobility 没价值，而是：

- 如果 tract/CBG 内的 home/work support 已经由 `MTFCC` 定义
- 再让 mobility 同时决定“哪里可住/可工作”
- 解释上就会把 physical plausibility 和 behavior prior 混在一起

因此，三期的 clean design 是：

- `MTFCC` 决定可行域
- `mobility` 若以后使用，只在可行域内部做排序或重权

这会让方法解释明显更清楚。

---

## 9. 输出对象

三期建议至少输出四类表。

### 9.1 候选点表

- `home_candidates.parquet`
- `work_candidates.parquet`

字段建议包括：

- `candidate_id`
- `small_area_id`
- `candidate_role`
- `geometry`
- `source_mtfcc`
- `source_stage`

### 9.2 最终分配表

- `person_locations.parquet`

字段建议包括：

- `person_id`
- `tract_geoid` / `cbg_geoid`
- `home_location`
- `work_location`
- `home_assignment_mode`
- `home_source_stage`
- `work_source_stage`
- `home_fallback_flag`
- `work_fallback_flag`

### 9.3 household-level 表（若可得）

- `household_locations.parquet`

### 9.4 tract/CBG 诊断表

- `group_diagnostics.csv`
- `no_candidate_groups.csv`

其中应至少包含：

- `n_persons`
- `n_workers`
- `n_home_candidates`
- `n_work_candidates`
- `n_work_unassigned`
- `work_candidates_per_worker`

这两张表的作用不是做最终结果，而是把 zero-fallback 下仍然失败的小区域直接暴露出来，供下一阶段 destination/support 建模使用。

---

## 10. 评估与敏感性分析

三期的评估重点不再是 demographic TVD，而应转向：

- 候选点覆盖率
- zero-fallback coverage
- tract/CBG 内 assignment completeness
- `home` 的 `S1400 only` vs `S1400 + S1740` 敏感性
- worker assignment completeness
- 与 POI / mobility 的下游一致性（放在后续模块）

这一步的核心问题不是“是否更接近 ACS”，而是：

> 在不引入难解释黑箱的前提下，是否得到一个物理上合理、显式可审查的小区域位置支持层。

---

## 11. 推荐的产品版默认口径

如果现在就要定一个可落地的 v0，建议如下：

1. 三期统一命名为 `small-area -> road-constrained home/work locations`  
2. `home` 默认优先走 `S1400 only`，并显式检查是否出现 `no_candidates`  
3. `work` 默认使用 `S1100 + S1200`，并对 `arterial-missing` tract 显式启用 `S1400` 例外支持  
4. `home` 优先 household-level；若当前无 household layer，则暂时采用 person proxy  
5. 产品 baseline 默认 `zero-fallback`，不自动扩展到其他支持语义  
6. `mobility` 不进入三期 v0 的 support 定义与 home assignment  
7. mobility 留到：
   - 下游 `POI / activity assignment`
   - 或候选点集合已固定后的软重权版本

---

## 12. 一句话总结

三期真正要做的不是“把人放进 building”，而是：

> **在二期已确定 tract/CBG 归属之后，利用 `MTFCC` 定义可解释的道路支持集，把 synthetic population 显式地落到 tract/CBG 内部的 home/work 候选位置上。**

在这个框架里：

- `ACS` 负责“放多少什么人”
- `MTFCC` 负责“哪里可以放”
- `mobility` 若以后进入，只负责“在可放的位置里，哪里更像真实行为”

这三个层次一旦拆开，方法就不会难解释。

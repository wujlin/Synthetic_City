# 实验日志：Paper 2 Mobility Destination

## 2026-04-16

### 新理论骨架决定

当前第二篇文章不再只围绕 `mobility 是否改善 destination` 展开，而是把 `Paper 1` 和 `Paper 2` 放进同一个 world model：

- 给定约束 `c`，先定义一个最大熵平衡态 `p_eq(c)`
- 真实分布写成 `p_true = p_eq(c) + δ(c)`
- 学习模型的价值，不在于重复约束，而在于恢复 **可预测的残差场**

这一步的作用不是增加一层概念包装，而是把后续实验从“继续调模型”改成“先测信号是否可识别”。

### 新实验优先级决定

先暂停把 `Paper 2` 的精力继续放在更复杂的 destination tweaking 上，先做一个最干净的 `P0`：

- 数据：PUMA-level 5-variable joint
- 约束：marginals only
- 参考系：pure MaxEnt，不带 national seed
- 问题：
  1. `δ` 是否显著存在？
  2. `δ` 是否能从约束 profile held-out 预测？

如果这一步都站不住，后面无论是 `Paper 1` 的 diffusion 解释，还是 `Paper 2` 的 coarse-to-fine residual，都缺少统一的理论支点。

### 已落地的最小工程入口

- 理论 note：
  - [`docs/equilibrium_residual_world_model_2026-04-16.md`](/Users/jinlin/Desktop/Project/Synthetic_City/docs/equilibrium_residual_world_model_2026-04-16.md)
- P0 实验脚本：
  - [`tools/exp_equilibrium_residual_predictability.py`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/exp_equilibrium_residual_predictability.py)

当前脚本做的是：

- 从现有 5-variable PUMA joint 表出发
- 直接构造 pure MaxEnt baseline
- 计算每个区域的 `TVD(p_true, p_eq)` 和 `KL(p_true || p_eq)`
- 用 `log_ratio = log(p_true) - log(p_eq)` 做 PCA
- 用 held-out ridge 预测 PC score，输出可预测残差方差占比

### 当前执行策略

先跑 `K = 32 / 128 / 512` 三档 marginals-only：

- 不混入 pairwise
- 不混入 hierarchical
- 不混入 destination mobility

先看两个量：

1. `equilibrium deviation` 是否随 `K` 增大显著升高
2. `predictable residual share` 是否随 `K` 增大显著下降

如果这两个趋势同时成立，`Paper 1` 的 breakdown 和 `Paper 2` 的 coarse-to-fine 主张就有了共同的解释基础。

### P0 首轮结果

本地已完成一轮 `pure MaxEnt residual` 计算：

- 输入：
  - [`outputs/_tmp_puma5var_us_smoke/puma_5var_joint_wide.csv`](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_tmp_puma5var_us_smoke/puma_5var_joint_wide.csv)
- 输出：
  - [`outputs/_exp_equilibrium_residual_predictability_20260416T125415Z/run_summary.json`](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_exp_equilibrium_residual_predictability_20260416T125415Z/run_summary.json)
  - [`outputs/_exp_equilibrium_residual_predictability_20260416T125415Z/metrics/concise_summary.json`](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_exp_equilibrium_residual_predictability_20260416T125415Z/metrics/concise_summary.json)

关键数字如下：

| K | held-out TVD(`p_true`,`p_eq`) | held-out KL(`p_true || p_eq`) | 可预测残差占比（nonneg R2 加权） | `R^2 >= 0.1` 的残差方差占比 |
|---|---:|---:|---:|---:|
| 32 | 0.2896 | 0.2879 | 0.1633 | 0.5031 |
| 128 | 0.4519 | 0.6703 | 0.1080 | 0.1967 |
| 512 | 0.5124 | 0.9058 | 0.0808 | 0.1251 |

### 当前解释

这轮结果已经支持两条重要判断：

1. `equilibrium deviation` 随 `K` 增大明显上升。  
   这说明问题并不是“真实世界几乎等于 MaxEnt”，而是联合结构偏离平衡态的幅度本身在增大。

2. held-out 可预测残差占比在下降。  
   这说明随着状态空间变细，虽然残差场更大，但其中能被 marginals 稳定推出的部分在变少。

所以 `K` 的 breakdown 更像：

- 不是残差不存在
- 而是残差越来越大，但越来越难从当前约束识别

### 当前下一步

下一步不重训 diffusion，而是先把已有 `K = 32 / 128 / 512` 的 marginal-conditioned diffusion 结果接到这组 `pure MaxEnt` 参考上，回答一个更尖锐的问题：

- diffusion 恢复的是残差总量的多少比例？
- 这个“残差恢复率”是否比 raw `Δ = TVD_MaxEnt - TVD_diffusion` 更能对应 `predictable residual share`？

## 2026-04-10

### 背景判断

- 第一篇文章的主线已经基本定型，重点在 demographic structure 与空间化结果
- 当前 pipeline 的下一个明显缺口是 `work destination`
- `mobility` 不应继续作为 tract demographic allocation 的 competing prior
- 更合理的位置是 destination layer

### 已确认的事实

1. 当前 work 失败的主要原因不是点位 geometry，而是上游缺少 `destination tract` 建模。
2. 仓库中已经存在 Detroit work destination 的轻量实现：
   - `LODES tract OD`
   - `distance deterrence`
   - `earn/age segment weights`
   - `gravity accessibility`
   - `job center` 相关特征
3. 现有 mobility 验证脚本已经能评估：
   - tract share
   - OD share
   - commute distance

### 论文定位决定

第二篇文章不写成：

- mobility 替代 gravity
- mobility 预测 SES
- mobility 覆盖 ACS demographic constraints

而写成：

- gravity / LODES 给出 destination skeleton
- mobility 提供 behavior-driven residual heterogeneity
- 目标是把 synthetic population 从 demographic consistency 推进到 interaction realism

### 首轮实验路线决定

先不直接做复杂 joint model。  
先跑一个结构清楚的 ladder：

- `B0` current baseline
- `B1` distance-only
- `B2` gravity + LODES
- `B3` gravity + LODES + worker/home context
- `M1` + mobility

### 当前待办

1. 确认 Detroit 首轮实验所需输入路径
2. 固化 `B0-B3` 参数版本
3. 设计 mobility feature table 的最小版本
4. 再决定 `M1` 用 tract-level feature 还是 OD-level feature

### 已确认的可直接复用输入

- `persons_with_households`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase25_detroitmsa_core_households_20260329T093404Z/synthetic/persons_with_households.parquet`
- `tract_od` 候选
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_lodes_tract_od_detroitmsa_core_2020_homecenterfix_20260330T050954Z/tract_od.csv`
- `mobility_csv`
  - `/home/jinlin/data/Mobility_Data/place20190206.csv`

### 新决定

首轮 baseline ladder 先不混入 `POI blend`。

原因：

- `POI` 会让 destination attractiveness 来源变多
- 不利于判断 `mobility` 是否真的提供了独立增量
- 第一轮应先比较：
  - `distance / gravity`
  - `LODES`
  - `worker/home context`
  - `mobility`

等这条主线成立，再讨论 `POI` 是否作为 refinement 接入

### 新增工程入口

已建立统一 baseline 脚本：

- [`tools/run_detroit_destination_baseline_v1.sh`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/run_detroit_destination_baseline_v1.sh)

当前覆盖：

- `distance_only`
- `gravity_lodes`
- `contextual`

这意味着首轮 `B1/B2/B3` 已经不需要手工拼接命令。

### WSA 实跑已启动

已在 WSA 后台顺序启动 Detroit 三档 baseline：

- `distance_only`
- `gravity_lodes`
- `contextual`

远端信息：

- host: `10.13.12.164` / `dell-desktop`
- pid: `1783929`
- log: `/tmp/paper2_detroit_baselines_20260410T154113.log`

启动检查显示：

- `distance_only` 已进入运行阶段
- `persons / tract_od / mobility / roads / tiger_bg` 输入路径均已确认存在

### 当前下一步

1. 等待 `distance_only` 跑出第一批输出
2. 抓取 `phase3b / phase3 / validation` 的 run summary
3. 形成 `B1/B2/B3` 的首轮对照表

### 首个中间结果：distance_only 的 phase3b 已完成

远端输出：

- `/home/jinlin/projects/Synthetic_City/outputs/_phase3b_detroit_dest_distance_only_20260410T074113Z/`

当前已确认：

- `assignment_mode = independent`
- `distance_beta = 0.08`
- `work_destination_assigned = 2,696,819`
- `work_destination_unassigned = 0`
- `same_tract_share_among_assigned = 0.1127`

这说明：

- 仅靠距离衰减，已经能显著摆脱旧版近乎完全留在 home tract 的极端偏置
- 但它仍只是 destination backbone 的第一层结果
- 还需要看 `phase3` 与 mobility validation 才能判断 tract / OD / commute 的整体表现

### 首轮 baseline ladder 已全部完成

WSA 上三档顺序任务已全部完成：

- `distance_only`
- `gravity_lodes`
- `contextual`

当前第一轮关键指标如下：

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| distance_only | 0.1127 | 0.5554 | 0.8415 | 0.3009 | 0.0281 | 0.2324 | 0.9087 | 0.9692 | 0.1459 | 8.214 |
| gravity_lodes | 0.1000 | 0.5700 | 0.8368 | 0.3045 | 0.0273 | 0.2294 | 0.9132 | 0.9764 | 0.1321 | 8.708 |
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |

### 初步判断

1. `same-tract share` 随着模型结构增强而持续下降，说明 destination layer 正在摆脱旧版过强的本地化偏置。
2. `work tract Spearman` 从 `distance_only -> gravity_lodes -> contextual` 持续上升，说明 richer destination structure 对 tract-level 排序确实有帮助。
3. 但 `work tract cosine / TVD` 并未同步改善，尤其 `contextual` 的 tract TVD 反而变差，说明更强的条件化开始在“排序一致性”和“总量分配一致性”之间产生张力。
4. `work OD` 仍然整体很弱，三档都没有突破，说明当前 first-round ladder 仍主要改善了 destination ranking / commute realism，而没有真正解决 OD interaction structure。
5. `commute` 这一层改善最明显：`contextual` 已把 synthetic median commute 拉到 `9.98 km`，非常接近 mobility 的 `9.33 km`，且 `commute TVD` 降到 `0.090`。

### 当前结论

第一轮结果支持一个很重要的判断：

- `gravity / LODES / context` 已经足以显著改善 **commute realism** 和一部分 **tract ranking**
- 但它们还不足以解决 **OD structure**
- 因此第二篇文章里把 `mobility` 放到 `OD residual / interaction compatibility` 这一层，仍然是必要且合理的

### 下一步

下一步不应继续盲调现有权重，而应补一个真正的 `M1 mobility-enhanced` 版本：

- tract-level destination mobility feature
或
- origin-destination mobility compatibility feature

然后直接与当前 `contextual` 做 head-to-head 比较

### 当前风险

- mobility 可能只是重复 destination attractiveness，而不是补 interaction residual
- 即使 tract 指标改善，OD 也可能仍然弱
- 如果 mobility 只能提升 tract ranking，论文主张需要收缩

### 下一步

进入首轮实验准备：

- 盘点 Detroit 数据输入
- 把现有可跑 baseline 串成统一 runbook
- 再定义 mobility 中间特征表

### M1 的最小实现已落地

当前已经把 `mobility` 接入到 `work destination allocator`，但不是作为新的 tract-level attractiveness，而是作为：

- `home tract -> work tract` 的 `OD pair prior`

具体做法：

1. 从 mobility stay 数据中提取 `home/work anchor`
2. 把 anchor 点 join 到 BG，再聚合到 tract-OD
3. 对每个 `home tract` 计算：
   - `lodes_od_share`
   - `mobility_od_share_smoothed`
   - `mobility_od_residual = mobility_od_share_smoothed / lodes_od_share`
4. 在 allocator 中新增 `od_pair_prior_col` 与 `od_pair_prior_weight`

这样做的含义是：

- `LODES` 继续提供主骨架
- `mobility` 只修正 origin-conditional destination preference
- 当 `od_pair_prior_weight = 1` 时，destination choice 会完全退化为平滑后的 mobility conditional share
- 当 `0 < weight < 1` 时，本质上是在 `LODES skeleton` 和 `mobility residual` 之间做几何插值

这比直接把 mobility 塞成 tract attractiveness 更符合第二篇文章的主线。

### Mobility OD prior 预处理结果

WSA 已完成：

- `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_od_pair_prior_20260410T083332Z/tract_od_with_mobility.csv`

关键统计：

- mobility bbox 内事件数：`672,868`
- devices：`416,591`
- home anchors：`144,382`
- valid home-work devices：`7,628`
- LODES candidate tract-OD pairs：`554,057`
- 其中被 mobility 命中的 pairs：`4,037`
- nonzero share：`0.73%`
- raw OD share alignment：
  - Spearman `0.097`
  - cosine `0.242`
  - TVD `0.957`

这说明：

1. mobility 的 tract-OD 信息非常稀疏，不能替代 `LODES`
2. 但它正适合作为一个 sparse residual prior
3. 第二篇文章里不应把 mobility 写成 primary skeleton，而应写成 interaction refinement

### 当前运行状态

`mobility_m1` 已在 WSA 后台启动：

- run tag: `20260410T083535Z`
- log: `/tmp/paper2_detroit_m1_run_20260410T083535Z.log`
- enriched tract_od:
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_od_pair_prior_20260410T083332Z/tract_od_with_mobility.csv`

当前参数：

- mode: `mobility_m1`
- assignment mode: `hierarchical_county_center`
- `od_pair_prior_col = mobility_od_residual`
- `od_pair_prior_weight = 0.5`

这轮实验的目标不是追求 tract 指标全面提升，而是看：

1. `work OD` 是否优于 `contextual`
2. `commute realism` 是否基本不被破坏
3. tract ranking 是否保持在可接受水平

### M1 第一轮结果（`od_pair_prior_weight = 0.5`）

已完成运行：

- `phase3b`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3b_detroit_dest_mobility_m1_20260410T083535Z/`
- `phase3`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_detroit_dest_mobility_m1_nj48_20260410T083535Z/`
- `validate`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_m1_20260410T083535Z/`

关键指标：

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_m1_w0.5 | 0.0458 | 0.5774 | 0.8330 | 0.2933 | 0.0474 | 0.2650 | 0.9352 | 0.9770 | 0.1161 | 11.866 |

### 对这轮结果的判断

1. `same-tract share` 从 `0.076 -> 0.046`，说明 mobility residual 明显加强了跨 tract destination dispersion。
2. `work OD Spearman` 与 `work OD cosine` 都明显提升，说明这条线确实在补 interaction structure，而不只是重复 tract attractiveness。
3. `work tract cosine / TVD` 反而也优于 `contextual`，说明 M1 不是简单破坏 tract allocation。
4. 但 `work tract Spearman` 略降，说明 tract ranking 有轻微牺牲。
5. 更重要的是 `commute median` 被拉长到 `11.87 km`，明显高于 mobility 的 `9.33 km`，而 `commute TVD` 也从 `0.090` 恶化到 `0.116`。

当前最合理的解释是：

- `w=0.5` 对稀疏 mobility residual 赋权过强
- 它带来了真实的 OD 增益
- 但同时过度削弱了原本由 `distance + gravity + context` 维持的 commute realism

因此，这不是失败，而是说明：

- `mobility` 的接入位置是对的
- 但当前权重太大
- 下一步应该沿着小权重做 trade-off sweep，而不是退回“mobility 无效”的结论

### 已追加 follow-up

已在 WSA 后台追加更轻的版本：

- run tag: `20260410T084245Z_w025`
- `od_pair_prior_weight = 0.25`

它的目的很明确：

- 尽量保住 `OD` 增益
- 同时把 `commute` 拉回到 `contextual` 附近

### M1 第二轮结果（`od_pair_prior_weight = 0.25`）

已完成运行：

- `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_m1_20260410T084245Z_w025/`

关键指标：

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_m1_w0.25 | — | 0.5873 | 0.8352 | 0.2989 | 0.0371 | 0.2424 | 0.9286 | 0.9837 | 0.0932 | 10.846 |
| mobility_m1_w0.5 | 0.0458 | 0.5774 | 0.8330 | 0.2933 | 0.0474 | 0.2650 | 0.9352 | 0.9770 | 0.1161 | 11.866 |

这里的 `w=0.25` 明显比 `w=0.5` 更平衡：

1. `OD` 仍然优于 `contextual`
2. `tract cosine / TVD` 继续优于 `contextual`
3. `commute` 基本被拉回到可接受范围

但这一步仍然只是 `proof-of-signal`，不能成为最终主模型，因为 tract-OD coverage 依旧过稀。

### Coverage diagnostics 已完成

已完成运行：

- `/home/jinlin/projects/Synthetic_City/outputs/_diagnose_detroit_mobility_destination_coverage_20260410T105505Z/`

关键结果：

| level | n_units | hit share | covered LODES mass share | Spearman | cosine | TVD |
|---|---:|---:|---:|---:|---:|---:|
| tract-OD pair | 554,057 | 0.73% | 4.34% | — | — | — |
| tract -> center | 131,431 | 2.59% | 18.55% | 0.230 | 0.539 | 0.815 |
| tract -> county | 22,019 | 9.61% | 58.51% | 0.482 | 0.764 | 0.460 |

top-K 结果：

| top-K within origin | hit share | covered LODES mass share | origins with any hit |
|---|---:|---:|---:|
| 5 | 9.76% | 16.62% | 38.06% |
| 10 | 7.24% | 13.12% | 49.76% |
| 20 | 4.94% | 10.16% | 60.57% |
| 50 | 2.79% | 7.37% | 71.50% |

### 对诊断结果的当前判断

1. tract-OD 全空间确实过稀，不能作为主监督。
2. county 层面的 coverage 和 alignment 明显最好，说明 mobility 在更粗 destination state 上是稳定可用的。
3. center 层面虽然弱于 county，但显著优于 tract-OD，说明它仍有希望作为更细的上层状态候选。
4. top-K tract 候选内部的命中率已经远高于全空间，说明 mobility 更适合做 candidate-set reranking，而不是 global prior。

### 当前方法主线的收敛方向

第二篇文章的主方法应从：

- `tract-OD mobility residual`

收敛到：

- `coarse destination state + tract refinement`

当前最稳的两个候选是：

1. `home tract -> work county -> tract within county`
2. `home tract -> work center -> top-K tract reranking within center`

如果目标是“先拿到稳的可发表主结果”，应优先做第 1 条。  
如果目标是“在稳的基础上争取更强的方法贡献”，第 2 条应作为下一步主攻方向。

### M2 county-state 已实现并完成 Detroit 首轮实跑

新增工程入口：

- `tools/exp_prepare_mobility_county_prior.py`
- `tools/run_detroit_destination_baseline_v1.sh`
  - 新模式：`mobility_county_m2`

核心做法：

1. 从 `tract_od_with_mobility.csv` 聚合得到 `home tract -> work county` 的 mobility count
2. 计算：
   - `lodes_work_county_share`
   - `mobility_work_county_share_smoothed`
   - `mobility_work_county_residual`
3. 把 county residual 回填到每条 tract-OD 记录
4. 在 allocator 中使用：
   - `assignment_mode = hierarchical_county`
   - `od_pair_prior_col = mobility_work_county_residual`
   - `od_pair_prior_weight = 0.5`
5. 显式去掉 `same_county_weight`，避免在 county-state 模型里叠加手工“留在本县”的偏置

运行产物：

- county prior 预处理
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_county_prior_20260412T074842Z/`
- M2 county-state
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3b_detroit_dest_mobility_county_m2_20260412T075100Z/`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_detroit_dest_mobility_county_m2_nj48_20260412T075100Z/`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_county_m2_20260412T075100Z/`

### M2 county-state 结果

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_m1_w0.25 | — | 0.5873 | 0.8352 | 0.2989 | 0.0371 | 0.2424 | 0.9286 | 0.9837 | 0.0932 | 10.846 |
| mobility_county_m2 | 0.0720 | 0.5855 | 0.8182 | 0.3237 | 0.0296 | 0.2251 | 0.9240 | 0.9872 | 0.0809 | 10.421 |

### 对 M2 county-state 的当前判断

1. 它明显比 tract-OD residual 更稳。
2. `OD` 仍有增益，但增益幅度比 `M1` 小。
3. `commute` 反而是当前最好的一版：
   - cosine 最好
   - TVD 最低
4. tract 层几乎与 `contextual` 持平，没有出现显著退化。

因此，这一版非常适合作为：

- 第二篇文章的稳健主结果
- 一个“coarse destination state 比 sparse tract-OD residual 更可靠”的直接证据

### 当前主线收敛

如果只考虑稳健性与可发表性，当前应先把主方法收敛到：

- `home tract -> work county -> tract within county`

然后再把：

- `home tract -> work center -> top-K tract reranking`

作为下一步争取更强方法贡献的版本。

### M3 raw center-state 已实现并完成 Detroit 首轮实跑

新增工程入口：

- `tools/exp_prepare_mobility_center_prior.py`
- `tools/run_detroit_destination_baseline_v1.sh`
  - 新模式：`mobility_center_m3`

核心做法：

1. 从 `tract_od_with_mobility.csv` 聚合得到 `home tract -> work center` 的 mobility count
2. 计算：
   - `lodes_work_center_share`
   - `mobility_work_center_share_smoothed`
   - `mobility_work_center_residual`
3. 把 center residual 回填到每条 tract-OD 记录
4. 在 allocator 中使用：
   - `assignment_mode = hierarchical_county_center`
   - `od_pair_prior_col = mobility_work_center_residual`
   - `od_pair_prior_weight = 0.5`
5. 保留 center hierarchy，但去掉显式 `same_county_weight` 与 `same_home_center_weight`

运行产物：

- center prior 预处理
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_center_prior_20260413T001117Z/`
- M3 raw center-state
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3b_detroit_dest_mobility_center_m3_20260413T001249Z/`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_detroit_dest_mobility_center_m3_nj48_20260413T001249Z/`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_center_m3_20260413T001249Z/`

### M3 raw center-state 结果

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_county_m2 | 0.0720 | 0.5855 | 0.8182 | 0.3237 | 0.0296 | 0.2251 | 0.9240 | 0.9872 | 0.0809 | 10.421 |
| mobility_center_m3 | 0.0530 | 0.5834 | 0.7981 | 0.3514 | 0.0347 | 0.2381 | 0.9296 | 0.9708 | 0.1285 | 13.009 |

### 对 M3 raw center-state 的当前判断

1. `OD` 继续上升，说明 center 层确实带有比 county 更强的 interaction signal。
2. 但 tract share 与 commute 同时明显恶化，说明把 `center residual` 作为全局稠密 prior 仍然太强。
3. 这不是 center 线失败，而是说明它不能直接写成 `global center prior`；它必须转化为更局部的 candidate reranking。

因此，当前最合理的下一步不是继续调 `M3` 的全局权重，而是进入：

- `M4 = center hierarchy + top-K tract reranking within center`

它的原则应当是：

1. mobility 只奖励被观测到的 tract pair
2. mobility 不惩罚未观测 pair
3. mobility 只在 `(home tract, work center)` 的高概率 tract 候选内发声

这样才能把 center signal 保留下来，同时避免 raw center-state 把 commute 结构拉坏。

### M4 center + top-K reranking 已完成 Detroit 首轮实跑

新增工程入口：

- `tools/exp_prepare_mobility_center_topk_prior.py`
- `tools/run_detroit_destination_baseline_v1.sh`
  - 新模式：`mobility_center_topk_m4`

首轮具体设定：

1. 输入使用 `M3` 的 enriched tract-OD：
   - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_center_prior_20260413T001117Z/tract_od_with_mobility_center.csv`
2. 在每个 `(home tract, work center)` 内，按 `S000` 降序、`distance_km` 升序做 tract rank
3. 只对 `top-K` 且 `mobility_od_count > 0` 的 tract pair 给予 bonus
4. 未命中的 pair 一律保持中性 `1.0`

这意味着 `M4` 的 prior 已经不是 residual penalty，而是 **bonus-only local reranking**。

### top-K 诊断

`K=10` 在 `(home tract, work center)` 这一层几乎不裁剪候选：

- `share_topk_pairs = 0.894`
- `topk_lodes_mass_share_of_total = 0.947`

因此首轮正式实验不采用 `K=10`，而改用更有区分度的 `K=3`。

`K=3` 的预处理结果：

- 预处理产物：
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_center_topk_prior_k3_20260413T010800Z/`
- 关键统计：
  - `share_topk_pairs = 0.538`
  - `topk_lodes_mass_share_of_total = 0.711`
  - `share_topk_hit_within_hit_pairs = 0.567`
  - `share_origins_with_any_topk_hit = 0.716`

这说明 `K=3` 已经形成了有意义的 local candidate gate，同时没有把 observed mobility signal 裁得过碎。

### M4 首轮结果（`K=3`, `bonus-only`, `weight=1.0`）

运行产物：

- precompute
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_mobility_center_topk_prior_k3_20260413T010800Z/`
- phase3b
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3b_detroit_dest_mobility_center_topk_m4_20260413T003300Z_k3w10/`
- phase3
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_detroit_dest_mobility_center_topk_m4_nj48_20260413T003300Z_k3w10/`
- validate
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_center_topk_m4_20260413T003300Z_k3w10/`

关键指标：

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_county_m2 | 0.0720 | 0.5855 | 0.8182 | 0.3237 | 0.0296 | 0.2251 | 0.9240 | 0.9872 | 0.0809 | 10.421 |
| mobility_center_m3 | 0.0530 | 0.5834 | 0.7981 | 0.3514 | 0.0347 | 0.2381 | 0.9296 | 0.9708 | 0.1285 | 13.009 |
| mobility_center_topk_m4_k3w1.0 | 0.0759 | 0.5833 | 0.8225 | 0.3228 | 0.0307 | 0.2208 | 0.9232 | 0.9864 | 0.0906 | 9.981 |

### 对 M4 首轮结果的当前判断

1. `M4` 成功避免了 `M3` 的明显副作用：
   - `same-tract share` 基本回到 `contextual`
   - `commute` 几乎与 `contextual` 重合
2. `OD Spearman` 相比 `contextual` 有净提升：
   - `0.0234 -> 0.0307`
3. 但整体增益仍然偏保守：
   - tract 层几乎不变
   - `OD cosine` 只小幅提升，且仍未超过 `M2`

因此当前最合理的结论不是“`M4` 已经赢了”，而是：

- `M4` 证明了 **center signal 可以通过 local reranking 的方式安全接回模型**
- 但当前这组参数仍然偏弱，尚未形成比 `M2 county-state` 更强的主结果

这意味着下一步若继续推进 `M4`，最值得调的不是结构，而是 **bonus 强度**，例如：

1. 固定 `K=3`，提高 `od_pair_prior_weight`
2. 或在 bonus-only 结构下尝试更激进的 `K`

而如果只考虑当前最稳的论文主结果，`M2 county-state` 仍然是首选。

### M4 follow-up 已追加：`K=3`, `weight=2.0`

由于 `K=3, weight=1.0` 的结果方向正确但增益偏弱，因此已追加更强的 bonus 版本：

- `RUN_TAG = 20260413T004100Z_k3w20`
- `od_pair_prior_weight = 2.0`
- 其余结构保持不变

当前已确认：

- `phase3b` 已完成
- `same_tract_share_among_assigned = 0.07585`

这说明把 bonus 强度从 `1.0` 提高到 `2.0`，至少没有在 assignment 这一层引入新的明显副作用。  
最终 tract / OD / commute 指标现已完成。

### M4 follow-up 最终结果（`K=3`, `weight=2.0`）

运行产物：

- validate
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase3_validate_detroit_dest_mobility_center_topk_m4_20260413T004100Z_k3w20/`

关键指标：

| mode | same-tract share | work tract Spearman | work tract cosine | work tract TVD | work OD Spearman | work OD cosine | work OD TVD | commute cosine | commute TVD | synthetic commute median km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contextual | 0.0761 | 0.5846 | 0.8226 | 0.3226 | 0.0234 | 0.2189 | 0.9239 | 0.9865 | 0.0904 | 9.980 |
| mobility_county_m2 | 0.0720 | 0.5855 | 0.8182 | 0.3237 | 0.0296 | 0.2251 | 0.9240 | 0.9872 | 0.0809 | 10.421 |
| mobility_center_topk_m4_k3w1.0 | 0.0759 | 0.5833 | 0.8225 | 0.3228 | 0.0307 | 0.2208 | 0.9232 | 0.9864 | 0.0906 | 9.981 |
| mobility_center_topk_m4_k3w2.0 | 0.0758 | 0.5839 | 0.8230 | 0.3226 | 0.0371 | 0.2241 | 0.9220 | 0.9865 | 0.0904 | 9.980 |

### 对 M4 `k3w2.0` 的更新判断

1. 把 bonus 强度从 `1.0` 提高到 `2.0` 后，`M4` 的 tract 与 commute 基本仍和 `contextual` 重合，没有引入新的副作用。
2. `work OD Spearman` 从 `0.0234` 提升到 `0.0371`，`work OD TVD` 从 `0.9239` 降到 `0.9220`，说明更强的 local reranking 确实能把一部分 center signal 转译成更好的 OD 结构。
3. 但 `M4` 仍然没有在 county-level 形成像 `M2` 那样稳定而明确的 coarse-state 改善，因此它更像一个细化层增益，而不是当前最稳的主结果。

### 分层 destination evaluation 已完成

为避免继续被全矩阵 tract-OD 指标牵着走，已新增分层评估脚本：

- `tools/exp_validate_destination_layers.py`

对应产物：

- `contextual`
  - `/home/jinlin/projects/Synthetic_City/outputs/_validate_layers_detroit_contextual_m5_20260416T000001Z/`
- `mobility_county_m2`
  - `/home/jinlin/projects/Synthetic_City/outputs/_validate_layers_detroit_m2_m5_20260416T000001Z/`
- `mobility_center_topk_m4_k3w2.0`
  - `/home/jinlin/projects/Synthetic_City/outputs/_validate_layers_detroit_m4_k3w20_m5_20260416T000001Z/`

这里采用 `min_mobility_total = 5` 作为 relaxed threshold。严格阈值 `20` 时，within-county / within-center 的 eligible group 太少，不足以承载解释。

#### 1. county-level OD

| mode | county OD Spearman | county OD cosine | county OD TVD |
|---|---:|---:|---:|
| contextual | 0.6328 | 0.7237 | 0.3833 |
| mobility_county_m2 | 0.6446 | 0.7477 | 0.3675 |
| mobility_center_topk_m4_k3w2.0 | 0.6341 | 0.7240 | 0.3830 |

#### 2. center-level OD

| mode | center OD Spearman | center OD cosine | center OD TVD |
|---|---:|---:|---:|
| contextual | 0.2452 | 0.5041 | 0.7023 |
| mobility_county_m2 | 0.2526 | 0.5211 | 0.7003 |
| mobility_center_topk_m4_k3w2.0 | 0.2552 | 0.5055 | 0.7010 |

#### 3. within-county tract allocation（eligible groups = 200）

| mode | weighted mean Spearman | weighted mean cosine | weighted mean TVD | eligible mobility mass share |
|---|---:|---:|---:|---:|
| contextual | 0.0557 | 0.2866 | 0.8539 | 0.2346 |
| mobility_county_m2 | 0.0577 | 0.2859 | 0.8544 | 0.2346 |
| mobility_center_topk_m4_k3w2.0 | 0.0739 | 0.2946 | 0.8492 | 0.2346 |

#### 4. within-center tract allocation（eligible groups = 30）

| mode | weighted mean Spearman | weighted mean cosine | weighted mean TVD | eligible mobility mass share |
|---|---:|---:|---:|---:|
| contextual | 0.0234 | 0.3698 | 0.7278 | 0.0339 |
| mobility_county_m2 | 0.0365 | 0.3699 | 0.7323 | 0.0339 |
| mobility_center_topk_m4_k3w2.0 | 0.0616 | 0.3797 | 0.7166 | 0.0339 |

### 对分层评估的当前判断

1. `M2` 的真实增益集中在 **county-level destination choice**，这和它的建模层级完全一致。
2. `M4` 没有在 county-level 明显胜过 `M2`，但在 **conditional within-county / within-center tract refinement** 上有更清楚的弱增益。
3. 这说明 `M2` 和 `M4` 并不是互相替代的同类模型：
   - `M2` 主要修正 coarse destination state
   - `M4` 主要修正 finer local allocation
4. 因而当前更合理的论文表述，不是继续问“谁在全矩阵 tract-OD 上更高”，而是明确区分：
   - county 选错了多少
   - county 选对以后 tract 还错了多少

### destination ceiling / compatibility analysis 已完成

新增脚本：

- `tools/exp_analyze_mobility_destination_ceiling.py`

对应产物：

- 第一版：
  - `/home/jinlin/projects/Synthetic_City/outputs/_analyze_detroit_mobility_destination_ceiling_20260416T000001Z/`
- 加入 support-overlap 指标后的修订版：
  - `/home/jinlin/projects/Synthetic_City/outputs/_analyze_detroit_mobility_destination_ceiling_support_20260416T000001Z/`

anchor 规模：

- `n_events = 672,868`
- `n_devices = 416,591`
- `n_home_anchor_devices = 144,382`
- `n_work_anchor_devices = 7,628`
- `n_home_work_devices = 7,628`

#### cross-source compatibility（LODES vs mobility）

| level | n_units | Spearman | cosine | TVD | top-20 overlap | support Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| tract-OD | 555,995 | 0.0168 | 0.0155 | 0.9573 | 1 | 0.0073 |
| tract -> center | 131,968 | 0.1909 | 0.0448 | 0.8040 | 0 | 0.0339 |
| tract -> county | 22,484 | 0.3832 | 0.0953 | 0.4885 | 0 | 0.1165 |

#### mobility split-half reliability

| level | mean Spearman | mean cosine | mean TVD | mean top-20 overlap | mean support Jaccard |
|---|---:|---:|---:|---:|---:|
| tract-OD | -0.8805 | 0.9890 | 0.7927 | 5.9 | 0.0380 |
| tract -> center | -0.6653 | 0.9891 | 0.6830 | 8.4 | 0.1116 |
| tract -> county | -0.1709 | 0.9892 | 0.4757 | 11.3 | 0.3298 |

### 对 ceiling analysis 的当前判断

1. tract-OD 层不是“方法还没调好”，而是 **cross-source compatibility 本身就极弱**：
   - `LODES vs mobility Spearman = 0.0168`
   - `support Jaccard = 0.0073`
2. county 层虽然仍然不算高一致，但显著优于 tract-OD 和 center：
   - `Spearman = 0.3832`
   - `support Jaccard = 0.1165`
3. split-half 的负 Spearman 不能被机械解读为“真实负相关”；在这种极稀疏、离散且严格分半的设置下，它更接近一种 **秩统计失真告警**。
4. 更可信的 split-half 信号来自 support overlap：
   - tract-OD 的 mean top-20 overlap 只有 `5.9/20`
   - county 层升到 `11.3/20`
   - mean support Jaccard 也从 `0.038` 升到 `0.330`
5. 这些数字共同支持一个更稳的结论：
   - 当前 mobility 信号足以支持 **county-level correction**
   - 但不足以支撑 tract-OD 全矩阵上的强监督或强主张

### 当前实验路线的收敛结论

1. `M2 county-state` 目前仍是最稳的主结果，因为它在自己优化的 coarse layer 上给出了清楚而一致的改善。
2. `M4 center + top-K reranking` 的意义不在于取代 `M2`，而在于表明 center signal 可以作为 finer refinement 层安全接回，且确有弱增益。
3. 下一步不应继续在 `M4` 上做无约束的参数调优，而应把论文主线改写为：
   - **coarse-to-fine destination modeling under sparse behavioral supervision**
4. 对 tract-OD 指标的主张也应相应降级：
   - 重点不再是“mobility 让 tract-OD 大幅提升了多少”
   - 而是“mobility 在 coarse state 上有效，在 fine OD 上受 coverage 与 cross-source incompatibility 限制”

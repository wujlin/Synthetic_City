# Detroit 首轮 Runbook v1

## 1. 目标

用现有 Detroit 数据链先跑出一套 **baseline ladder**，验证这条新论文主线是否成立。

首轮只回答一个问题：

**mobility 在 `gravity + LODES + worker/home context` 之外，是否提供了独立增益？**

---

## 2. 已有可复用脚本

### 2.1 准备 tract OD 骨架

- [`tools/prepare_detroit_lodes_tract_od.py`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/prepare_detroit_lodes_tract_od.py)

它已经能输出：

- Detroit 内部 `tract_od.csv`
- `distance_km`
- `work_access_jobs_gravity`
- `work_access_job_centers_gravity`
- `work_center_geoid`

这意味着：

- `distance`
- `gravity accessibility`
- `job center`

这些 backbone 变量已经具备，不用从零实现。

### 2.2 分配 work destination tract

- [`tools/exp_phase3b_assign_work_destinations.py`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/exp_phase3b_assign_work_destinations.py)

它当前已经支持：

- `distance_beta`
- `od_age_segment_weight`
- `od_earn_segment_weight`
- `destination_access_weight`
- `destination_center_weight`
- `same_tract_weight`
- `same_county_weight`
- `same_home_center_weight`
- `job_family_weight`
- 多种按 `earn` / `age` 的 multiplier map

### 2.3 外部验证

- [`tools/exp_phase3_validate_mobility_anchor.py`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/exp_phase3_validate_mobility_anchor.py)

它已经能对比：

- home / work tract share
- OD share
- commute distance

### 2.4 当前已确认的 Detroit 起点资产

从仓库已有 `outputs/_sync_*` 汇总可直接回溯到以下起点：

- `phase2.5 persons`
  - `/home/jinlin/projects/Synthetic_City/outputs/_phase25_detroitmsa_core_households_20260329T093404Z/synthetic/persons_with_households.parquet`
- 一组可直接复用的 LODES tract OD 起点
  - `/home/jinlin/projects/Synthetic_City/outputs/_prepare_detroit_lodes_tract_od_detroitmsa_core_2020_homecenterfix_20260330T050954Z/tract_od.csv`
- 现成 mobility 验证输入
  - `/home/jinlin/data/Mobility_Data/place20190206.csv`
- 一组已知较稳的 destination / roadloc / validation 串联参考
  - `outputs/_sync_phase3_detroit_homecenter_20260330/`
  - `outputs/_sync_phase3_detroit_homecenter_poi_20260331/`

这意味着首轮不需要先重新铺完整数据链，可以直接从现成 Detroit 起点往前推进 baseline ladder。

### 2.5 统一 baseline 入口

已新增：

- [`tools/run_detroit_destination_baseline_v1.sh`](/Users/jinlin/Desktop/Project/Synthetic_City/tools/run_detroit_destination_baseline_v1.sh)

当前支持六档：

- `MODE=distance_only`
- `MODE=gravity_lodes`
- `MODE=contextual`
- `MODE=mobility_m1`
- `MODE=mobility_county_m2`
- `MODE=mobility_center_m3`
- `MODE=mobility_center_topk_m4`

其中：

- `mobility_m1`：tract-OD residual proof-of-signal
- `mobility_county_m2`：county-state coarse-to-fine 主版本
- `mobility_center_m3`：raw center-state 诊断版本
- `mobility_center_topk_m4`：center hierarchy + top-K tract reranking

---

## 3. 首轮 baseline ladder

## B0. 当前 best available baseline

目的：

- 作为现状参照
- 明确当前 `tract / OD / commute` 三层基线

输入：

- 现有 Detroit best run 的 `persons_with_worktract.parquet`
- 或重跑当前 best 参数

## B1. Distance-only

建议参数思路：

- 只保留 `distance_beta`
- 其余 segment / access / center 权重全置零

目的：

- 检查纯距离衰减是否已解释大部分 destination pattern

## B2. Gravity + LODES

建议参数思路：

- `distance_beta > 0`
- `destination_access_weight > 0`
- 其余 worker-type 特征先关闭

目的：

- 形成透明、可解释的 destination skeleton baseline

## B3. Gravity + LODES + worker/home context

建议参数思路：

- 在 `B2` 基础上加入：
  - `od_age_segment_weight`
  - `od_earn_segment_weight`
  - `destination_segment_weight`
  - `destination_age_segment_weight`
  - `same_home_center_weight`
  - `job_family_weight`

目的：

- 识别仅靠 demographic/home context 能解释多少 residual

## M1. Gravity + LODES + worker/home context + mobility

首轮不建议直接做复杂 end-to-end joint training。  
优先做两种可落地版本之一：

1. **destination-side mobility attractiveness**
   - 将 tract mobility 强度整理成 destination-side feature
2. **origin-destination mobility compatibility**
   - 将 home tract 与 candidate work tract 的 mobility relation 整理成附加打分项

目的：

- 检查 mobility 是否带来独立增益

## M2. Home tract -> work county -> tract within county

当前推荐的主版本。

实现方式：

1. 先把 tract-OD mobility 聚合成 `home tract -> work county`
2. 计算 `mobility_work_county_residual`
3. 在 `hierarchical_county` 模式下，把该 residual 只作用于 county choice
4. tract within county 的细化继续由：
   - LODES
   - distance
   - destination accessibility
   - origin/home context

当前首轮结果表明：

- 它比 tract-OD residual 更稳
- commute realism 最好
- OD 有小幅但净提升

## M3. Home tract -> work center -> tract within center（诊断版）

定位：

- 不是主结果
- 只用于确认 center 层确实存在更强 interaction signal

当前首轮结果表明：

- OD 继续上升
- 但 tract share 与 commute 明显恶化

因此它只能作为诊断版本存在，不能直接作为论文主模型。

## M4. Center hierarchy + top-K tract reranking within center

定位：

- 当前最值得推进的增强版本
- 在 `M2` 的稳健性与 `M3` 的更强 center signal 之间求折中

实现原则：

1. mobility 只奖励被观测到的 tract pair
2. mobility 不惩罚未观测 pair
3. mobility 只在 `(home tract, work center)` 的高概率 tract 候选内发声

这意味着 `M4` 不是全局 center prior，而是局部 reranking。

---

## 4. 首轮建议执行顺序

1. 确认 Detroit `areas`、`persons`、`mobility_csv`、`tract_od` 输入路径
2. 固化 `B0/B1/B2/B3` 四组参数
3. 每组都跑：
   - destination assignment
   - phase3 explicit work placement
   - mobility-anchor validation
4. 汇总 tract / OD / commute 指标
5. 跑 `M1` 证明 mobility 有信号
6. 跑 coverage diagnostics
7. 跑 `M2 county-state` 作为主结果候选
8. 跑 `M3 raw center-state` 做结构诊断
9. 跑 `M4 center + top-K reranking` 争取 stronger method contribution

### 当前推荐的首轮起跑版本

最稳的起跑点不是 `POI blend`，而是先用：

- `persons`: 现成 `phase2.5 households`
- `tract_od`: `homecenterfix` 版本的 enriched LODES tract OD
- `validation`: 现成 mobility-anchor 脚本

原因：

- `POI` 会引入另一条 destination-side 信息源
- 首轮目标是判断 mobility 是否在 `gravity + LODES + context` 之外有独立贡献
- 因此第一轮应尽量控制变量，先不混入 `POI` 解释

### 推荐命令

在 WSA 上可直接这样跑：

```bash
bash tools/run_detroit_destination_baseline_v1.sh
MODE=distance_only bash tools/run_detroit_destination_baseline_v1.sh
MODE=gravity_lodes bash tools/run_detroit_destination_baseline_v1.sh
MODE=contextual bash tools/run_detroit_destination_baseline_v1.sh
MODE=mobility_m1 bash tools/run_detroit_destination_baseline_v1.sh
MODE=mobility_county_m2 bash tools/run_detroit_destination_baseline_v1.sh
MODE=mobility_center_m3 bash tools/run_detroit_destination_baseline_v1.sh
MODE=mobility_center_topk_m4 bash tools/run_detroit_destination_baseline_v1.sh
```

默认模式是 `contextual`。

---

## 5. 首轮判定标准

如果首轮结果满足下面任一条，就说明这条论文线值得继续：

- `B3` 相比 `B2` 有明显提升，说明 worker/home context 有必要单独建模
- `M1` 相比 `B3` 在 tract / OD / commute 中至少两层有净提升
- 即使 `M1` 只显著提升 commute / OD，也说明 mobility 不只是 destination mass proxy

反过来，如果 `M1` 只改善 tract ranking，不改善 OD / commute，就要收缩主张，把它写成 proof-of-signal，而不是主模型。

当前根据已完成结果，更稳的主推荐已经变成：

- `M2 county-state`

而当前最值得继续推进的增强版本是：

- `M4 center + top-K reranking`

---

## 6. 当前最需要补的输入

为了真正运行 `M1`，还缺一类中间产物：

- tract-level 或 tract-pair-level mobility feature table

也就是说，下一步最具体的工程任务不是再调 destination 权重，而是：

**把 WSA 上的 mobility 数据整理成 destination model 可直接消费的特征表。**

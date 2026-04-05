# Phase 3 Detroit Mobility-Anchor Validation

## 问题

本次实验回答一个更具体的问题：

- `place20190206.csv` 这类单日 GPS stay data，能否作为 Detroit `phase3` 空间结果的外部验证信号？
- 如果可以，它更适合验证 `home`，还是 `work`？

## 验证设定

- synthetic 输入：
  - Detroit metro `phase3` household-aware, zero-fallback run
  - run: `/home/jinlin/projects/Synthetic_City/outputs/_phase3_roadloc_detroitmsa_core_household_cons_works1100s1200_gapS1400_fixshort_20260329T132311Z`
- mobility 输入：
  - `/home/jinlin/data/Mobility_Data/place20190206.csv`
  - 单日 `2019-02-06` stay-event 数据
- 地理单元：
  - TIGER 2023 Michigan block groups
  - BG 聚合后再回到 tract

anchor 规则使用最保守的单日口径：

- `home anchor`：夜间停留且 `time_spent >= 6h`
- `work-like anchor`：白天停留且 `time_spent >= 3h`
- 若 `home-work distance < 500m`，则不保留 `work-like anchor`

## 核心结果

### 1. `home` 可以作为弱外部验证

Detroit study area 内：

- `672,868` 条 mobility stay events
- `416,591` 台设备
- `144,382` 台设备具备 `home anchor`

`home` 的 tract-level 对比已经相当强：

- tract share Spearman: `0.808`
- tract share cosine: `0.881`
- tract share TVD: `0.117`

更关键的是 tract 内 BG 排序：

- `1,684` 个 tract 中，`1,534` 个达到最小 anchor 覆盖阈值
- 其中 `1,516` 个 tract 可计算 BG-level Spearman
- mean BG Spearman: `0.353`
- median BG Spearman: `0.600`
- `62.9%` 的 tract 达到 `Spearman >= 0.5`

这说明单日 night-anchor 虽然不是点级真值，但已经足以作为 **fine-scale residential pattern** 的弱外部验证。

### 2. `work` 当前没有被验证通过，而且暴露的是方法边界，不是点位噪声

`work-like anchor` 覆盖显著更低：

- `44,103` 台设备有白天长停留候选
- 但同时满足 `home + work-like` 的只有 `7,628` 台

`work` 指标明显不成立：

- tract share Spearman: `0.195`
- tract share cosine: `0.570`
- tract share TVD: `0.399`
- OD share Spearman: `-0.778`
- OD share cosine: `0.129`
- OD share TVD: `0.929`

最关键的解释性数字是：

- synthetic `same-tract work share = 0.99`
- mobility `same-tract work share = 0.0709`

这说明当前 `phase3 work` 的主要问题不是：

- road support 不够
- 点几何不合法
- mobility anchor 太噪

而是更根本的结构问题：

**当前 phase3 只是在 home tract 内放置了 work-support points，并没有显式建模 `work destination tract`。**

因此，mobility 对 `work` 的验证失败，本质上是在暴露方法上限，而不是在否定当前点位生成器本身。

## 结论

这次验证把 `home` 和 `work` 的结论彻底分开了：

- `home`：
  - 已经有可发表的弱外部空间验证
  - 更准确地说，是对 tract 内 residential fine-scale distribution 的验证
- `work`：
  - 当前不能宣称通过了 mobility-based spatial validation
  - 问题不在 `phase3` 的 road placement，而在更上游缺少 `destination tract` 建模

因此，这次实验真正回答的问题不是“图好不好看”，而是：

**单日 mobility 对 residential pattern 有验证价值；对 work，它首先揭示的是我们当前 pipeline 仍然是 `within-home-tract work placement`，而不是完整的 commuting model。**

## 产物

- local sync:
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/run_summary.json`
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/metrics/home_bg_spearman_by_tract.csv`
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/metrics/home_tract_comparison.csv`
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/metrics/work_tract_comparison.csv`
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/metrics/work_od_comparison.csv`
  - `outputs/_sync_phase3_validate_mobility_anchor_detroit_20260329/metrics/commute_distance_bins.csv`

## 下一步

如果要继续把 `work` 验证做成立，最合理的顺序是：

1. 先在 `phase2/2.5` 之上显式生成 `home tract -> work tract` destination
2. 再把 `phase3` 限定为 `destination tract 内的 explicit work-point placement`
3. 最后再用 mobility anchor 或 LODES 做 OD / commute validation

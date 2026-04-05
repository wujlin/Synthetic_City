# phase3 结果记录：work support 的 arterial-missing 显式例外

> 日期：2026-03-29  
> 范围：Michigan 10-PUMA pilot  
> 目标：判断 `work` 剩余缺口到底是分配器问题，还是 support definition 问题

---

## 1. 核心发现

`work` 的主问题不是分配器，也不是几何合法性，而是 `S1100 + S1200` 在一批 tract 上**根本没有支持集**。

对推荐配置下剩余 `55` 个 `no-candidate` tract 做诊断后，发现：

- `55/55` 个 tract 的 `S1100 + S1200 = 0`
- `55/55` 个 tract 都有 `S1400`
- 这 `55` 个 tract 恰好对应全部 `91,180` 个未分配 worker

这说明问题结构非常明确：

> `work` 的缺口来自 support-definition mismatch，而不是 allocator failure。

---

## 2. 方法改动

没有使用 runtime fallback。  
改动的是一条**预声明的显式规则**：

- 常规 tract：`work = S1100 + S1200`
- `arterial-missing` tract：若 tract 内 `S1100 + S1200 = 0`，则改用 `S1400`

实现上，这个阶段在候选点表里被单独标记为：

- `source_stage = arterial_missing_exception`

并且 validator 明确将它视为：

- 合法的显式 exception stage
- 而不是 black-box fallback

---

## 3. 结果

运行目录：

- `/home/jinlin/projects/Synthetic_City/outputs/_phase3_roadloc_10puma_cons_nofb_works1100s1200_gapS1400_legalfix2_20260329T042726Z`

同步到本地的轻量结果：

- [run_summary_gapS1400.json](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_sync_phase3_roadloc_gapS1400_20260329/run_summary_gapS1400.json)
- [summary_gapS1400.json](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_sync_phase3_roadloc_gapS1400_20260329/summary_gapS1400.json)
- [roadloc_validation_gapS1400.json](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_sync_phase3_roadloc_gapS1400_20260329/roadloc_validation_gapS1400.json)
- [group_diagnostics_gapS1400.csv](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_sync_phase3_roadloc_gapS1400_20260329/group_diagnostics_gapS1400.csv)
- [no_candidate_groups_gapS1400.csv](/Users/jinlin/Desktop/Project/Synthetic_City/outputs/_sync_phase3_roadloc_gapS1400_20260329/no_candidate_groups_gapS1400.csv)

关键数字：

- `work_stage_counts`：`370 primary + 55 arterial_missing_exception`
- `work_assigned = 667,888`
- `work_unassigned = 0`
- `work_fallback_assignments = 0`
- `work_zero_fallback_success = true`
- `overall_zero_fallback_success = true`

同时，work 候选点几何合法率没有变差，反而略升：

- 旧推荐配置：`0.99222`
- gap-exception 配置：`0.99424`

---

## 4. 含义

这轮结果说明两件事。

第一，`work` 现在已经不再是“未解决状态”。  
在当前 tested scope 下，我们已经拿到了一条 **zero-fallback、可解释、可验证** 的完整显式解。

第二，`S1400` 不能被简单表述成“work road”。  
更准确的说法是：

> 当 tract 内不存在任何 arterial-style work support 时，`S1400` 被用作 local-access work-support proxy。

这让 `home/work` 的语义在那 `55` 个 tract 上变弱，但这种变弱是：

- 局部的
- 预声明的
- 可审查的

因此它与 runtime fallback 有本质区别。

---

## 5. 当前推荐口径

- `home`: `S1400 only`
- `work`: `S1100 + S1200`
- 例外：若 tract 内 `S1100 + S1200 = 0`，则 `work = S1400`
- 保持 `zero-fallback`

一句话总结：

> `work` 的缺口已经被从“黑箱 fallback”重构成“显式 support exception”，因此现在的 phase3 已经可以作为完整的 road-constrained baseline 使用。

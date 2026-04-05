# Phase-2 Solver Pilot Note (2026-03-28)

> 角色：二期 `PUMA -> small area` 求解器小规模 pilot 记录  
> 问题：当前 `M_{p,k,g}` 分解器在 tract 级是否可收敛？`mobility home-origin mass` 是否能改进 tract 结构？

## 1. 这轮真正发现了什么

这轮最关键的发现不是某个超参更好，而是：

> **不先协调 tract targets 与 PUMA joint，总体约束系统本身就是不可行的。**

在 unreconciled 版本里，`AGEP_bin` 和 `SEX` 的 tract marginals 可以被压得很准，但 `row conservation`
始终收不拢。`10/10` 个 PUMA 在 `200` 次迭代内都未收敛，说明问题不在“算得不够久”，而在
`small-area ACS targets` 与 `PUMA-level full joint` 本身不完全相容。

因此这一步的正确修正不是继续加迭代，而是：

1. 先在 region 内把 hard targets 协调到与 `type-count` 同一个可行域  
2. 再做 `type -> tract` 的 IPF-style 分解

这一步完成后，`10/10` 个 PUMA 全部收敛，`row_sum` 和 hard marginals 都进入数值精度范围。

## 2. Reconciliation 后的结果

WSA run:

- baseline: `outputs/_phase2_p2t_10puma_recon_20260328T100817Z`
- mobility mass: `outputs/_phase2_p2t_10puma_recon_mobmass_20260328T100838Z`

同步回本地的 summary:

- `outputs/_sync_phase2_p2t_10puma_20260328/run_summary_recon.json`
- `outputs/_sync_phase2_p2t_10puma_20260328/metrics_recon.json`
- `outputs/_sync_phase2_p2t_10puma_20260328/run_summary_recon_mobmass.json`
- `outputs/_sync_phase2_p2t_10puma_20260328/metrics_recon_mobmass.json`

核心结果：

- baseline recon:
  - `10/10` PUMA 收敛
  - `row_sum max_rel` 平均约 `5.46e-10`
  - `AGEP_bin max_rel` 平均约 `8.17e-11`
  - `SEX max_rel` 平均约 `4.02e-16`
- tract-level raw-target fit:
  - `AGEP_bin mean TVD ≈ 0.00964`
  - `SEX mean TVD ≈ 0.00152`
  - `SCHL_allpop mean TVD ≈ 0.01772`
  - `ESR_allpop mean TVD ≈ 0.03854`
  - `EARN_16p_bin mean TVD ≈ 0.04169`

这说明：

> **solver 本身已经稳定；剩余误差主要不是约束求解失败，而是 tract-level soft prior 本身的信息边界。**

## 3. mobility mass 的结论

这轮 mobility 只以 `home-origin share` 的形式进入 `group weight`：

- `weighted_mean_entropy`: `5.7146 -> 5.7167`
- `mean_max_group_share`: `0.05266 -> 0.05099`
- `corr_vs_group_weights ≈ 0.661`

但它没有改进 tract-level held-out structure：

- `SCHL_allpop mean TVD`: `0.01772 -> 0.01940`
- `ESR_allpop mean TVD`: `0.03854 -> 0.04251`
- `EARN_16p_bin mean TVD`: `0.04169 -> 0.04187`

因此当前可以明确写成：

> **type-independent mobility mass prior 只会改变“人往哪些 tract 更分散地流”，但不会自动提升
> `SCHL / ESR / EARN` 的 tract 结构恢复。**

## 4. 下一步该做什么

下一步不应该继续在 `group mass weight` 上打转，而应该推进到：

1. `type-conditioned mobility prior`
   - 让 `q(g|k)` 不再只是 tract baseline mass
   - 而是让某些 `k` 与某些 tract mobility profile 产生可学习兼容性
2. 区分 `hard feasibility` 和 `behavior prior`
   - feasibility 已由当前 solver 解决
   - mobility 的任务应该是提供额外空间异质性，而不是替代 ACS

一句话总结：

> 二期现在已经有一个**可收敛、可运行、可验证**的 tract-level `PUMA -> small area` 基线；
> 下一阶段的问题不再是“怎么让 solver 跑起来”，而是“怎么让 mobility 进入 `q(g|k)` 时真的携带类型信息”。

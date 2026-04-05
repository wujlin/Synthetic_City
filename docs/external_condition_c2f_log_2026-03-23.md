# External Condition 与 Coarse-to-Fine 实验记录

日期：2026-03-23  
状态：进行中  
负责人：Codex + wsA 后台实验

## 1. 这轮实验回答的问题

这轮实验的核心问题不是“如何把一个更复杂的工程系统跑起来”，而是两个更本质的科学问题：

1. 当 `condition` 由真实外部 census aggregate summaries 主导时，distribution-level diffusion 还能否恢复区域级 joint distribution？
2. 当 full joint 的细粒度状态空间变大时，失败到底来自 external condition 不足，还是来自多轴细化后的表示与误差传播问题？

因此，这轮实验分成三步：

- `external-condition v1`：建立 condition-led 口径
- `refinement ablation`：定位失败发生在哪种细化上
- `coarse-to-fine`：验证“先学 coarse，再学 fine”是否真能解决问题

## 2. 数据与 schema

### 2.1 full external v1

- condition：ACS 2022 external summaries
- target：PUMS 2023 按同一 schema 重聚合
- schema：`AGEP_bin(10) × SEX(2) × SCHL_allpop(5) × ESR_allpop(5)`
- `K = 500`

变量级 condition/target 对齐良好：

- `AGEP_bin mean TVD ≈ 0.0132`
- `SEX mean TVD ≈ 0.0028`
- `SCHL_allpop mean TVD ≈ 0.0144`
- `ESR_allpop mean TVD ≈ 0.0089`

因此，`K=500` 的失败不能简单归因为 condition 本身对不齐。

### 2.2 lite schema

- schema：`AGEP_lite(4) × SEX(2) × SCHL_lite(3) × ESR_lite(3)`
- `K = 72`

这是当前 coarse stage 成功的基础版本。

## 3. 已完成实验

### 3.1 external-condition v1-lite

关键 run：

- `outputs/_us_puma_external_v1_lite_retry_20260323T075526Z`
- 多 seed 汇总：
  - `outputs/_us_puma_external_v1_lite_multiseed_20260323T0804Z/summary.json`

结果：

- diffusion `tvd_joint = 0.04224 ± 0.00054`
- IPF baseline `0.04944`
- 相对提升约 `14.56%`

结论：

- external condition 不是无效信号
- 在 coarse schema 下，diffusion 可以稳定超过 IPF

### 3.2 refinement ablation

| Variant | K | Diffusion TVD | IPF TVD | 相对结果 |
|---|---:|---:|---:|---|
| lite | 72 | 0.0422 | 0.0494 | 优于 IPF |
| age_refine | 180 | 0.0557 | 0.0595 | 优于 IPF |
| schl_refine | 120 | 0.0533 | 0.0592 | 优于 IPF |
| esr_refine | 120 | 0.0496 | 0.0529 | 优于 IPF |
| age_schl_refine | 300 | 0.0776 | 0.0748 | 开始输给 IPF |
| full | 500 | 0.1188 | 0.0809 | 明显输给 IPF |

关键 run：

- `outputs/_us_puma_external_v1_age_refine_20260323T083911Z`
- `outputs/_us_puma_external_v1_schl_refine_20260323T084109Z`
- `outputs/_us_puma_external_v1_esr_refine_20260323T084240Z`
- `outputs/_us_puma_external_v1_age_schl_refine_20260323T084421Z`
- `outputs/_us_puma_external_v1_diffusion_20260323T071929Z`

结论：

- 单轴细化并不会立刻击穿模型
- 真正的困难出现在 **多轴细化叠加**
- 当前最关键的组合是 `AGEP + SCHL`

这说明问题不是“某一个变量太难”，而是 fine-grained cross-axis dependence 开始主导误差。

### 3.3 teacher-forced stage-2

目标：

- 固定 subgroup：`(SEX, ESR_lite)`
- 条件：该 subgroup 内的 coarse `AGEP_lite × SCHL_lite` table
- target：该 subgroup 内的 fine `AGEP_fine × SCHL_fine` conditional table

关键 run：

- `outputs/_us_puma_external_c2f_age_schl_teacher_20260323T103646Z`

结果：

- `tvd_joint_raw = 0.0737`
- `tvd_joint_projected = 0.0620`
- `uniform_parent baseline = 0.2931`

结论：

- 第二阶段 refinement 本身是可学的
- 问题不在于 “fine age × education 永远学不出来”

### 3.4 end-to-end 两阶段组合（K=300）

组合方式：

- Stage 1：`v1-lite` diffusion + external IPF
- Stage 2：teacher-forced 学到的 refinement model
- 评估目标：`AGEP_fine(10) × SEX(2) × SCHL_fine(5) × ESR_lite(3)`，即 `K=300`

关键 run：

- `outputs/_us_puma_external_c2f_age_schl_eval_20260323T104833Z`

结果：

- `coarse_to_fine tvd_joint = 0.0896`
- `uniform_refine_baseline = 0.3131`
- 对照 one-shot `age_schl_refine`：
  - diffusion `0.0776`
  - IPF `0.0748`

结论：

- 两阶段绝不是“没学到东西”
- 它远好于 uniform refine，说明 coarse-to-fine 的第二阶段是有效的
- 但 **当前这版“独立训练的 stage-2 + stage-1 预测直接输入”的简单组合，还没有超过 one-shot，更没有超过 IPF**

### 3.5 oracle stage-2（误差分解）

目标：

- 保持 second-stage refinement model 不变
- 但不给它 stage-1 的预测 coarse parent table
- 改为直接提供 **真实 coarse parent table**

这样得到的结果不再回答“当前两阶段方法整体有多好”，而是专门回答：

> 如果 coarse-to-fine 的第二阶段吃到的是无误差 coarse 结构，那么它本身能把 `K=300` 做到什么程度？

关键 run：

- `outputs/_us_puma_external_c2f_age_schl_oracle_20260323T110315Z`

结果：

- `oracle stage-2 tvd_joint = 0.0693`
- 对照 naive 两阶段：
  - `coarse_to_fine tvd_joint = 0.0896`
- 对照 one-shot `age_schl_refine`：
  - diffusion `0.0776`
  - IPF `0.0748`

结论：

- 一旦去掉 stage-1 传播误差，第二阶段 refinement 立刻明显改善
- `0.0693 < 0.0776`，说明在 `K=300` 上，**oracle-guided stage-2 已经优于当前 one-shot diffusion**
- `0.0693 < 0.0748`，说明它也已经优于当前 IPF baseline

这说明：

- 当前 coarse-to-fine 失败的主要原因不是 second-stage learnability 不足
- 也不是 “fine age × education 永远学不出来”
- 真正的主要损失来自 **stage-1 coarse prediction 如何传递到 stage-2**

因此，当前问题最自然的科学表述不是“模型还需要再调”，而是：

> external condition 已经足以支撑 coarse structure，fine refinement 本身也已经可学；当前真正的瓶颈是多尺度生成中的 stage coupling 与 error propagation。

### 3.6 exposure-matched stage-2（teacher-forcing 偏移检验）

目标：

- 不再用真实 coarse parent table 训练 stage-2
- 改为先运行 stage-1 lite diffusion，再把它的 projected coarse parent table 写入 stage-2 训练集
- 也就是说，让 stage-2 在训练时直接看到推理阶段会遇到的 coarse 条件分布

这一步专门回答：

> 当前 naive 两阶段的劣势，是否主要来自 teacher-forced 训练分布与推理分布之间的 covariate shift？

关键 run：

- stage-2 训练：
  - `outputs/_us_puma_external_c2f_age_schl_exposure_20260323T111554Z`
- end-to-end 评估：
  - `outputs/_us_puma_external_c2f_age_schl_exposure_eval_20260323T112349Z`

数据构建 metadata：

- `mean_parent_tvd_pred_to_true = 0.0343`
- `mean_parent_mass_abs_err = 0.0053`

结果：

- 原 naive coarse-to-fine：
  - `tvd_joint = 0.0896`
- exposure-matched coarse-to-fine：
  - `tvd_joint = 0.0886`
- oracle stage-2：
  - `tvd_joint = 0.0684`

结论：

- 让 stage-2 直接接触 predicted parent 确实带来了一点改善，但幅度很小
- `0.0896 -> 0.0886` 说明 teacher-forcing 分布偏移 **不是当前 gap 的主要来源**
- 因为如果问题主要来自 covariate shift，那么 exposure-matched 训练应该明显向 oracle `0.0684` 靠近；但现在它几乎没有缩小这个 gap

这意味着：

- 当前 coarse-to-fine 的主要难点不是“stage-2 没见过 predicted parent”
- 而更可能是：
  - stage-1 coarse prediction 本身的信息损失
  - coarse summary 对 fine age x education structure 的约束仍然不够
  - 或 stage-1 / stage-2 之间需要比 point-estimate parent table 更丰富的 coupling 变量

### 3.7 共享区域 latent 的联合层级模型

目标：

- 不再把 coarse 与 fine 写成两个独立模块
- 改为单个 encoder 从 external condition 中提取共享区域 latent `z`
- 同时训练：
  - `Coarse head(z) -> p_coarse`
  - `Fine head(z, p_coarse) -> p_fine`
- 并加入 `aggregate(fine) ~= coarse` 的一致性约束

这个模型直接回答：

> 如果当前问题的根源是 coarse-to-fine 接口的信息压缩，那么让 coarse 与 fine 共享同一个区域 latent，是否能显著恢复之前丢失的信息？

关键 run：

- `outputs/_us_puma_external_joint_hier_age_schl_20260323T120228Z`
- 多 seed + 结构 ablation 汇总：
  - `outputs/_us_puma_external_joint_hier_age_schl_batch_summary/summary.json`

结果：

- hierarchical joint `tvd_joint_raw = 0.0667`
- hierarchical joint `tvd_joint = 0.0663`
- coarse head `tvd_coarse_head = 0.0396`
- aggregated fine `tvd_coarse_from_fine = 0.0392`

对照：

- one-shot diffusion (`age_schl_refine`): `0.0776`
- IPF baseline: `0.0748`
- naive coarse-to-fine: `0.0896`
- oracle stage-2: `0.0684`

结论：

- 共享区域 latent 的联合层级模型，已经明显优于：
  - one-shot diffusion
  - IPF
  - naive coarse-to-fine
- 它甚至略优于 oracle stage-2（`0.0663 < 0.0684`）

这说明：

- 当前问题不是 “coarse-to-fine 这个分解本身错了”
- 真正有问题的是：
  - 把 coarse state 压缩成独立的 point-estimate parent table
  - 再把它单向传给 fine stage
- 一旦 coarse 与 fine 通过共享区域 latent 联合建模，模型就能显著恢复之前丢失的信息

### 3.8 联合层级模型的多 seed 与结构 ablation

目标：

- 验证共享 latent 路线是否只是在单 seed 上偶然有效
- 判断真正起作用的是：
  - 共享区域 latent `z`
  - 显式 coarse probability table
  - 还是某种更“learned” 的 coarse latent

实验设置：

- 默认结构：`fine_head(z, coarse_prob)`，记作 `z_coarse_prob`
- 对照 1：`fine_head(z)`，记作 `z_only`
- 对照 2：`fine_head(z, coarse_latent)`，记作 `z_coarse_latent`

关键 run：

- `outputs/_us_puma_external_joint_hier_age_schl_batch_seed0`
- `outputs/_us_puma_external_joint_hier_age_schl_batch_seed1`
- `outputs/_us_puma_external_joint_hier_age_schl_batch_seed2`
- `outputs/_us_puma_external_joint_hier_age_schl_batch_zonly_seed0`
- `outputs/_us_puma_external_joint_hier_age_schl_batch_zlatent_seed0`
- 汇总：
  - `outputs/_us_puma_external_joint_hier_age_schl_batch_summary/summary.json`

结果：

- `z_coarse_prob`，3 seeds：
  - `tvd_joint = 0.06615 ± 0.00015`
  - 相对 IPF 提升：`11.58% ± 0.20%`
- `z_only`，seed 0：
  - `tvd_joint = 0.06617`
- `z_coarse_latent`，seed 0：
  - `tvd_joint = 0.06730`

结论：

- 共享 latent 联合层级模型的优势不是单 seed 偶然：
  - 三个 seed 都稳定优于 IPF
  - 三个 seed 之间波动极小
- 结构 ablation 显示：
  - `z_only` 与默认 `z_coarse_prob` 几乎相同
  - `z_coarse_latent` 反而略差

这说明当前最关键的信息通道不是：

- 显式 coarse probability table 本身
- 也不是额外引入的 learned coarse latent

而是：

> 共享 encoder 提取出来的区域级 latent `z`

换句话说，`AGEP × SCHL` 细粒度恢复真正依赖的是一个共享区域状态，而不是一个被压缩成 point-estimate coarse table 的中间对象。

### 3.9 shared latent 路线的 full `K=500` 扩展

目标：

- 不再停留在 `AGEP × SCHL` 的 `K=300` 机制配置
- 直接测试同一个共享区域 latent 原理，能否推广到 full external schema：
  - `AGEP_fine(10) × SEX(2) × SCHL_fine(5) × ESR_fine(5)`
  - `K = 500`

实现：

- 保持 coarse head 仍为 lite schema（`K=72`）
- fine head 直接输出 full `K=500` joint
- 仍使用共享 encoder 提取区域级 latent `z`
- 当前先测试最干净的通道：
  - `fine_head(z)`，即 `z_only`

关键 run：

- `outputs/_us_puma_external_joint_hier_full_z_only_20260324T021217Z`
- 多 seed 汇总：
  - `outputs/_us_puma_external_joint_hier_full_batch_summary/summary.json`

结果：

- `z_only`，3 seeds：
  - `tvd_joint_raw = 0.07288 ± 0.00003`
  - `tvd_joint = 0.07248 ± 0.00005`
  - `tvd_coarse_head = 0.03859 ± 0.00014`
  - `tvd_coarse_from_fine = 0.03839 ± 0.00008`

对照：

- full one-shot diffusion：`0.1188`
- full IPF baseline：`0.0809`

结论：

- 共享区域 latent 的作用并不局限于 `K=300`
- 在 full `K=500` 上，它同样显著优于：
  - one-shot diffusion
  - IPF baseline
- 相对 full IPF 的提升为 `10.39% ± 0.06%`
- `raw` 与 post-IPF 几乎一致，说明提升主要来自模型本身学到的 joint structure，而不是后处理校准
- 三个 seeds 的波动极小，说明这不是 full setting 下的单 seed 偶然结果

这一步非常关键，因为它说明：

- `K=300` 不是唯一有效的 lucky setting
- 共享区域 latent 的收益可以直接推广到 full schema
- 因而当前主问题不应再表述为“为什么只选 `AGEP × SCHL`”，而应表述为：

> shared regional context 是否是跨尺度 joint recovery 所需的核心状态表示？

## 4. 当前的科学判断

现在最重要的判断是：

1. `external condition` 是有效信号，不是这轮失败的根源。
2. `K=500` 的失败也不能简单解释为“模型容量不够”。
3. 真正的问题是：
   - 单轴 refinement 可学
   - `AGEP + SCHL` 联合细化时，误差开始由多轴细粒度结构主导
   - naive 两阶段拼接会引入明显的 **stage-1 error propagation**
   - oracle stage-2 已经表明 second-stage refinement 本身并不是当前主瓶颈
   - exposure-matched stage-2 只带来极小改善，说明单纯的 teacher-forcing / covariate shift 也不是主瓶颈
   - 共享区域 latent 的联合层级模型在 `K=300` 和 `K=500` 上都显著改善，说明主问题是 coarse-to-fine 接口的信息压缩方式
   - 多 seed 与结构 ablation 进一步表明，真正需要保留的主要不是显式 coarse table，而是共享区域 latent state

因此，当前问题的本质更像：

> 我们已经证明 fine refinement 本身是可学的；当前真正的瓶颈不是 second-stage 能力，而是 **跨尺度接口该保留什么状态信息**。现有证据最支持的答案是：需要保留的是共享区域 context，而不仅是局部 coarse summary。

这比“再调大 hidden dims / 再加 epochs”更接近问题本身。

## 4.1 收入类变量的最新推进：B20001 可以先作为 earnings proxy 接入

为了判断“收入”能否自然接入当前 external-condition 主线，这一轮没有直接把 `PINCP` 硬塞回 full joint，而是先单独验证一条更干净的 proxy 变量：

- external 条件：`B20001 -> EARN_16p_bin`
- target：`PUMS PERNP -> EARN_16p_bin`

其中 `EARN_16p_bin` 被定义成一个 all-population 变量：

- `not_in_earnings_universe`
- `lt_25k`
- `25k_50k`
- `50k_75k`
- `75k_100k`
- `ge_100k`

构造方式是：

- `not_in_earnings_universe = total population - population 16+ with earnings`
- 其余五档来自 `B20001` 的 male/female earnings bins 合并后再聚合

在 wsA 上用 Michigan 68 个 PUMA 做完正式构建后，`B20001` 与 `PUMS PERNP` 的 PUMA-level 对齐结果是：

- mean TVD = `0.01695`
- max TVD = `0.03057`
- `n_groups = 68`

这说明：

- `B20001` 不能被表述成 `PINCP` 的严格同义物
- 但它可以被稳定地表述成一个 **earnings proxy**
- 并且这个 proxy 在 PUMA 层面已经足够干净，值得作为下一步收入类扩展的起点

因此，当前关于收入的最稳结论不是“收入不能做”，而是：

> 当前 external 端没有 clean 的 person-level income table，但 `B20001 -> EARN_16p_bin` 已经提供了一个对齐良好的 earnings proxy，可以作为收入类变量的第一步扩展。

## 5. 为什么这不是纯工程问题

这轮实验最需要避免的误判是：

> “two-stage 还没赢，只是实现没调好，所以继续工程优化。”

当前证据更支持一个结构性解释：

- `teacher-forced stage-2` 成功，说明 second-stage model 不是坏的
- `end-to-end coarse-to-fine` 失败，说明问题主要在 coarse prediction 如何约束 fine prediction
- 共享 latent 的联合层级模型在 `K=300` 与 `K=500` 都有效，说明有效的解决方式不是继续 patch two-stage，而是保留共享区域状态

也就是说，当前问题更接近：

- information flow
- multiscale dependence decomposition
- error propagation across scales

而不是：

- 再换一个优化器
- 再多跑一点 epoch
- 再加一层 MLP

因此，下一步实验应该继续服务这个科学问题，而不是退化成工程调参。

## 6. 下一步最值得做的诊断

### P0：把共享 latent 路线正式定为主线

当前最值得推进的，不是继续在 two-stage patching 上堆 trick，而是把共享 latent 路线正式定为主线设计：

- 保留共享 encoder
- 保留 coarse/fine 联合训练
- 当前 evidence 最支持：
  - `z` 是主要有效状态
  - `p_coarse` 的显式输入不是决定性贡献
  - 这一点在 `K=300` 与 full `K=500` 上都成立

因此，后续更值得回答的问题是：

- 怎样让 `z` 更明确地承担跨尺度状态表示
- 怎样把这个 insight 写成方法贡献，而不是只写成一个更好的网络结构

## 7. 相关代码与运行入口

本地新增脚本：

- `tools/build_external_c2f_age_schl_teacher.py`
- `tools/train_external_c2f_age_schl_teacher.py`
- `tools/run_external_c2f_age_schl_teacher.sh`
- `tools/eval_external_c2f_age_schl_pipeline.py`
- `tools/run_external_c2f_age_schl_pipeline.sh`
- `tools/build_external_c2f_age_schl_exposure.py`
- `tools/run_external_c2f_age_schl_exposure.sh`
- `tools/train_external_joint_hier_age_schl.py`
- `tools/run_external_joint_hier_age_schl.sh`
- `tools/run_external_joint_hier_age_schl_batch.sh`
- `tools/summarize_external_joint_hier_runs.py`
- `tools/train_external_joint_hier_full.py`
- `tools/run_external_joint_hier_full.sh`
- `tools/external_earn_v1_schema.py`
- `tools/build_external_condition_earn_v1_michigan.py`
- `tools/build_external_target_earn_v1_michigan.py`
- `tools/run_external_earn_v1_michigan.sh`
- `tools/build_external_target_earn_v1_us.py`
- `tools/train_external_earn_from_context.py`
- `tools/run_external_earn_from_context.sh`
- `tools/run_external_earn_from_context_batch.sh`
- `tools/summarize_external_earn_from_context_runs.py`
- `tools/build_external_condition_earn_v1_acs_puma.py`
- `tools/merge_external_condition_v1_with_earn.py`
- `tools/train_external_joint_hier_full_earn_aux.py`
- `tools/run_external_joint_hier_full_earn_aux.sh`

wsA 后台窗口：

- `synthetic-73:c2f_age_schl`
- `synthetic-73:c2f_eval_retry`
- `synthetic-73:c2f_exposure_train`
- `synthetic-73:c2f_exposure_eval`
- `synthetic-73:joint_hier_train`
- `synthetic-73:joint_hier_batch`
- `synthetic-73:joint_hier_full500`
- `synthetic-73:earn_v1_build`
- `synthetic-73:earn_ctx`
- `synthetic-73:earn_ctx_batch`
- `synthetic-73:earn_full_aux`

## 8. 当前结论（一句话）

> external condition 是成立的，fine refinement 本身也是可学的；当前真正的瓶颈不是“模型完全学不会”，而是 **多轴细化后的跨尺度状态表示**。现有证据表明，必须保留的核心信息主要是共享区域 latent，而不是 point-estimate coarse table，而且这一点已经从 `K=300` 推广到了 full `K=500`。

## 9. 收入类变量的最新推进：`B20001 -> EARN_16p_bin`

这一步解决的问题不是“把 `PINCP` 直接塞回外部条件”，而是先回答一个更稳的问题：

> 当前 external 条件下，是否存在一个定义清楚、可对齐、可训练的收入类 proxy？

当前采用的定义是：

- external 端：`ACS B20001`
- target 端：`PUMS PERNP`
- 变量名：`EARN_16p_bin`

其中：

- `B19001` 是 household income，不适合直接接当前 person-level 主线
- `B20001` 是 earnings，不是 `PINCP`
- 因此这一条线的正确解释是 earnings proxy，而不是 personal income 的严格替代

### 9.1 Michigan 对齐检查

Michigan PUMA 级 condition / target 对齐已经单独跑通：

- wsA run：
  - `outputs/_earn_v1_build_mi_puma_20260324T043118Z`
- 关键产物：
  - `extcond_earn_v1_acs5_2022_puma_state26_michigan.csv`
  - `exttarget_earn_v1_pums_2023_puma_state26_michigan.csv`
  - `exttarget_earn_v1_pums_2023_puma_state26_michigan.metadata.json`

对齐结果：

- `mean TVD = 0.01695`
- `max TVD = 0.03057`
- `n_groups = 68`

这说明：

- `B20001 -> EARN_16p_bin`
  已经足够干净，可以作为外部 earnings proxy 使用
- 但它仍然应被解释为 earnings-universe variable，而不是 `PINCP`

### 9.2 earnings-from-context：当前 4 属性 regional context 是否携带收入类信息

在 Michigan 对齐站住后，下一步做的是一个更直接的问题：

> 由当前 4 属性 external condition 编码出的 regional context，能否在不直接观测 earnings 的前提下，预测 `EARN_16p_bin`？

为此新增了：

- `tools/build_external_target_earn_v1_us.py`
- `tools/train_external_earn_from_context.py`
- `tools/run_external_earn_from_context.sh`

这个实验的设置是：

- 输入：
  - 现有 nationwide full external condition  
    `AGEP_bin + SEX + SCHL_allpop + ESR_allpop`
- 目标：
  - nationwide PUMS-derived `EARN_16p_bin`
- split：
  - non-Michigan train
  - Michigan holdout test
- 模型：
  - `encoder(condition) -> regional latent z -> earnings head`
- baseline：
  - training-set mean earnings distribution

exploratory single-seed wsA run：

- `outputs/_us_puma_external_earn_from_context_20260324T055113Z`

对应的正式 multi-seed batch：

- seed runs：
  - `outputs/_us_puma_external_earn_from_context_batch_seed0`
  - `outputs/_us_puma_external_earn_from_context_batch_seed1`
  - `outputs/_us_puma_external_earn_from_context_batch_seed2`
- summary：
  - `outputs/_us_puma_external_earn_from_context_batch_summary/summary.json`

正式 3-seed 结果：

- model `tvd_earn = 0.03102 ± 0.00030`
- baseline `train_mean_earn = 0.07231`
- 相对下降 `57.10% ± 0.42%`

另外：

- `cosine_earn = 0.99799 ± 0.00012`
- `mae_earn = 0.01034 ± 0.00010`
- `n_test_mi = 68`

### 9.3 这一步意味着什么

这条结果说明：

- 当前 4 属性 external condition 不只足以恢复人口结构本身
- 它还携带了明显的收入类区域上下文信息
- 即使不直接把 earnings 作为输入，shared regional context 仍然能够外推出一个稳定的 earnings proxy

更重要的是，这一步没有把问题重新写成一个更大的 joint simplex，而是回答了一个更干净的问题：

> shared regional context 是否已经编码了收入相关的地区差异？

当前证据支持：

- **是**
- 而且这个信号强度明显高于简单的 training-set mean baseline

## 10. 5-condition 合并试验：在主 full joint 中接入 earnings

这一步的目标不是直接把输出扩成 `500 x 6 = 3000` cell 的更大 joint，而是一个更稳的 first merge：

- condition 增加 `EARN_16p_bin`
- 主输出仍然保持 4 属性 full joint
- 额外增加一个 `earnings auxiliary head`

这样做的 rationale 很明确：

- 先验证 earnings condition 是否能改善 shared `regional context`
- 同时让输出里确实包含一个收入相关的新量
- 但不立刻把问题推成更稀疏的 3000-cell joint

### 10.1 新增的数据与训练入口

新增脚本：

- `tools/build_external_condition_earn_v1_acs_puma.py`
- `tools/merge_external_condition_v1_with_earn.py`
- `tools/train_external_joint_hier_full_earn_aux.py`
- `tools/run_external_joint_hier_full_earn_aux.sh`

这条线的输入和输出是：

- 输入 condition：
  - 原 `external_condition_v1`
  - 加上 `EARN_16p_bin`
- 主输出：
  - 现有 4 属性 full joint (`K=500`)
- 辅助输出：
  - `EARN_16p_bin`

### 10.2 exploratory single-seed 结果

wsA run：

- `outputs/_us_puma_external_joint_hier_full_earn_aux_z_only_20260324T072754Z`

condition 维度：

- 原 4-condition: `22`
- 合并后 5-condition: `28`

具体 block：

- `AGEP_bin = 10`
- `SEX = 2`
- `SCHL_allpop = 5`
- `ESR_allpop = 5`
- `EARN_16p_bin = 6`

single-seed 结果：

- 主 joint：
  - `tvd_joint = 0.07209`
- 对照：
  - 原 full shared-latent z-only multi-seed mean：`0.07248 ± 0.00005`
  - full IPF baseline：`0.08088`

- earnings auxiliary head：
  - `tvd_earn = 0.01627`
  - `cosine_earn = 0.99955`
  - `mae_earn = 0.00542`
  - baseline `train_mean_earn = 0.07231`
  - 相对 baseline 下降约 `77.49%`

### 10.3 当前可下的结论

这一步至少说明三件事：

1. 合并后的 5-condition 管线是自洽的
- nationwide `EARN_16p_bin` condition 可以从 ACS PUMA API 直接构建
- 与原 4-condition 合并后可直接进入 shared-latent full trainer

2. 主 full joint 没有被 earnings condition 破坏
- 当前 single-seed `0.07209`
- 至少没有出现明显回退
- 但是否真的优于原 4-condition full model，**还不能在 single-seed 上下结论**

3. earnings 头本身非常强
- `tvd_earn = 0.01627`
- 远优于 `train_mean_earn = 0.07231`
- 说明 merged condition 下的 shared regional context 已经能够稳定承载收入类 proxy

### 10.4 当前状态的正确定位

这条 5-condition 合并试验目前应被视为：

- **成功跑通**
- **结果正面**
- **但仍是 exploratory single-seed**

如果要进入主文或正式 claim，下一步必须补：

- `>= 3 seeds`
- 并直接比较：
  - 4-condition full shared-latent
  - 5-condition full shared-latent + earn aux

### 10.5 multi-seed 正式结果

为避免把 single-seed exploratory 结果误写成正式结论，这里补充 3-seed batch：

- batch launcher:
  - `tools/run_external_joint_hier_full_earn_aux_batch.sh`
- batch summary:
  - `tools/summarize_external_joint_hier_full_earn_aux_runs.py`

wsA batch run：

- `outputs/_us_puma_external_joint_hier_full_earn_aux_batch_20260324T074125Z_z_only_seed0`
- `outputs/_us_puma_external_joint_hier_full_earn_aux_batch_20260324T074125Z_z_only_seed1`
- `outputs/_us_puma_external_joint_hier_full_earn_aux_batch_20260324T074125Z_z_only_seed2`
- 汇总：
  - `outputs/_us_puma_external_joint_hier_full_earn_aux_batch_20260324T074125Z_summary/summary.json`

本地已同步：

- `outputs/_us_puma_external_joint_hier_full_earn_aux_batch_20260324T074125Z_summary.json`

3-seed 结果：

- 主 full joint：
  - `tvd_joint = 0.07258 ± 0.00041`
  - 对照 full IPF：
    - `0.08088`
  - 相对下降：
    - `10.27% ± 0.51%`

- earnings auxiliary head：
  - `tvd_earn = 0.01657 ± 0.00062`
  - `cosine_earn = 0.99952 ± 0.00005`
  - `mae_earn = 0.00552 ± 0.00021`
  - 对照 baseline `train_mean_earn = 0.07231`
  - 相对下降：
    - `77.09% ± 0.86%`

与原 4-condition full shared-latent 对照：

- 原 4-condition full shared-latent：
  - `tvd_joint = 0.07248 ± 0.00005`
- 当前 5-condition + earn aux：
  - `tvd_joint = 0.07258 ± 0.00041`

因此，这一步最稳的结论不是“earnings condition 明显改善了主 full joint”，而是：

1. 5-condition 合并不会破坏当前 full joint 主结果
2. 它让模型稳定获得了一个强的 earnings-related output
3. 当前收益主要体现在 **earnings auxiliary head**，而不是主 full joint 再次显著下降

所以从状态管理上，这条线现在已经从：

- exploratory single-seed

进入到：

- **artifact-complete multi-seed result**

但从论文叙事上，它更适合作为：

- `regional context` 可自然外推到收入类 proxy 的扩展证据

而不是当前 full joint 主结果的替代定义。

## 11. 条件收入分配：把 earnings 真正落到 4 属性个体上

前面的 earnings 扩展仍然有一个明显逻辑缺口：

- 它预测的是 **地区级 earnings 分布**
- 不是“给定某个人的年龄、性别、教育、就业后，这个人应该落在哪个 earnings 档”

因此，这一步改成更具体的目标：

> `p(EARN_16p_bin | AGEP_bin, SEX, SCHL_allpop, ESR_allpop, regional context)`

也就是说：

- 先保留当前已经稳定的 4 属性主 joint
- 再对每个 `PUMA × 4-attr cell` 学一个条件 earnings 分布

这样以后 materialize synthetic individuals 时，earnings 的分配就不再是“整区统一抽签”，而是：

- 先确定这个人属于哪个 4-attr cell
- 再从该 cell 在该地区对应的 earnings 条件分布中抽样

### 11.1 新增 target 与训练入口

新增脚本：

- `tools/build_external_target_earn_conditional_v1_us.py`
- `tools/train_external_earn_conditional_from_context.py`
- `tools/run_external_earn_conditional_from_context.sh`

新 target 的形式是：

- 每一行对应一个 `PUMA × 4-attr cell`
- 列出：
  - `AGEP_bin`
  - `SEX`
  - `SCHL_allpop`
  - `ESR_allpop`
  - 该 cell 的 mass
  - 该 cell 上的 `EARN_16p_bin` 条件分布

所以这里学习的对象不再是：

- `p(EARN | region)`

而是：

- `p(EARN | region, 4-attr cell)`

### 11.2 merged5 pilot 结果

wsA run：

- `outputs/_us_puma_external_earn_conditional_merged5_retry_20260324T114309Z`

本地已同步：

- `outputs/_us_puma_external_earn_conditional_merged5_retry_20260324T114309Z_run_summary.json`
- `outputs/_us_puma_external_earn_conditional_merged5_retry_20260324T114309Z_earn_conditional_summary.json`

输入：

- regional condition：merged 5-condition
  - `AGEP_bin`
  - `SEX`
  - `SCHL_allpop`
  - `ESR_allpop`
  - `EARN_16p_bin`
- person-side attributes：4-attr one-hot

数据规模：

- total conditional rows: `390,155`
- train rows: `379,457`
- Michigan test rows: `10,698`

pilot（single seed, `epochs=1500`）结果：

- 条件收入主指标（按 cell mass 加权）：
  - `weighted_tvd_earn = 0.09229`
  - `weighted_cosine_earn = 0.97147`
  - `weighted_mae_earn = 0.03076`

- 区域聚合后的 earnings 重构：
  - `aggregated_region_tvd_earn = 0.01873`

对照 baseline：

1. 完全不看 cell，只看训练集总体 earnings 均值：
   - `weighted_tvd_earn = 0.51451`
   - `aggregated_region_tvd_earn = 0.07218`

2. 只看 4-attr cell，不看地区 context 的 train-cell mean baseline：
   - `weighted_tvd_earn = 0.09458`
   - `aggregated_region_tvd_earn = 0.03689`

### 11.3 当前最稳的结论

这条 pilot 说明三件事：

1. 这条 conditional earnings 管线是可行的  
- 不需要再把问题写成 3000-cell 的 5-variable joint
- 但 earnings 已经可以真正依附到 4 属性个体上

2. 只靠 4-attr cell 的全局平均还不够  
- `0.09458 -> 0.09229`
- 虽然提升不大，但说明 `regional context` 不是冗余的

3. 在区域聚合层面，conditional model 的提升更明显  
- `0.03689 -> 0.01873`
- 说明这条模型确实在利用地区上下文，把 earnings 分配得更符合地区结构

因此，这一步解决的不是“再多预测一个地区级 histogram”，而是：

> 把 earnings 从地区级附加分布，推进成一个真正可用于 person-level assignment 的条件属性。

### 11.4 3-seed batch 结果

为避免把 single-seed pilot 误写成正式结论，这里补充 3-seed batch：

- wsA summary：
  - `outputs/_us_puma_external_earn_conditional_batch_20260324T115257Z_summary/summary.json`
- 本地已同步：
  - `outputs/_us_puma_external_earn_conditional_batch_20260324T115257Z_summary.json`

3 seeds（`0,1,2`）结果：

- 条件收入主指标：
  - `weighted_tvd_earn = 0.09288 ± 0.00048`
  - `weighted_cosine_earn = 0.97109 ± 0.00028`
  - `weighted_mae_earn = 0.03096 ± 0.00016`

- 区域聚合后的 earnings 重构：
  - `aggregated_region_tvd_earn = 0.01950 ± 0.00108`

对照 baseline：

1. 完全不看 cell，只看训练集总体 earnings 均值：
   - `weighted_tvd_earn = 0.51451`
   - `aggregated_region_tvd_earn = 0.07218`
   - 相对提升：
     - cell-level：`81.95% ± 0.09%`
     - region-level：`72.99% ± 1.49%`

2. 只看 4-attr cell，不看地区 context 的 train-cell mean baseline：
   - `weighted_tvd_earn = 0.09458`
   - `aggregated_region_tvd_earn = 0.03689`
   - 相对提升：
     - cell-level：`1.80% ± 0.51%`
     - region-level：`47.15% ± 2.92%`

这说明：

1. 这条 conditional earnings 管线在多 seed 下是稳定的  
- 它不只是 single-seed 偶然值

2. `regional context` 对个体条件收入分配的增益是存在的，但层次分明  
- 在 cell-level 条件分布上，增益相对 train-cell mean baseline 不大但稳定  
- 在 region-level 聚合结构上，增益明显更大

3. 因此，当前最合理的收入扩展不是 3000-cell 的 5-way full joint  
- 而是：
  - `4-attr joint`
  - `+ conditional earnings assignment`

### 11.5 当前状态定位

这条结果现在已经从：

- `single-seed pilot`

升级成了：

- **3-seed 稳定结果**

因此它已经足够支持下面这个更具体的说法：

> 当前最合理的收入扩展方式，是在已经稳定的 4 属性主 joint 之上，再学习 `p(EARN_16p_bin | AGEP_bin, SEX, SCHL_allpop, ESR_allpop, regional context)`，从而把 earnings 真正落到 synthetic persons 上。

但它还不支持下面这个更强的说法：

- “earnings 已经应该直接并入 5-way full joint，替代当前主产品定义”

所以这条线当前最适合的位置是：

- **正式的模型扩展结果**
- 不是新的主 full-joint 定义

## 12. 直接 5-way full joint：single-seed exploratory pilot

用户提出了一个更直接的问题：

> 既然现在已经有了 `EARN_16p_bin`，是否应该直接把它并入新的 5-way full joint，而不是继续走“4-attr joint + conditional earnings assignment”这条两段式路线？

这一步的 rationale 不是“它一定更好”，而是：

- 它是更直接的最终产品定义
- 如果 shared `regional context` 真能自然支撑收入轴，那么它应当至少在 5-way full joint 上给出一个可运行、可比较的结果

### 12.1 最小实现策略

这次没有再人为设计一个 earnings coarse hierarchy。  
相反，采用的是更干净的 diagnostic setting：

- fine target：
  - `AGEP_bin(10) × SEX(2) × SCHL_allpop(5) × ESR_allpop(5) × EARN_16p_bin(6)`
  - `K = 3000`
- coarse head：
  - 仍保持已经验证过的 4-attr lite schema
  - `AGEP_lite(4) × SEX(2) × SCHL_lite(3) × ESR_lite(3)`
  - `K = 72`
- earnings 轴在 coarse 侧直接被聚合掉

这样做的目的很明确：

- 先回答 `regional context` 是否能直接支撑完整 5 属性 joint
- 不把结果混入新的 hand-crafted earnings coarse 设计

### 12.2 新增数据与训练入口

新增脚本：

- `tools/build_external_target_v1_full_earn.py`
- `tools/train_external_joint_hier_full_earn.py`
- `tools/run_external_joint_hier_full_earn.sh`

其中：

- `build_external_target_v1_full_earn.py`
  - 不是重新发明 target
  - 而是把已经构造好的
    - `p(AGEP, SEX, SCHL, ESR | region)`
    - `p(EARN | AGEP, SEX, SCHL, ESR, region)`
  - 精确展开成：
    - `p(AGEP, SEX, SCHL, ESR, EARN | region)`

因此这条 5-way target 是：

- 与现有 conditional earnings target 同源
- 不依赖额外近似

### 12.3 exploratory pilot 结果

wsA run：

- `outputs/_us_puma_external_joint_hier_full_earn_z_only_20260324T121638Z`

本地已同步：

- `outputs/run_summary_full5_earn_20260324T121638Z.json`
- `outputs/hierarchical_summary_full5_earn_20260324T121638Z.json`

single-seed 结果：

- 5-way full joint raw：
  - `tvd_joint_raw = 0.12614`
- 5-way full joint（post-IPF）：
  - `tvd_joint = 0.12540`

对照：

- full 5-way IPF baseline：
  - `0.12698`
- independence：
  - `0.77473`

同时，coarse 72 侧仍然能被学到：

- `tvd_coarse_head = 0.03821`
- `tvd_coarse_from_fine = 0.03723`

### 12.4 当前解释

这轮结果说明三点：

1. 直接 5-way full joint **不是跑不动**  
- 它已经能稳定优于 independence
- 也略优于 full 5-way IPF baseline

2. 但它当前的收益 **很弱**
- `0.12698 -> 0.12540`
- 这说明 shared `regional context` 在 3000-cell 设定下并没有像 4-attr full joint 那样带来明显优势

3. 因此，这轮 pilot 目前更像一个**边界测试**
- 它证明了 5-way full joint 方向可以运行
- 但还没有证明它已经优于
  - `4-attr joint + conditional earnings assignment`
  这条当前更稳的产品路线

更具体地说：

- 当前 4-attr full shared-latent 的主结果在多 seed 下是：
  - `tvd_joint = 0.07248 ± 0.00005`
- 而 5-way full pilot 当前是：
  - `0.12540`

所以至少在现在这个阶段：

- 5-way full joint 还不能替代当前主产品定义
- 它更适合作为一个 exploratory extension

### 12.5 3-seed batch 结果

为避免把 single-seed exploratory 结果误读成稳定结论，这里继续补 3-seed batch：

- wsA summary：
  - `outputs/_us_puma_external_joint_hier_full_earn_batch_20260324T122734Z_summary/summary.json`
- 本地已同步：
  - `outputs/_us_puma_external_joint_hier_full_earn_batch_20260324T122734Z_summary.json`

3 seeds（`0,1,2`）结果：

- `tvd_joint_raw = 0.12630 ± 0.00051`
- `tvd_joint = 0.12544 ± 0.00033`
- `tvd_coarse_head = 0.03809 ± 0.00010`
- `tvd_coarse_from_fine = 0.03722 ± 0.00026`

对照：

- full 5-way IPF baseline：
  - `0.12698`
- relative gain vs IPF：
  - `1.21% ± 0.26%`

这说明：

1. single-seed 的结论不是偶然  
- 5-way full joint 的确能稳定略优于 full 5-way IPF

2. 但这个优势仍然很弱  
- 不是随机波动
- 也还没有达到“应该接管主产品定义”的强度

3. 所以当前更稳的判断是  
- `5-way full joint` 已经证明“可行”
- 但还没有证明“值得作为主路线替代 4+1 方案”

### 12.6 当前状态定位

这条线现在已经从：

- `single-seed exploratory pilot`

升级成了：

- **3-seed 稳定 exploratory extension**

它已经回答了一个问题：

- 直接 5-way full joint 值得不值得尝试？  
  - 值得，因为它不是不可训练的死路，而且在多 seed 下能稳定略优于 5-way IPF

但它给出的答案仍然是：

- 它是否已经优于当前的
  - `4-attr joint + conditional earnings assignment`
  - 并因此应当接管主产品定义？  
  - **目前答案仍然是否定的**

## 13. Shared-Latent Hierarchical Diffusion Pilot

### 13.1 目的

前面的 shared-latent 机制实验帮助我们识别出：

- `regional context`
- `group-level latent variable`

是高维外源条件设定下的关键状态表示；但那些 strongest results 本身并不是 diffusion。  
因此，这一轮新增一个最小可行的：

- **shared-latent hierarchical diffusion**

目标不是立即替代现有主结果，而是先回答一个更基础的问题：

- shared `regional context` 能否被真正并回 diffusion 主线？

### 13.2 实现

新增脚本：

- `tools/train_external_joint_hier_diffusion_full.py`
- `tools/run_external_joint_hier_diffusion_full.sh`

当前实现是最小可行版，而不是双层 diffusion：

1. `external condition -> encoder -> regional context z`
2. `z -> diffusion denoiser -> fine/full joint`
3. `z -> coarse head -> 72-cell coarse joint`
4. 用 coarse auxiliary loss 和 fine-to-coarse consistency loss 约束层级结构

这里 diffusion 仍然是主 joint generator；shared latent 通过 condition injection 进入 diffusion，而不是用 MLP head 替代 diffusion。

### 13.3 本地 smoke

本地先用 Michigan staging 数据做了一个 2-fold smoke：

- Python 静态检查通过
- shell 语法检查通过
- 训练、fold 评估、metrics 写出都已跑通

期间修了两个兼容性/稳定性问题：

1. `datetime.UTC` 改为 `datetime.timezone.utc`
2. coarse aggregation 增加 `nan/inf` 清洗，避免在 nonfatal warning 下中断整体流程

另一个发现是：

- 本地 Michigan 条件文件 `extcond_v1_acs5_2022_puma_state26_michigan.csv`
- 缺少显式的 `statefp/puma_uid`

这不影响 nationwide remote pilot，但说明 Michigan 旧条件文件的 geography 键仍需后续刷新。

### 13.4 wsA exploratory pilot

已在 wsA 上完成一轮 exploratory pilot：

- run:
  - `outputs/_us_puma_external_joint_hier_diffusion_full_pilot_20260325T064159Z`
- split:
  - `leave_mi_out`
- setting:
  - `timesteps=200`
  - `epochs=300`
  - `latent_dim=128`
  - `encoder_hidden_dims=256,256`
  - `diffusion_hidden_dims=512,512`
  - `coarse_weight=0.5`
  - `consistency_weight=1.0`

本地已同步：

- `outputs/_us_puma_external_joint_hier_diffusion_full_pilot_20260325T064159Z/run_summary.json`
- `outputs/_us_puma_external_joint_hier_diffusion_full_pilot_20260325T064159Z/hier_diffusion_summary.json`

核心结果：

- `tvd_joint_raw = 0.21269`
- `tvd_joint = 0.11235`
- `tvd_coarse_head = 0.04890`
- `tvd_coarse_from_fine = 0.05981`

对照：

- old full diffusion baseline:
  - `0.1188`
- IPF baseline:
  - `0.08088`

### 13.5 当前判断

这轮 pilot 给出的信号是清楚的：

1. 这条线已经真正进入 diffusion 主线  
- 它不是概念草图
- 也不是非 diffusion 机制探针
- 而是一个可运行、可评估的 shared-latent diffusion 实现

2. 当前 300-epoch pilot 还没有追上 IPF  
- `0.11235 > 0.08088`
- 因此它现在不能进入主文结果

3. 但它已经显著改善了 old full diffusion  
- `0.1188 -> 0.11235`
- relative drop 约为 `5.4%`

4. coarse signal 已经学出来了  
- `tvd_coarse_head = 0.04890`
- `tvd_coarse_from_fine = 0.05981`

这说明：

- shared `regional context` 并回 diffusion 后，不是完全无效
- 当前主要问题不是 coarse state 没学到
- 而是 fine/full joint generation 还没有充分把这一状态转化成比 IPF 更强的最终 joint recovery

### 13.6 定位

因此，这条 shared-latent hierarchical diffusion 线当前最合理的定位是：

- **positive pilot**

它已经回答了：

- 这条路能不能实现、能不能运行、有没有正向信号？  
  - **能**

但它还没有回答：

- 它是否已经足够强，可以替代当前 non-diffusion shared-latent 主结果？  
  - **目前还不能**

### 13.7 训练长度与注入方式 screening

在 300-epoch pilot 之后，这条 diffusion 线最先需要回答的不是“再换什么结构”，而是两个更基本的问题：

1. 当前 gap 是否主要来自训练深度不足？
2. `regional context` 的注入方式，`concat` 与 `FiLM` 是否存在决定性差异？

因此先做了一轮最小 screening：

- `concat + 1000 epochs`
- `FiLM + 1000 epochs`

关键 run：

- `outputs/_us_puma_external_joint_hier_diffusion_full_concat_1000ep_20260325T000000Z`
- `outputs/_us_puma_external_joint_hier_diffusion_full_film_1000ep_20260325T000000Z`

结果：

- `concat + 1000 epochs`
  - `tvd_joint_raw = 0.10062`
  - `tvd_joint = 0.08098`
  - `tvd_coarse_head = 0.04168`
  - `tvd_coarse_from_fine = 0.04331`
- `FiLM + 1000 epochs`
  - `tvd_joint_raw = 0.10134`
  - `tvd_joint = 0.08112`
  - `tvd_coarse_head = 0.04192`
  - `tvd_coarse_from_fine = 0.04392`

对照：

- IPF baseline：
  - `0.08088`

结论：

- 训练长度确实是当前 diffusion 主线的关键因素：
  - `300 epochs` 时仍明显落后于 IPF
  - `1000 epochs` 后已经基本追平 IPF
- `FiLM` 没有优于 `concat`
  - `0.08112` 略差于 `0.08098`
  - coarse 指标也没有显示出更强的优势

因此，这轮 screening 最直接的结论不是“注入方式决定成败”，而是：

> shared-latent diffusion 的主要瓶颈首先在训练深度，而不在 `concat` 与 `FiLM` 的选择。

这一步也很重要，因为它说明：

- `regional context` 并回 diffusion 后，确实已经进入可竞争区间
- 但当前最需要做的不是继续堆 condition injection trick，而是继续把主训练跑到稳定区间

### 13.8 `concat + 2000 epochs` 的 3-seed 结果

在确认 `1000 epochs` 已基本追平 IPF 后，下一步只继续沿当前最优设定推进：

- `condition injection = concat`
- `epochs = 2000`

这样做的目的很明确：

- 不是再做广泛超参 sweep
- 而是判断这条 diffusion 主线在合理训练长度下，是否能稳定超过 IPF，而不是只在单 seed 上偶然追平

关键 run：

- seed 0：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_2000ep_20260325T000000Z`
- seed 1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_2000ep_s1_20260325T000000Z`
- seed 2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_2000ep_s2_20260325T000000Z`
- 3-seed 汇总：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_2000ep_batch_20260325T000000Z_summary.json`

单 seed 结果：

- seed 0：
  - `tvd_joint_raw = 0.08763`
  - `tvd_joint = 0.07826`
  - `tvd_coarse_head = 0.04029`
  - `tvd_coarse_from_fine = 0.04243`
- seed 1：
  - `tvd_joint_raw = 0.09879`
  - `tvd_joint = 0.08005`
  - `tvd_coarse_head = 0.04019`
  - `tvd_coarse_from_fine = 0.04396`
- seed 2：
  - `tvd_joint_raw = 0.09061`
  - `tvd_joint = 0.08020`
  - `tvd_coarse_head = 0.04028`
  - `tvd_coarse_from_fine = 0.04406`

3-seed 汇总：

- `tvd_joint = 0.07950 ± 0.00108`
- `tvd_joint_raw = 0.09234 ± 0.00578`
- `tvd_coarse_head = 0.04025 ± 0.00006`
- `tvd_coarse_from_fine = 0.04348 ± 0.00091`
- IPF baseline：
  - `0.08088`
- 相对 IPF 提升：
  - `1.70%`

结论：

- 在 `concat + 2000 epochs` 下，shared-latent diffusion 已经不再只是 positive pilot
- 三个 seed 的平均结果已经稳定略优于 IPF
- coarse 指标的方差极小，说明 `regional context` 对 coarse state 的组织已经很稳定
- full joint 的优势仍然不大，这说明当前 diffusion 主线已经进入“可竞争但尚未强势领先”的区间

因此，这条线当前最准确的定位应该是：

> 这是一个 **mechanism-informed diffusion improvement**：它已经证明 shared `regional context` 可以被真正并回 diffusion 主线，并在 full `K=500` 设定下稳定带来小幅但一致的增益；但它还不是一个足以取代所有现有主结果的强优势模型。

这一步非常关键，因为它把当前 diffusion 主线从：

- “概念上可行”

推进到了：

- “结果上已经可引用，但需要谨慎措辞”

### 13.9 旧的 `3000/4000` 崩塌结果是 confounded，不足以单独解释 epoch 效应

在 `concat + 2000 epochs` 之后，最初又做了两条更长训练：

- `outputs/_us_puma_external_joint_hier_diffusion_full_concat_3000ep_20260325T000000Z`
- `outputs/_us_puma_external_joint_hier_diffusion_full_concat_4000ep_20260325T000000Z`

这两条 run 表面上对应：

- `concat + 3000 epochs`
- `concat + 4000 epochs`

但回查配置后发现，这两条结果**不能**被直接解释成“只增加训练长度后的表现”，因为它们同时把 diffusion chain 从：

- `timesteps = 200`

改成了：

- `timesteps = 1000`

因此，这两条 run 混入了两个变化：

1. 训练更长  
2. diffusion reverse chain 更长

结果确实出现了明显崩塌：

- `3000ep`：
  - `tvd_joint = 0.97865`
  - `tvd_coarse_head = 0.04362`
  - `tvd_coarse_from_fine = 0.84975`
- `4000ep`：
  - `tvd_joint = 0.97549`
  - `tvd_coarse_head = 0.04486`
  - `tvd_coarse_from_fine = 0.84636`

但这批结果更准确的含义是：

> 在当前 shared-latent diffusion 设定下，`1000-step` diffusion chain 明显不稳定；它不能被直接拿来支持“训练越久越差”这一结论。

因此，这两条 run 的主要价值是：

- 暴露 `timesteps` 是一个强影响因子
- 说明后续讨论 epoch 时，必须固定 `timesteps = 200`
- 推动我们把重点从“继续盲目加 epoch”转到：
  - validation selection
  - EMA
  - sampling stability

### 13.10 固定 `timesteps=200` 后，validation + EMA 明显改善 diffusion 主线

在识别出旧 `3000/4000` run 的 confound 之后，下一步重新固定回与 `2000ep` 最优结果可比的设置：

- `timesteps = 200`
- `n_eval_joint_samples = 64`
- `condition injection = concat`

并额外加入三项训练管理机制：

- validation split
- EMA
- best checkpoint selection

这样做的目的很明确：

- 把“训练更长”与“diffusion chain 更长”分开
- 检查当前瓶颈究竟来自训练深度不足，还是来自 checkpoint / sampling 管理

首先得到的严格可比单 seed run 是：

- `outputs/_us_puma_external_joint_hier_diffusion_full_concat_3000ep_t200_valema_20260325T000000Z`

结果：

- `best_epoch = 3000`
- `best_val_tvd_joint = 0.08223`
- `tvd_joint_raw = 0.08186`
- `tvd_joint = 0.07501`
- `tvd_coarse_head = 0.03881`
- `tvd_coarse_from_fine = 0.04026`

对照：

- `concat + 2000 epochs` 的 3-seed mean：
  - `tvd_joint = 0.07950`
- IPF baseline：
  - `0.08089`

这一步已经说明：

- 问题不只是“训练还不够久”
- 更关键的是：
  - final checkpoint 选择
  - EMA 稳定性
  - training objective 与最终 sample quality 之间的管理

也就是说，shared-latent diffusion 并不是本体上卡住了，而是训练管理不足掩盖了它的真实能力。

### 13.11 `3000ep_t200_valema` 的 3-seed 结果：优势扩大到 `7.45%`

在确认 `3000ep_t200_valema` 的 seed 0 已明显优于旧 `2000ep` 结果后，继续补齐：

- seed 1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_3000ep_t200_valema_s1_20260325T082642Z`
- seed 2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_3000ep_t200_valema_s2_20260325T082642Z`
- 3-seed 汇总：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_concat_3000ep_t200_valema_batch_20260325T082642Z_summary.json`

三个 seed 的 best epoch 分别为：

- seed 0：
  - `3000`
- seed 1：
  - `2600`
- seed 2：
  - `2400`

3-seed 汇总结果：

- `tvd_joint = 0.07485 ± 0.00014`
- `tvd_joint_raw = 0.08238 ± 0.00055`
- `tvd_coarse_head = 0.03906 ± 0.00022`
- `tvd_coarse_from_fine = 0.04053 ± 0.00035`
- IPF baseline：
  - `0.08088 ± 0.00002`
- 相对 IPF 提升：
  - `7.45%`

这一步的重要性在于：

- diffusion 主线不再只是“略微优于 IPF”
- `regional context` 被并回 diffusion 之后，full `K=500` 的优势已经进入 manuscript-relevant 区间
- 且结果非常稳定：
  - `tvd_joint` 的 seed 间标准差只有 `0.00014`
  - `tvd_coarse_head` 和 `tvd_coarse_from_fine` 也都极稳

因此，这条 diffusion 主线现在最准确的定位已经不再是：

- “small pilot gain”

而应更新为：

> shared-latent hierarchical diffusion 在 full `K=500` external-condition setting 下，已经能稳定并且非微弱地优于 IPF；其主要提升来自把 `regional context` 真正并入 diffusion，并通过 validation + EMA + best checkpoint 管理释放这条主线的性能。

### 13.12 `4000ep_t200_valema` 没有继续提升，`3000ep` 更像当前 sweet spot

在 `3000ep_t200_valema` 3-seed 已经稳定后，又追加了一条严格可比的更长训练：

- `outputs/_us_puma_external_joint_hier_diffusion_full_concat_4000ep_t200_valema_20260325T082642Z`

结果：

- `best_epoch = 4000`
- `best_val_tvd_joint = 0.08203`
- `tvd_joint_raw = 0.08147`
- `tvd_joint = 0.07513`
- `tvd_coarse_head = 0.03846`
- `tvd_coarse_from_fine = 0.03985`
- IPF baseline：
  - `0.08089`
- 相对 IPF 提升：
  - `7.12%`

与 `3000ep_t200_valema` 的 3-seed mean 对比：

- `3000ep`：
  - `0.07485 ± 0.00014`
- `4000ep` seed 0：
  - `0.07513`

差异很小，但方向上没有继续提升，反而略微回退：

- `4000ep - 3000ep mean = +0.00028`

因此，现在更合理的结论是：

> 在 fixed `timesteps=200`、带 validation + EMA + best checkpoint 的设定下，继续把训练上限从 `3000` 推到 `4000` 并不会带来进一步收益；`3000ep` 更像当前 shared-latent diffusion 的有效 sweet spot。

这一步把训练问题收得更清楚了：

- 之前的主要问题并不是“必须无限加长训练”
- 而是：
  - 训练配置要严格可比
  - checkpoint 选择要显式控制
  - sampling stability 要纳入主流程

一旦这些管理机制补上，diffusion 主线就已经能稳定达到当前最强区间。

### 13.13 `5-way full diffusion` 的 failure 定位：问题不在 regional context，而在 added earnings axis 没有被训练口径显式锚定

对 `5-way full joint diffusion` 的首次 3-seed 结果：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_batch_.json`

显示：

- `tvd_joint = 0.58083 ± 0.00446`
- `tvd_joint_raw = 0.91984 ± 0.00124`
- `tvd_coarse_head = 0.03938 ± 0.00021`
- `tvd_coarse_from_fine = 0.27648 ± 0.00692`
- `5-way IPF = 0.12699 ± 0.00002`

这组数字的关键含义是：

- coarse head 仍然很好
- 但 raw diffusion sample 对 full `3000-cell` joint 基本失控

因此 failure 不是：

- `regional context` 学不到

而是：

- `5-way` 新增的 earnings 轴没有被当前 hierarchy 显式约束
- raw fine sample 也缺少直接的 marginal consistency training signal

进一步检查 target 稀疏度：

- `4-attr`：
  - zero fraction `0.682`
- `5-way`：
  - zero fraction `0.858`

说明 direct `3000-cell` diffusion 确实把模型推到了更稀疏、更不稳定的 simplex 上。

### 13.14 `5-way full diffusion v2`：加入 earnings-aware coarse auxiliary 和 marginal consistency loss 后，failure 明显缓解

在新的 `v2` 训练口径中，做了两件最小修复：

1. coarse auxiliary 从旧的 `72-cell` 四属性 lite state，扩成含 earnings 的 `288-cell` coarse state
2. 在训练中对 full joint 的 raw `x0` prediction 增加显式 marginal consistency loss

相关脚本：

- `tools/train_external_joint_hier_diffusion_full.py`
- `tools/train_external_joint_hier_diffusion_full_earn.py`
- `tools/run_external_joint_hier_diffusion_full_earn.sh`

先做了一个 single-seed pilot：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_pilot_20260325T160301Z`

结果：

- `best_epoch = 600`
- `best_val_tvd_joint = 0.53615`
- `tvd_joint = 0.52594`
- `tvd_joint_raw = 0.88984`
- `tvd_coarse_head = 0.09021`
- `tvd_coarse_from_fine = 0.37133`

相对旧版 `5-way` seed 0：

- old:
  - `tvd_joint = 0.57877`
  - `tvd_joint_raw = 0.92000`
- v2:
  - `tvd_joint = 0.52594`
  - `tvd_joint_raw = 0.88984`

说明：

- 训练口径修复是有效的
- `5-way full diffusion` 的 failure 不是不可救的
- 但当前 `v2` 还没有进入可引用区间

### 13.15 `marginal_weight` 不能简单拉大；当前更像 denoiser capacity bottleneck

在同一 `v2` 设定下，追加了：

- `marginal_weight = 10`
- run:
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_m10_20260325T160558Z`

结果反而更差：

- `best_val_tvd_joint = 0.60625`
- `tvd_joint = 0.60577`

这说明当前问题不是：

- marginal consistency signal 不存在

而更像是：

- 这个 signal 过强时会直接压坏 diffusion denoising

因此下一步更合理的方向不是继续加大 `marginal_weight`，而是提升 denoiser 对 `3000-cell` sparse joint 的表达能力。

### 13.16 更宽的 denoiser 是当前最有希望的方向，3-seed 正在运行

在 `v2` 基础上，只做一个改动：

- `diffusion_hidden_dims = 1024,1024`

single-seed pilot：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_wide_20260325T160811Z`

结果：

- `best_epoch = 400`
- `best_val_tvd_joint = 0.46666`
- `tvd_joint = 0.43957`
- `tvd_joint_raw = 0.77289`
- `tvd_coarse_head = 0.19262`
- `tvd_coarse_from_fine = 0.28123`

相对 `v2` baseline：

- `0.52594 -> 0.43957`

说明：

- 在训练口径修复之后，capacity 的确成为下一阶段主要瓶颈
- 更宽的 denoiser 能显著改善 `5-way` full diffusion

为验证这个提升是否稳定，又启动了：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_wide400_batch_20260325T161019Z_*`

这组 `3 seeds` 目前正在远端运行，用来确认：

- `v2 + wide denoiser + 400ep`

是否能把 `5-way full diffusion` 继续从 exploratory failure 推向 manuscript-relevant regime。

### 13.17 `5-way` 问题重新回到数据源头：3000-cell target 的主矛盾是 dead-cell 占比过高，而不是 drift 仍未解决

在 `rawclip3` 之后，`5-way` 的症状已经从：

- old `5-way`：
  - `tvd_joint ≈ 0.58`

收缩到：

- `rawclip3`：
  - `tvd_joint = 0.15584 ± 0.00002`
  - `tvd_joint_raw = 0.24248 ± 0.00116`
  - `5-way IPF = 0.12699`

相关 summary：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_rawclip3_batch_20260326T000000Z_summary.json`

这说明：

- raw sample drift 的主要症状已经被显著压住
- 当前真正剩下的 gap 是：
  - `0.15584 -> 0.12699`

为定位这条剩余 gap，又直接对 `3000-cell` target 做了支持度统计：

- `outputs/_us_puma_external_joint_deadcell_summary_20260326T000000Z.json`

结果非常极端：

- 每个区域平均只有 `424.93 / 3000` 个非零 cells
- 平均零占比：
  - `0.85836`
- 在全部 `2456` 个 PUMA 中始终为零的 cells 有：
  - `1776`
- 只在 `1` 个区域非零的 cells 有：
  - `23`
- 只在 `2-5` 个区域非零的 cells 有：
  - `62`

这说明当前 `5-way` 训练面对的不是“稍微稀疏”的 joint，而是：

- 大多数维度都没有真实概率质量
- 真正承载 copula 异质性的 live cells 只占很小一部分

因此 `5-way` 当前最像的问题，不再是：

- `regional context` 学不到
- 或 raw drift 还没压住

而是：

- `epsilon MSE` 对 3000 个 cells 一视同仁
- 导致 dead cells 抢走了大量梯度预算

### 13.18 对 `epsilon MSE` 做 cell-weighted reweighting 出现正向信号；`alpha` 过大开始回退

基于上面的 dead-cell 诊断，在 `rawclip` baseline 上进一步做了最小改动：

- 训练脚本：
  - `tools/train_external_joint_hier_diffusion_full.py`
- 运行脚本：
  - `tools/run_external_joint_hier_diffusion_full_earn.sh`

改动内容：

- 对 `loss_diff = mse(eps_pred, noise)` 做 cell-wise reweighting
- 权重来自真实 `x0` 对应的 probability mass
- 目标是让 live cells 比 dead cells 获得更大的 diffusion 梯度

先做了 `alpha = 0.5` 的 single-seed pilot：

- `outputs/run_summary_full5diff_weighted_a05_20260326T003011Z.json`

结果：

- `tvd_joint = 0.15325`
- `tvd_joint_raw = 0.22327`

相对 `rawclip3` seed-level baseline：

- `tvd_joint`：
  - `0.15583 -> 0.15325`
- `tvd_joint_raw`：
  - `0.24401 -> 0.22327`

再叠加 `detach_coarse_encoder`：

- `outputs/run_summary_full5diff_weighted_a05_detach_20260326T003141Z.json`

结果进一步改善：

- `tvd_joint = 0.15280`
- `tvd_joint_raw = 0.21873`

最后试了更强的 `alpha = 1.0`（同时保留 detach）：

- `outputs/run_summary_full5diff_weighted_a10_detach_20260326T003303Z.json`

结果是：

- `tvd_joint = 0.15475`
- `tvd_joint_raw = 0.21835`

这说明：

- dead-cell reweighting 的方向是对的
- `alpha = 0.5` 已经能把 `5-way` seed 从 `0.15583` 再往下压一小步
- 再加 `detach` 也有轻微叠加收益
- 但 `alpha` 继续加大时，`raw` 还会继续变好一点，`post-IPF joint` 已经开始回退

因此当前最合理的判断是：

- `cell-weighted epsilon loss` 已经给出了正向信号
- 但还需要更系统的 sweep 和多 seed，不能直接据 single-seed 结果下结论
- 当前最值得继续围绕的配置是：
  - `rawclip + diff_loss_reweight_alpha = 0.5 + detach`

### 13.19 `rawclip + weighted epsilon + detach` 的 3-seed 已经稳定复现正向提升

沿着 13.18 中最有希望的 single-seed 配置，进一步固定：

- `logp_clip_quantile_low = 0.001`
- `logp_clip_quantile_high = 0.999`
- `selection_metric = val_tvd_joint_raw`
- `diff_loss_reweight_alpha = 0.5`
- `diff_loss_reweight_floor = 0.05`
- `diff_loss_reweight_cap = 5.0`
- `detach_coarse_encoder = true`

并补了正式 `3 seeds` batch：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_batch_20260326T004000Z_summary.json`

结果：

- `tvd_joint = 0.15244 ± 0.00033`
- `tvd_joint_raw = 0.21821 ± 0.00069`
- `tvd_coarse_head = 0.10490 ± 0.00447`
- `tvd_coarse_from_fine = 0.08602 ± 0.00034`
- `5-way IPF = 0.12699`

相对 `rawclip3`：

- `tvd_joint`：
  - `0.15584 -> 0.15244`
- `tvd_joint_raw`：
  - `0.24248 -> 0.21821`

三个 seeds 都很一致：

- seed0:
  - `0.15280`
- seed1:
  - `0.15200`
- seed2:
  - `0.15251`

说明：

- dead-cell reweighting 的改进不是 single-seed 偶然结果
- 这条线已经稳定地把 `5-way` 的 raw sample 和 post-IPF joint 都往下压了一步

但同时也要诚实地说：

- 这条配置虽然显著优于 `rawclip3`
- 仍然没有追上 `5-way IPF = 0.12699`

因此当前最准确的判断是：

- `dead-cell gradient budget` 的确是主矛盾
- `cell-weighted epsilon loss` 是有效方向
- 但还需要继续优化，才能真正把 `5-way full diffusion` 推到超过 IPF 的区间

### 13.20 `5-way coarse-to-fine` 的真正 seed-matched 3-run 已跑完，稳定优于单阶段主线

在 `5-way full diffusion` 单阶段主线稳定卡在：

- `tvd_joint = 0.15244 ± 0.00033`

之后，沿着显式 `coarse-to-fine` 路线继续推进：

- Stage 1：
  - `288-cell` coarse predictor
- Stage 2：
  - teacher-forced fine refinement model
- evaluator：
  - 先用 Stage 1 预测 coarse
  - 再由 Stage 2 细化回完整 `3000-cell` joint

前面已经做过一个“半正式”版本：

- 3 个 Stage 1 seeds
- 但 Stage 2 固定为 `seed0`

这次补齐了真正的 seed-matched 版本：

- `Stage1 seed0 + Stage2 seed0`
- `Stage1 seed1 + Stage2 seed1`
- `Stage1 seed2 + Stage2 seed2`

对应汇总文件：

- `outputs/_us_puma_external_c2f_full_earn_eval_seedmatched_true3_20260326T164000Z_summary.json`

结果如下：

- Stage 1 coarse raw:
  - `tvd = 0.10490 ± 0.00447`
- Stage 1 coarse IPF:
  - `tvd = 0.07268 ± 0.00130`

- pipeline, 直接用 Stage 1 coarse:
  - `tvd_joint_raw = 0.18174 ± 0.00355`
  - `tvd_joint = 0.14707 ± 0.00307`

- pipeline, 对 Stage 1 coarse 先做 coarse-level IPF 再细化:
  - `tvd_joint_raw = 0.16200 ± 0.00176`
  - `tvd_joint = 0.14663 ± 0.00294`

- oracle Stage 2（真 coarse）:
  - `tvd_joint = 0.12068 ± 0.00241`

- uniform refine baseline:
  - `tvd_joint = 0.26528 ± 0.00046`

对照当前单阶段 best：

- one-shot `5-way full diffusion`:
  - `tvd_joint = 0.15244`

相对提升：

- `coarse-to-fine + coarse IPF` 相对单阶段提升：
  - `3.81%`

与 `5-way IPF` 的比较：

- `5-way IPF = 0.12699`
- `coarse-to-fine + coarse IPF = 0.14663`

这轮结果把几件事说明白了：

- `coarse-to-fine` 已经不是单 seed 偶然现象，而是在真正 seed-matched 设置下稳定优于当前单阶段主线
- Stage 2 不是当前瓶颈
  - 因为 `oracle Stage 2 = 0.12068`，已经优于 `5-way IPF = 0.12699`
- 当前主要瓶颈仍然是 Stage 1 coarse prediction
  - coarse 更准，整条 `coarse-to-fine` 就有机会真正过 IPF

因此现在最合理的主线判断是：

- `5-way` 上继续堆单阶段 trick 的边际收益已经开始变小
- 显式 `coarse-to-fine` 是当前更有前景的结构性方向
- 后续如果继续投资源，优先级应该放在：
  - 如何把 Stage 1 coarse 做得更准

### 13.21 Stage 2 加 `cell-weighted epsilon loss` 后，true3 稳定变好；但 dedicated Stage1 的主问题仍在

PI 提的这个方向值得做，而且现在已经有了真正的 `3 seeds` 证据。

动机很直接：

- 当前 Stage 2 的 `54` 维 local target 里，平均只有约 `10` 个 active child slot
- 其余大部分维度都是 padding 零
- 原始 `epsilon MSE` 对所有维度等权，梯度预算被大量浪费在 inactive slot 上

这次改动不是从数据里估计 dead cell，而是直接利用 Stage 2 里本来就存在的 `child_mask`：

- active slot：
  - 用 `mask` 精确标出
- inactive slot：
  - 直接按 floor 降权
- 配置：
  - `alpha = 0.5`
  - `floor = 0.05`
  - `cap = 5.0`

对应 run：

- Stage 2 true3 汇总：
  - `outputs/_us_puma_external_c2f_full_earn_eval_maskw_a05_seedmatched_true3_20260326T133745Z_summary.json`
- seed1 Stage 2 训练：
  - `outputs/_us_puma_external_c2f_full_earn_teacher_maskw_a05_seed1_20260326T131322Z/run_summary.json`
- seed1 end-to-end eval：
  - `outputs/_us_puma_external_c2f_full_earn_eval_maskw_a05_seed1_stage2seed1_fixcond_20260326T131810Z/metrics/coarse_to_fine_summary.json`

true3 汇总结果：

- Stage 2 teacher-forced projected:
  - `0.28671 ± 0.00051`
- Stage 2 inactive mass:
  - `0.01638 ± 0.00148`
- pipeline, Stage 1 coarse 先做 coarse IPF 再细化:
  - `tvd_joint = 0.14069 ± 0.00162`
- pipeline, 直接用 Stage 1 coarse:
  - `tvd_joint = 0.14043 ± 0.00151`
- oracle Stage 2:
  - `tvd_joint = 0.11430 ± 0.00185`
- uniform refine baseline:
  - `tvd_joint = 0.26528 ± 0.00046`

对照旧的 seed-matched true3：

- `pipeline_stage1_coarse_ipf_tvd_joint`：
  - `0.14663 -> 0.14069`
- `pipeline_stage1_raw_tvd_joint`：
  - `0.14707 -> 0.14043`
- `oracle_stage2_tvd_joint`：
  - `0.12068 -> 0.11430`
- `uniform_refine_tvd_joint`：
  - `0.26528 -> 0.26528`

这组对照说明：

- 改善确实来自 Stage 2 refinement，本身不是评估噪声
- 而且不是 single-seed 偶然，而是在 true3 上稳定复现

但更关键的是，专门补做的 dedicated Stage1 对照也说明：

- 即使把 Stage 2 换成新的 weighted 版本
- dedicated Stage1 的主要问题仍然没有被解决

对应 run：

- dedicated Stage1 + old Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarse_seed1_20260326T091000Z/coarse_to_fine_summary.json`
- dedicated Stage1 + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarse_maskw_a05_seed1_stage2seed1_20260326T214800Z/metrics/coarse_to_fine_summary.json`

seed1 dedicated 对照结果：

- Stage 1 coarse IPF:
  - `0.17254 -> 0.17254`
- pipeline coarse-IPF:
  - `0.21952 -> 0.21714`
- pipeline raw:
  - `0.23615 -> 0.22165`
- oracle Stage 2:
  - `0.12396 -> 0.11643`

这意味着：

- Stage 2 weighted loss 是有效的，而且值得并入当前 `coarse-to-fine` 主线
- 但 dedicated Stage1 的负结果并不是 Stage 2 太弱导致的假象
- dedicated Stage1 目前真正的问题仍然是 Stage 1 coarse seed 本身太差

因此此刻最准确的判断变成：

- `weighted Stage2` 已经可以视为当前 `c2f` 主线的更强默认配置
- 但如果目标是进一步逼近甚至超过 `5-way IPF`
- 后续主矛盾依然在 dedicated Stage1 的建模形式，而不是 Stage 2 padding loss

### 13.22 dedicated Stage1 加回 `clean coarse head` 后，负结果大幅缓解；但仍未追上当前主线

在确认 `Stage 2 weighted loss` 已经不是主瓶颈之后，下一步直接回到 Stage 1 本身：

- 旧 dedicated Stage1 的主要问题，不再抽象地说成“独立 Stage1 不行”
- 而是更具体地怀疑：
  - `coarse-only diffusion sampler` 这条实现本身在伤害 coarse seed

因此这轮改动不是继续动 Stage 2，而是在 dedicated Stage1 里补回一条显式 `clean coarse head`：

- 训练时：
  - 保留原 coarse diffusion loss
  - 额外加入 noising-free coarse CE supervision
- 推理时：
  - `predict_coarse()` 直接走 `head`
  - 不再走 diffusion sampling average

对应代码已经加到：

- `tools/train_external_c2f_full_earn_stage1_coarse.py`
- `tools/run_external_c2f_full_earn_stage1_coarse.sh`

这轮先做了 `seed1` 的最小 sweep，核心是 3 个 dedicated 变体：

1. old dedicated diffusion-only
2. head + diffusion，`coarse_head_weight = 0.5`
3. pure head-only，`diffusion_weight = 0.0`
4. head + diffusion，`coarse_head_weight = 1.0`

对应 run：

- old diffusion-only：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_seed1_20260326T090000Z.run_summary.remote.json`
- head + diffusion, `w=0.5`：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_head_seed1_20260326T140100Z/run_summary.json`
- pure head-only：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headonly_seed1_20260326T140500Z/run_summary.json`
- head + diffusion, `w=1.0`：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_seed1_20260326T140800Z/run_summary.json`

Stage 1 自身 coarse 指标如下：

- old diffusion-only：
  - `tvd_joint = 0.17092`
  - `tvd_joint_raw = 0.33069`
- head + diffusion, `w=0.5`：
  - `tvd_joint = 0.10524`
  - `tvd_joint_raw = 0.28554`
- pure head-only：
  - `tvd_joint = 0.11126`
  - `tvd_joint_raw = 0.25737`
- head + diffusion, `w=1.0`：
  - `tvd_joint = 0.10191`
  - `tvd_joint_raw = 0.26653`

这组数字先说明第一件事：

- dedicated Stage1 的大问题，确实主要来自旧的 `coarse diffusion sampler`
- 一旦换回 `clean coarse head`
  - coarse seed 质量会显著提升

但它也说明第二件事：

- pure head-only 不是最优
- 说明 diffusion 分支作为辅助正则并不是纯负作用
- 当前最好的是：
  - `head + diffusion`
  - 且 `coarse_head_weight = 1.0`

接着把这两个新的 dedicated Stage1 接到当前最强的 weighted Stage2 上，得到：

- old dedicated + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarse_maskw_a05_seed1_stage2seed1_20260326T214800Z/metrics/coarse_to_fine_summary.json`
- head + diffusion, `w=0.5` + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarsehead_maskw_a05_seed1_stage2seed1_20260326T140600Z/metrics/coarse_to_fine_summary.json`
- head + diffusion, `w=1.0` + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarseheadw1_maskw_a05_seed1_stage2seed1_20260326T141000Z/metrics/coarse_to_fine_summary.json`

seed1 end-to-end 结果：

- old dedicated + weighted Stage2：
  - `stage1_coarse_tvd_ipf = 0.17254`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.21714`
  - `pipeline_stage1_raw_tvd_joint = 0.22165`
- head + diffusion, `w=0.5`：
  - `stage1_coarse_tvd_ipf = 0.10524`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.16712`
  - `pipeline_stage1_raw_tvd_joint = 0.16908`
- head + diffusion, `w=1.0`：
  - `stage1_coarse_tvd_ipf = 0.10191`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.16478`
  - `pipeline_stage1_raw_tvd_joint = 0.16713`

同时，oracle Stage 2 在这些 dedicated run 里保持不变：

- `oracle_stage2_tvd_joint = 0.11642`

因此这轮实验把问题进一步钉实了：

- dedicated negative result 的大头，的确来自 Stage1 架构实现
- 把 `coarse-only diffusion sampler` 改成 `clean coarse head`
  - dedicated Stage1 会明显变好
- 但即使 dedicated Stage1 已经显著改善
  - 当前最好配置 `0.16478`
  - 仍然明显落后于当前主线：
    - `mainline c2f + weighted Stage2 = 0.14298`
  - 也还没有超过：
    - `one-shot 5-way diffusion = 0.15244`
    - `5-way IPF = 0.12699`

所以现在的判断比之前更精确：

- “dedicated Stage1 方向不行” 这个说法不成立
- “dedicated Stage1 用 coarse-only diffusion sampler 不行” 这个说法成立
- 但即使 dedicated Stage1 换成 clean head
  - 仍然没有追上 full-model coarse head

这说明当前剩下的 gap 更可能来自：

- full-task / fine-task 对 coarse latent 的反向帮助
- 或者 coarse/fine consistency 在 full model 里的联动监督

而不是单纯因为 dedicated Stage1 缺少一个 clean predictor

### 13.23 dedicated Stage1 再补上 `head <-> diffusion consistency` 后还能继续提升，而且主增益主要来自 consistency

上一轮把 dedicated Stage1 从：

- `coarse-only diffusion sampler`

改成了：

- `clean coarse head + diffusion`

已经证明 dedicated negative result 的一大块问题来自旧实现本身。  
但当时还有一个明显缺口没有补：

- old full model 里一直存在 `coarse/fine consistency`
- dedicated Stage1 里仍然没有任何 `head <-> diffusion` 联动约束

因此这轮没有再去掉 diffusion，而是沿着主线逻辑继续补 dedicated 里缺失的联动项：

- `consistency_weight = 1.0`
- 同时测试：
  - `marginal_weight = 0.0`
  - `marginal_weight = 1.0`

对应 run：

- best 旧版 head + diffusion（无 consistency）：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_seed1_20260326T140800Z/run_summary.json`
- consistency-only：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_cons1_seed1_20260326T143400Z/run_summary.json`
- consistency + marginal：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_cons1_marg1_seed1_20260326T142900Z/run_summary.json`

Stage 1 自身 coarse 指标：

- head + diffusion（旧 best）：
  - `tvd_joint = 0.10191`
  - `tvd_joint_raw = 0.26653`
- consistency-only：
  - `tvd_joint = 0.09079`
  - `tvd_joint_raw = 0.20228`
- consistency + marginal：
  - `tvd_joint = 0.09200`
  - `tvd_joint_raw = 0.20708`

这组结果已经很说明问题：

- dedicated Stage1 继续变好，不是靠砍掉 diffusion
- 而是靠把 `head` 和 `diffusion` 真正耦合起来
- 而且在 `seed1` 上，主增益看起来主要来自：
  - `consistency`
- `marginal` 至少在这一步没有带来额外收益

接着把这两组接到当前最强 weighted Stage2 上：

- consistency-only + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarseheadw1_cons1_maskw_a05_seed1_stage2seed1_20260326T143600Z/metrics/coarse_to_fine_summary.json`
- consistency + marginal + weighted Stage2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarseheadw1_cons1_marg1_maskw_a05_seed1_stage2seed1_20260326T143100Z/metrics/coarse_to_fine_summary.json`

seed1 end-to-end 对比：

- head + diffusion（旧 best）：
  - `stage1_coarse_tvd_ipf = 0.10191`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.16478`
  - `pipeline_stage1_raw_tvd_joint = 0.16713`
- consistency-only：
  - `stage1_coarse_tvd_ipf = 0.09079`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.15589`
  - `pipeline_stage1_raw_tvd_joint = 0.15671`
- consistency + marginal：
  - `stage1_coarse_tvd_ipf = 0.09200`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.15671`
  - `pipeline_stage1_raw_tvd_joint = 0.15715`

这轮结果把 dedicated Stage1 的问题又缩小了一步：

- 从最早的：
  - `0.21714`
- 到 clean head：
  - `0.16478`
- 再到 consistency-only：
  - `0.15589`

说明：

- dedicated Stage1 的剩余 gap，确实和缺失的联动监督有关
- 而且 `consistency` 比单纯加 `marginal` 更像是主导因素

但同样要诚实地说：

- 即使 dedicated Stage1 已经补到 `clean head + diffusion + consistency`
- 当前 seed1 仍然没有追上：
  - current mainline `0.14298`
- 也仍然还没有超过：
  - one-shot `0.15244`
  - `5-way IPF = 0.12699`

所以此刻最准确的判断进一步更新为：

- dedicated Stage1 的问题并没有“解决”
- 但已经从“架构明显错位”收敛成了“还差一段 full-model 联动带来的 coarse latent 质量”
- 当前 dedicated 线里最值得保留的形态不是：
  - pure head-only
  - coarse-only diffusion
- 而是：
  - `clean head + diffusion + consistency`

### 13.24 对 frozen full-model coarse head 做 output distillation，当前没有带来 dedicated Stage1 的净收益

在 dedicated Stage1 已经收敛到 `clean head + diffusion + consistency` 之后，下一步最直接的问题是：

- dedicated 和 current mainline 剩下的 gap
- 到底是不是因为 dedicated 没有显式看到 full-model 已经学到的 coarse 输出行为

因此这轮没有去做 latent-level 对齐，而是先做最保守的 output-level distillation：

- teacher：
  - current best full-model 的 frozen coarse head
- student：
  - dedicated Stage1，仍然保留
    - diffusion loss
    - clean head loss
    - consistency loss
- 额外加：
  - `KL(student coarse || teacher coarse)`

先做了两个 `seed1` 变体：

- `distill_weight = 0.5`：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_cons1_kd05_seed1_20260326T145900Z/run_summary.json`
- `distill_weight = 0.1`：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_cons1_kd01_seed1_20260326T150154Z/run_summary.json`

与当前 best dedicated `consistency-only` 对比：

- consistency-only：
  - `tvd_joint = 0.09079`
  - `tvd_joint_raw = 0.20228`
- + distill, `w = 0.5`：
  - `tvd_joint = 0.09632`
  - `tvd_joint_raw = 0.21238`
- + distill, `w = 0.1`：
  - `tvd_joint = 0.09327`
  - `tvd_joint_raw = 0.20706`

这组结果说明得比较直接：

- teacher soft target 不是完全没信息
- 但当前这种 frozen output distillation 形式
  - 并没有给 dedicated Stage1 带来净收益
- 而且 teacher 权重一旦偏大
  - 还会明确伤害结果

因此当前更准确的判断是：

- dedicated 和 full-model 之间的剩余差距
- 不能简单理解成“缺一个 coarse soft label”
- 更像是：
  - full-model 联合训练过程中的表征耦合
  - 比事后蒸馏一个 frozen coarse 输出更重要

这也意味着：

- 如果后面还继续给 dedicated 一次机会
- 更值得测的方向应该是：
  - 训练动态相关的耦合
  - 或更谨慎的低强度蒸馏
- 而不是继续把 `distill_weight` 往大处推

### 13.25 只在低噪声阶段施加 consistency，可以继续改善 dedicated Stage1；但 end-to-end 仍未超过 one-shot

上一步 `consistency-only` 虽然有效，但它还有一个可疑点：

- 当前 `consistency`
  - 在所有 diffusion timestep 上等权施加
- 这可能会把高噪声阶段的不稳定 `p_pred`
  - 也强行拉向 clean coarse head

因此这轮测试的是：

- 保持 `clean head + diffusion + consistency`
- 只把 consistency 限制在低噪声阶段
- 具体设置：
  - `aux_t_gate = 50`
  - `timesteps = 200`

对应 run：

- Stage1：
  - `outputs/_us_puma_external_c2f_full_earn_stage1_coarse_headw1_cons1_gate50_seed1_20260326T150251Z/run_summary.json`
- end-to-end eval：
  - `outputs/_us_puma_external_c2f_full_earn_eval_stage1coarseheadw1_cons1_gate50_maskw_a05_seed1_stage2seed1_20260326T150341Z/metrics/coarse_to_fine_summary.json`

Stage1 coarse 对比：

- consistency-only：
  - `tvd_joint = 0.09079`
  - `tvd_joint_raw = 0.20228`
- consistency-only + `aux_t_gate = 50`：
  - `tvd_joint = 0.08714`
  - `tvd_joint_raw = 0.21436`

这说明：

- 对 dedicated Stage1 真正有帮助的 consistency
- 更像是低噪声区域的约束
- 把高噪声时刻也纳入 consistency
  - 反而可能在伤害 coarse seed

接到当前最强 weighted Stage2 后，seed1 end-to-end 结果是：

- consistency-only：
  - `stage1_coarse_tvd_ipf = 0.09079`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.15589`
  - `pipeline_stage1_raw_tvd_joint = 0.15671`
- consistency-only + `aux_t_gate = 50`：
  - `stage1_coarse_tvd_ipf = 0.08714`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.15323`
  - `pipeline_stage1_raw_tvd_joint = 0.15443`

同时：

- `oracle_stage2_tvd_joint` 维持不变：
  - `0.11642`

这表示这次改善确实来自：

- Stage1 coarse seed 本身

但也要诚实地说，这一轮改善仍然不够：

- dedicated + `gate50`：
  - `0.15323`
- one-shot：
  - `0.15244`
- current mainline：
  - `0.14298`
- `5-way IPF`：
  - `0.12699`

所以当前 dedicated 线的最新判断更新为：

- `teacher output distillation`
  - 暂时不是有效突破口
- `low-t consistency gating`
  - 是真实有效的小改进
- 但它还不足以让 dedicated 线超过：
  - one-shot
  - current mainline
  - `5-way IPF`

也就是说：

- dedicated Stage1 的问题还没有被解决
- 但“哪些因素在伤害它，哪些因素在帮助它”
- 已经比前一轮清楚很多

### 13.26 把 `low-t consistency gating` 放回 full-model 主线后，one-shot 变差，但 `c2f` 主线稳定改善

在 dedicated 线里，`aux_t_gate = 50` 已经给出过一个稳定信号：

- `consistency` 真正有效的区域更像是低噪声 timestep

下一步更关键的问题不是 dedicated 本身，而是：

- 这个信号放回真正的 full-model 主线之后
- 到底会改善什么

因此这轮没有再改 dedicated，而是直接回到当前 strongest full-model：

- baseline：
  - `weighted epsilon, alpha = 0.5`
  - `detach_coarse_encoder = true`
  - `selection_metric = val_tvd_joint_raw`
  - `marginal_weight = 1.0`
- 唯一改动：
  - `aux_t_gate = 50`

对应 3 个 full-model run：

- seed0：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_seed0_20260326T152258Z/run_summary.json`
- seed1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_seed1_20260326T151956Z/run_summary.json`
- seed2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_seed2_20260326T152520Z/run_summary.json`

然后用同一个 weighted Stage2 做 seed-matched eval：

- seed0：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_maskw_a05_seed0_stage2seed0_20260326T152346Z/metrics/coarse_to_fine_summary.json`
- seed1：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_maskw_a05_seed1_stage2seed1_20260326T152055Z/metrics/coarse_to_fine_summary.json`
- seed2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_maskw_a05_seed2_stage2seed2_20260326T152612Z/metrics/coarse_to_fine_summary.json`

新的 true3 汇总：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_maskw_a05_seedmatched_true3_20260326T153000Z_summary.json`

#### 13.26.1 first-order 结果：full-model 自己更差了，但 coarse head 明显更好了

3 seeds 的 full-model one-shot：

- old strongest one-shot：
  - `tvd_joint = 0.15244 ± 0.00033`
  - `tvd_coarse_head = 0.10490 ± 0.00447`
- `gate50` full-model：
  - `tvd_joint = 0.15691 ± 0.00092`
  - `tvd_coarse_head = 0.09351 ± 0.00243`

这说明：

- `aux_t_gate = 50` 对 full-model 不是“全面提升”
- 它会伤害 one-shot full-joint sampling
- 但会明显改善 coarse head 质量

这组结果本身已经很重要，因为它第一次把两个目标分开了：

- 对 one-shot 有利的训练目标
- 不一定对 `c2f Stage1` 最有利

#### 13.26.2 真正关键的结果：`c2f` 主线稳定变好，而且三颗 seeds 一致

对比当前 weighted Stage2 主线：

- old mainline：
  - `stage1_coarse_tvd_raw = 0.10490 ± 0.00447`
  - `stage1_coarse_tvd_ipf = 0.07268 ± 0.00130`
  - `pipeline_stage1_raw_tvd_joint = 0.14043 ± 0.00151`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.14069 ± 0.00162`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

- new mainline + `gate50`：
  - `stage1_coarse_tvd_raw = 0.09351 ± 0.00243`
  - `stage1_coarse_tvd_ipf = 0.06786 ± 0.00024`
  - `pipeline_stage1_raw_tvd_joint = 0.13718 ± 0.00161`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13735 ± 0.00156`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

绝对改善量：

- `stage1_coarse_tvd_raw`：
  - `0.10490 -> 0.09351`
  - 改善 `0.01139`
- `stage1_coarse_tvd_ipf`：
  - `0.07268 -> 0.06786`
  - 改善 `0.00482`
- `pipeline_stage1_raw_tvd_joint`：
  - `0.14043 -> 0.13718`
  - 改善 `0.00326`
- `pipeline_stage1_coarse_ipf_tvd_joint`：
  - `0.14069 -> 0.13735`
  - 改善 `0.00334`

同时：

- `oracle_stage2_tvd_joint`
  - 完全不变

这说明这轮增益非常干净：

- 不是 Stage2 偶然波动
- 就是 Stage1 coarse seed 质量更好了

#### 13.26.3 这轮最重要的 insight 不是“调参有效”，而是“one-shot optimum 和 c2f optimum 已经分叉”

当前结果揭示了一个更深层的结构性事实：

- 同一个 full-model
- 如果把训练信号往 low-t consistency 方向推
- coarse head 会更好
- 但 full-joint one-shot sample 会变差

而 `c2f` 主线真正用到的 Stage1，恰恰是：

- full-model 里的 coarse head

因此：

- 对 one-shot 最优的 full-model
- 不一定是对 `c2f Stage1` 最优的 full-model

这意味着主线的下一步不该再被表述成：

- “继续改 full-model 的 one-shot 指标”

而应该更准确地表述成：

- “把 full-model 训练/选择更明确地朝 coarse-head quality 对齐”

#### 13.26.4 当前系统位置

new mainline + `gate50` 现在达到：

- `pipeline_stage1_coarse_ipf_tvd_joint = 0.13735 ± 0.00156`

对照：

- old weighted Stage2 mainline：
  - `0.14069 ± 0.00162`
- best one-shot baseline：
  - `0.15244`
- `5-way IPF`：
  - `0.12699`

所以：

- 相对 old mainline
  - 新结果把与 `IPF` 的 gap 从 `0.01371` 压到 `0.01036`
  - gap 缩小了约 `24.4%`
- 相对 best one-shot baseline
  - 现在 `c2f` 的优势已经扩大到约 `9.90%`

但同样要诚实地说：

- 它仍然没有超过 `5-way IPF`

因此最准确的结论是：

- `low-t consistency gating` 已经被证明是主线里真实有效的方向
- 而且它带来的不是 one-shot improvement
- 而是更直接地作用在 `c2f` 所依赖的 Stage1 coarse head 上
- 这把主线往前推进了一步
- 但离真正稳定超过 `5-way IPF` 还差最后一段

### 13.27 把 full-model 的 checkpoint 选择直接对齐到 `val_tvd_coarse_head` 后，主线进一步逼近 `5-way IPF`

13.26 之后，真正需要回答的问题已经很明确：

- 既然 one-shot optimum 和 `c2f` optimum 已经分叉
- 那么 current mainline 继续按 `val_tvd_joint_raw` 选 checkpoint
- 其实还是在用 one-shot 目标挑 Stage1

这会带来一个结构性错位：

- `c2f` 实际消费的是 full-model 的 `coarse head`
- 但 checkpoint 选择还在优先看 full-joint raw sample

因此这轮不是再改 loss 权重，而是做了一个更直接的目标对齐：

- 在 `tools/train_external_joint_hier_diffusion_full.py` 中
  - 把 `selection_metric` 扩展为支持：
    - `val_tvd_coarse_head`
    - `val_tvd_coarse_from_fine`
- 然后固定：
  - `weighted epsilon, alpha = 0.5`
  - `detach_coarse_encoder = true`
  - `aux_t_gate = 50`
- 只把 checkpoint selection 改成：
  - `selection_metric = val_tvd_coarse_head`
- 同时把训练延长到：
  - `epochs = 1200`

对应 3 个 full-model run：

- seed0：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_seed0_20260326T155301Z/run_summary.json`
- seed1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_seed1_20260326T155002Z/run_summary.json`
- seed2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_seed2_20260326T155551Z/run_summary.json`

对应 seed-matched `c2f` eval：

- seed0：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_maskw_a05_seed0_stage2seed0_20260326T155418Z/metrics/coarse_to_fine_summary.json`
- seed1：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_maskw_a05_seed1_stage2seed1_20260326T155122Z/metrics/coarse_to_fine_summary.json`
- seed2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_maskw_a05_seed2_stage2seed2_20260326T155658Z/metrics/coarse_to_fine_summary.json`

新的 true3 汇总：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_maskw_a05_seedmatched_true3_20260326T155900Z_summary.json`

#### 13.27.1 first-order 结果：这次不是只让 coarse 更好，连 full-model one-shot 自己也被拉回来了

相对 `gate50` 但仍按 `val_tvd_joint_raw` 选 checkpoint 的版本：

- old `gate50` full-model：
  - `one-shot = 0.15691 ± 0.00092`
  - `tvd_coarse_head = 0.09351 ± 0.00243`
- new `gate50 + selcoarse` full-model：
  - `one-shot = 0.14912 ± 0.00039`
  - `tvd_coarse_head = 0.06152 ± 0.00041`

这点很关键，因为它说明：

- 问题不只是 training objective
- checkpoint selection 本身就在主导最终结论

也就是说：

- 同样一套训练轨迹
- 如果你用更接近 `Stage1` 目标的指标来挑 checkpoint
- 不仅 coarse head 会明显变好
- 连 one-shot full-joint 自己都能被一起拉回去

#### 13.27.2 真正关键的结果：`c2f` 主线已经被继续压到 `0.13176`

对比三条主线：

- old weighted Stage2 mainline：
  - `stage1_coarse_tvd_ipf = 0.07268 ± 0.00130`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.14069 ± 0.00162`

- `gate50` mainline：
  - `stage1_coarse_tvd_ipf = 0.06786 ± 0.00024`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13735 ± 0.00156`

- `gate50 + selcoarse` mainline：
  - `stage1_coarse_tvd_raw = 0.06152 ± 0.00041`
  - `stage1_coarse_tvd_ipf = 0.05911 ± 0.00023`
  - `pipeline_stage1_raw_tvd_joint = 0.13178 ± 0.00164`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13176 ± 0.00154`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

这组数字把一件事说得很清楚：

- `Stage2` 仍然没变
- 新增收益几乎全部来自 Stage1
- 而且这次不是小幅波动
- 是从 `0.14069 -> 0.13176`
  - 直接再降了 `0.00893`

#### 13.27.3 当前系统位置：离 `IPF` 只差最后 `0.00477`

对照：

- `gate50 + selcoarse` mainline：
  - `0.13176 ± 0.00154`
- `5-way IPF`：
  - `0.12699`
- best one-shot baseline：
  - `0.15244`

因此：

- 相对 best one-shot baseline
  - `c2f` 优势已经扩大到：
    - `13.56%`
- 相对 old weighted Stage2 mainline
  - `0.14069 -> 0.13176`
  - 改善约：
    - `6.35%`
- 相对 `IPF`
  - gap 已经从：
    - `0.01371`
  - 压到：
    - `0.00477`

也就是说：

- 相比 13.26，这一轮又把 remaining gap 砍掉了一大截
- 当前主线已经不是“离 IPF 很远”
- 而是“最后一小步还没迈过去”

#### 13.27.4 这轮真正的 insight

这轮最重要的洞见不是：

- “再多训几轮更好”

而是：

- 对 `c2f` 来说，full-model 的 checkpoint 选择必须和 `Stage1 coarse quality` 对齐

更具体地说：

- 用 `val_tvd_joint_raw` 选 checkpoint
  - 会把模型拉向 one-shot 目标
- 用 `val_tvd_coarse_head` 选 checkpoint
  - 才更接近 `c2f` 真正要消费的对象

因此 current mainline 的最准确表述已经更新为：

- 当前最有效的方向，不是继续把 one-shot 当代理目标
- 而是显式把 Stage1 selection 和 `c2f` 目标对齐

这件事现在已经不再是猜测，而是被真正的 `seed-matched true3` 结果坐实了。

### 13.28 `gate50 + selcoarse` 主线继续延长到 `1800 epochs` 后，non-oracle `c2f` 继续下降，且 3 seeds 的 best checkpoint 仍全部贴着训练上界

13.27 之后，最直接需要回答的问题不是“再换一个 loss 会不会更好”，而是：

- 既然 `gate50 + selcoarse` 的 3 个 run 都在 `1200` epoch 时被选中
- 而 validation `tvd_coarse_head` 仍在单调下降
- 那 current mainline 会不会只是被训练时长截断了

因此这轮没有改结构，也没有改权重，只做了一件事：

- 保持 current strongest mainline 完全不变：
  - `weighted epsilon, alpha = 0.5`
  - `detach_coarse_encoder = true`
  - `aux_t_gate = 50`
  - `selection_metric = val_tvd_coarse_head`
  - `logp_clip_quantile_low/high = 0.001 / 0.999`
- 只把训练时长从：
  - `epochs = 1200`
  - 延长到：
  - `epochs = 1800`

对应 3 个 full-model run：

- seed0：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep1800_seed0_20260326T161027Z/run_summary.json`
- seed1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep1800_seed1_20260326T160638Z/run_summary.json`
- seed2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep1800_seed2_20260326T161251Z/run_summary.json`

对应 seed-matched `c2f` eval：

- seed0：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep1800_maskw_a05_seed0_stage2seed0_20260326T161027Z/metrics/coarse_to_fine_summary.json`
- seed1：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep1800_maskw_a05_seed1_stage2seed1_20260326T160638Z/metrics/coarse_to_fine_summary.json`
- seed2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep1800_maskw_a05_seed2_stage2seed2_20260326T161251Z/metrics/coarse_to_fine_summary.json`

新的 true3 汇总：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep1800_maskw_a05_seedmatched_true3_20260327T000000Z_summary.json`

#### 13.28.1 first-order 结果：延长训练不是边角收益，而是干净地继续改进 `Stage1`

相对 `1200 epoch` 的 `gate50 + selcoarse` mainline：

- old `1200 epoch`：
  - `stage1_coarse_tvd_raw = 0.06152 ± 0.00058`
  - `stage1_coarse_tvd_ipf = 0.05911 ± 0.00023`
  - `pipeline_stage1_raw_tvd_joint = 0.13178 ± 0.00164`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13176 ± 0.00154`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

- new `1800 epoch`：
  - `stage1_coarse_tvd_raw = 0.05825 ± 0.00058`
  - `stage1_coarse_tvd_ipf = 0.05768 ± 0.00034`
  - `pipeline_stage1_raw_tvd_joint = 0.13079 ± 0.00158`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13080 ± 0.00147`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

这组数字说明：

- `Stage2` 完全没变
- 新增收益仍然全部来自 `Stage1`
- 而且不是 seed 偶然
  - 因为它在 `true3` 下也继续成立

#### 13.28.2 更关键的信号：3 个 full-model 的 best checkpoint 全都还在 `1800`

对这轮 3 个 full-model run：

- seed0：
  - `best_epoch = 1800`
- seed1：
  - `best_epoch = 1800`
- seed2：
  - `best_epoch = 1800`

同时其 one-shot / coarse-head 均值也继续下降：

- `one-shot = 0.14756 ± 0.00059`
- `tvd_coarse_head = 0.05825 ± 0.00058`

这点比纯数值改善更重要，因为它表明：

- current mainline 还没有真正收敛
- `1200` 只是一个过早截断点
- 当前最有效的下一步，不是回到 dedicated，也不是做小超参 sweep
- 而是继续沿着这条已经被证实有效的主线，把训练 horizon 再往前推

#### 13.28.3 当前系统位置：离 `IPF` 只剩最后 `0.00381`

对照：

- `gate50 + selcoarse + ep1800` mainline：
  - `0.13080 ± 0.00147`
- `5-way IPF`：
  - `0.12699`

因此：

- relative gap vs `IPF`
  - 从 13.27 的：
    - `3.76%`
  - 进一步降到：
    - `3.00%`
- absolute gap
  - 从：
    - `0.00477`
  - 再降到：
    - `0.00381`

也就是说：

- 当前主线还没有超过 `IPF`
- 但 remaining gap 已经继续被压缩
- 而且现在最重要的证据已经不是“它离 IPF 还差多少”
- 而是“这条主线仍在边界上持续变好”

### 13.29 把 current strongest mainline 继续推到 `3000 epochs` 后，non-oracle `c2f` 仍然变好，但边际收益开始收窄

13.28 之后，最直接的问题就变成了：

- 如果 `ep1800` 仍然有效
- 那把同一条 strongest mainline 继续推到更长的 training horizon
- 会不会继续逼近 `IPF`

这轮仍然没有改结构，也没有改任何 loss 或 selection 逻辑，只是继续延长训练：

- 固定：
  - `weighted epsilon, alpha = 0.5`
  - `detach_coarse_encoder = true`
  - `aux_t_gate = 50`
  - `selection_metric = val_tvd_coarse_head`
  - `logp_clip_quantile_low/high = 0.001 / 0.999`
- 把训练时长从：
  - `epochs = 1800`
  - 继续拉到：
  - `epochs = 3000`

对应 3 个 full-model run：

- seed0：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed0_20260326T162906Z/run_summary.json`
- seed1：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed1_20260326T163214Z/run_summary.json`
- seed2：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3000_seed2_20260326T163526Z/run_summary.json`

对应 seed-matched `c2f` eval：

- seed0：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_maskw_a05_seed0_stage2seed0_20260326T162906Z/metrics/coarse_to_fine_summary.json`
- seed1：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_maskw_a05_seed1_stage2seed1_20260326T163214Z/metrics/coarse_to_fine_summary.json`
- seed2：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_maskw_a05_seed2_stage2seed2_20260326T163526Z/metrics/coarse_to_fine_summary.json`

新的 true3 汇总：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_maskw_a05_seedmatched_true3_20260327T001000Z_summary.json`

#### 13.29.1 first-order 结果：`ep3000` 继续带来 clean 的 Stage1 改善

相对 `ep1800`：

- old `ep1800`：
  - `stage1_coarse_tvd_raw = 0.05825 ± 0.00058`
  - `stage1_coarse_tvd_ipf = 0.05768 ± 0.00034`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13080 ± 0.00147`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

- new `ep3000`：
  - `stage1_coarse_tvd_raw = 0.05613 ± 0.00035`
  - `stage1_coarse_tvd_ipf = 0.05660 ± 0.00036`
  - `pipeline_stage1_raw_tvd_joint = 0.13015 ± 0.00163`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.13011 ± 0.00156`
  - `oracle_stage2_tvd_joint = 0.11430 ± 0.00185`

因此：

- `Stage1` 仍然在继续变好
- non-oracle `c2f` 也继续下降
- `Stage2` 仍然完全没变

这说明：

- 新增收益依旧全部来自 `Stage1`
- current mainline 继续沿着正确方向在走

#### 13.29.2 但这次更重要的 insight 是：收益还在，但开始进入 late regime

这轮 full-model 的 best checkpoint epoch 分别是：

- seed0：
  - `best_epoch = 2600`
- seed1：
  - `best_epoch = 2800`
- seed2：
  - `best_epoch = 3000`

同时 full-model 自己的均值也继续变好：

- `one-shot = 0.14702 ± 0.00038`
- `tvd_coarse_head = 0.05613 ± 0.00035`

和 13.28 相比，这轮传达的信息不再是：

- “三条 run 都还卡在上界，还没训完”

而是：

- 两个 seed 的 best checkpoint 已经落在内部
- 只有一个 seed 仍在边界上

这说明：

- current mainline 还没有完全收敛
- 但已经不再是 13.28 那种“明显被 horizon 截断”的状态
- 它开始进入边际收益收窄的 late regime

#### 13.29.3 当前系统位置：离 `IPF` 还差 `0.00312`

对照：

- `gate50 + selcoarse + ep3000` mainline：
  - `0.13011 ± 0.00156`
- `5-way IPF`：
  - `0.12699`

因此：

- relative gap vs `IPF`
  - 从 13.28 的：
    - `3.00%`
  - 再降到：
    - `2.46%`
- absolute gap
  - 从：
    - `0.00381`
  - 再降到：
    - `0.00312`
- 相对 best one-shot baseline：
  - gain 扩大到：
    - `14.65%`

所以 current mainline 的最新状态可以准确表述为：

- 它还没有超过 `IPF`
- 但它继续稳定逼近 `IPF`
- 当前最大的信息增量，不再是“多训一定还能明显降很多”
- 而是“继续加 epoch 仍然有效，但收益已经开始进入收窄区间”

### 13.30 `seed2 ep3600` pilot：full-model 仍在继续变好，但 non-oracle `c2f` 已经接近平台区

13.29 之后，最值得回答的问题已经很窄了：

- 既然 `ep3000` 里只有 seed2 的 best checkpoint 还卡在上界
- 那么纯粹继续加 epoch
- 是否还能在 non-oracle `c2f` 上带来值得继续扩展到 true3 的收益

因此这轮没有再跑完整 `true3`，而是只做一个信息增益最高的 pilot：

- 固定 current strongest mainline 的全部设置不变
- 只把 seed2 从：
  - `epochs = 3000`
  - 延长到：
  - `epochs = 3600`

对应 run：

- full-model：
  - `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3600_seed2_20260327T001452Z/run_summary.json`
- c2f eval：
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600_maskw_a05_seed2_stage2seed2_20260327T001452Z/metrics/coarse_to_fine_summary.json`

#### 13.30.1 first-order 结果：teacher/full-model 还在继续变好

相对 seed2 `ep3000`：

- `one-shot = 0.14721 -> 0.14690`
- `tvd_coarse_head = 0.05609 -> 0.05601`
- `best_epoch = 3000 -> 3600`
- `best_val_tvd_coarse_head = 0.05879 -> 0.05816`

并且 validation coarse trace 仍然在继续下降：

- `3000: 0.058786`
- `3200: 0.058646`
- `3400: 0.058365`
- `3600: 0.058164`

所以就 full-model / Stage1 teacher 本身而言：

- 继续加 epoch 仍然有效
- 至少在 seed2 上，teacher 还没有完全收敛

#### 13.30.2 但真正重要的结果是：non-oracle `c2f` 的新增收益已经极小

相对 seed2 `ep3000`：

- `stage1_coarse_tvd_ipf = 0.0565986 -> 0.0566097`
  - 基本不变，甚至略差
- `pipeline_stage1_raw_tvd_joint = 0.1280733 -> 0.1280244`
  - 仅改善约：
    - `4.9e-05`
- `pipeline_stage1_coarse_ipf_tvd_joint = 0.1281040 -> 0.1280738`
  - 仅改善约：
    - `3.0e-05`
- `oracle_stage2_tvd_joint`
  - 完全不变

这意味着：

- teacher/full-model 的 coarse 指标还在慢慢变好
- 但这些微小改进已经几乎无法继续传导成 downstream `c2f` 的可观收益

#### 13.30.3 当前最重要的结论

这轮 pilot 的价值不在于数字本身，而在于它回答了一个关键策略问题：

- 纯粹继续把 epoch 往上加
  - 仍然能让 teacher 指标慢慢变好
- 但对当前 non-oracle `c2f`
  - 收益已经接近平台区

因此 current mainline 的最合理判断变成：

- `ep3000` 是有效推进
- 但再往上单纯加 epoch，不再是当前最高 ROI 的方向
- 如果下一步还想继续压那最后 `~0.003` 的 gap
  - 更值得做的已经不是更长 horizon
  - 而是新的结构性改动或更贴近 pipeline 目标的 selection / proxy

### 13.31 启动 `snapshot-enabled seed2 ep3600`：直接测 downstream best checkpoint，而不再只信 coarse-head proxy

13.30 之后，最关键的不确定性已经非常具体：

- 虽然 `val_tvd_coarse_head` 已经比 `val_tvd_joint_raw` 更对齐 `c2f`
- 但 late regime 里，teacher 指标的继续改善几乎不再传导到 downstream
- 这意味着：
  - 当前 best checkpoint 仍可能存在 proxy mismatch
  - 真正最优的 `c2f` checkpoint，不一定就是 coarse-head validation 最优点

所以这轮没有继续做更长 horizon，也没有切回别的分支，而是补了一个更贴近最终目标的实验基础设施：

- 在 full-model trainer 中加入：
  - `--save_eval_checkpoint_every`
- 使得每次 validation eval 都可以按固定 epoch 间隔保存 snapshot checkpoint
- 然后在同一条 current strongest mainline 上，重新启动：
  - `seed2 ep3600`
  - 但额外保存：
  - `epoch_0200, 0400, ..., 3600`

对应远端 run：

- `/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3600snap200_seed2_20260327T003137Z`

当前 launcher 已正常进入训练，前两个可见 epoch 为：

- `epoch=1`
  - `loss=4.263136`
- `epoch=200`
  - `loss=2.325306`

这轮实验要回答的问题不是“多训一点会不会再降一点”，而是：

- 在 current strongest mainline 的 late regime 里
- downstream non-oracle `c2f` 真正最优的 teacher checkpoint
- 是否仍然和 `val_tvd_coarse_head` 选出来的点一致

如果答案是否定的，那么下一步最有价值的改动就不是继续加 epoch，而是：

- 把 mainline checkpoint 选择进一步对齐到更真实的 pipeline proxy
- 甚至直接按 downstream sweep 结果来定义新的 selection target

### 13.32 `seed2 late-regime checkpoint sweep`：downstream best 确实不在 coarse-head best，但收益极小

13.31 的 snapshot run 完成后，最值得直接回答的问题是：

- 在 current strongest mainline 的 late regime
- `val_tvd_coarse_head` 选出来的 best checkpoint
- 是否真的等于 downstream non-oracle `c2f` 最优点

因此这轮没有再看 teacher summary，而是直接对：

- `epoch = 2400 / 2600 / 2800 / 3000 / 3200 / 3400 / 3600`

逐个跑真实 `c2f eval`。

对应 teacher run：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_v2_weighted_a05_detach_gate50_selcoarse_ep3600snap200_seed2_20260327T003137Z/run_summary.json`

对应 eval runs：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch2400_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch2600_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch2800_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch3000_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch3200_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch3400_20260327T003522Z/metrics/coarse_to_fine_summary.json`
- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3600snap200_maskw_a05_seed2_stage2seed2_epoch3600_20260327T003522Z/metrics/coarse_to_fine_summary.json`

最重要的结果不是整张 sweep，而是这三点：

- `epoch3000`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.1281040`
- `epoch3200`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.1280600`
- `epoch3600`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.1280738`

同时 teacher 选择端显示：

- `best_epoch = 3600`
- `selection_metric = val_tvd_coarse_head`

所以这轮回答非常清楚：

- downstream best 确实不在 coarse-head best
- 也就是说，late regime 里 residual selection mismatch 是真实存在的

但同样关键的是：

- 这个 mismatch 的量级只有：
  - 相对 `epoch3000`
    - `-4.39e-05`
  - 相对 `epoch3600`
    - `-1.38e-05`

因此这轮 sweep 的最终结论是：

- `pipeline-aware checkpoint selection`
  - 在 late regime 里仍然有残余收益
- 但这个收益太小
- 它已经不能解释 current mainline 离 `IPF` 的剩余 gap

### 13.33 `stage1ipfcond Stage2` pilot：把训练条件换成真实推理条件后，结果小幅变好

13.32 之后，最值得怀疑的残余问题变得更具体了：

- Stage2 训练时吃的是：
  - `true coarse`
- 但真正推理时吃的是：
  - `Stage1 prediction + coarse IPF`

这意味着：

- 当前 strongest Stage2
  - 仍然存在 train-test mismatch

因此这轮没有改 Stage2 网络，也没有改 loss，而是只改训练条件来源：

1. 用 current best downstream seed2 teacher：
   - `epoch3200`
2. 对全量 PUMA 生成：
   - `Stage1 predicted coarse + coarse IPF`
3. 用这个 conditioned coarse 重建 Stage2 teacher dataset
4. 用同一套 weighted Stage2 trainer 重新训练 seed2

对应 conditioned dataset：

- `/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/extc2f_full_earn_stage1ipfcond_pums_2023_puma_us_wide.csv`

对应 Stage2 run：

- `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcond_maskw_a05_seed2_20260327T005255Z/run_summary.json`

对应 end-to-end eval：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3200_stage1ipfcond_maskw_a05_seed2_stage2seed2_20260327T005716Z/metrics/coarse_to_fine_summary.json`

#### 13.33.1 teacher-forced Stage2：确实有小幅改善

相对旧的 weighted Stage2 seed2：

- `teacher_forced_stage2.tvd_joint_projected`
  - `0.2872320 -> 0.2869531`
- `inactive_mass_raw`
  - `0.0144124 -> 0.0141153`

这说明：

- 把 Stage2 的训练条件分布拉回到真实推理条件
- 确实能让 Stage2 本身更稳一点

#### 13.33.2 non-oracle end-to-end：改善真实传导到了 pipeline

固定同一个 `Stage1 epoch3200`：

- `stage1_coarse_tvd_ipf`
  - 完全不变：
  - `0.0565750 -> 0.0565750`
- `pipeline_stage1_raw_tvd_joint`
  - `0.1280187 -> 0.1278400`
- `pipeline_stage1_coarse_ipf_tvd_joint`
  - `0.1280600 -> 0.1278903`
- `oracle_stage2_tvd_joint`
  - `0.1118985 -> 0.1117982`

也就是说：

- 这次改善不是 Stage1 波动
- 而是 Stage2 的条件鲁棒性改进
- 并且它已经真实传导到了 non-oracle pipeline

#### 13.33.3 当前最重要的判断

这轮 pilot 的结论比 13.32 更有分量：

- `Stage2 train-test mismatch`
  - 是真实问题
- 把训练条件从 `true coarse` 拉回到 `Stage1 + coarse IPF`
  - 能带来可测的 end-to-end 改善

同时也要保持量级判断：

- 这次 gain 约：
  - `-1.70e-04`
- 它明显大于 13.32 的 checkpoint sweep gain
- 但仍然不是一枪过线的量级

因此 current mainline 的最新状态可以准确表述为：

- late-regime selection mismatch 真实存在，但已经不是主矛盾
- `Stage2 robustness to imperfect coarse` 也是有效方向
- 而且它比继续磨 late-regime checkpoint 更值得推进
- 但要稳定压过 `IPF`
  - 仍然需要把这条方向扩到多种子并继续放大增益

### 13.34 `stage1ipfcondmix Stage2`：把 `true coarse` 和 `predicted coarse` 混合训练后，单 seed 首次超过 `IPF`

13.33 之后，还剩一个更直接的问题：

- 如果 Stage2 只看一种 `predicted coarse` 误差模式
  - 它学到的鲁棒性可能还是太窄

因此这轮没有再改网络和 loss，而是把 Stage2 的条件分布进一步扩成混合支持：

- 对每个 `(PUMA, parent cell)`：
  - 保留一份 `true coarse`
  - 再追加一份 `Stage1 predicted coarse + coarse IPF`
- target 仍然是同一个 local fine split
- weighted epsilon loss 保持不变

对应 mixed dataset：

- `/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/extc2f_full_earn_stage1ipfcondmix_pums_2023_puma_us_wide.csv`

对应 Stage2 run：

- `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_maskw_a05_seed2_20260327T011422Z/run_summary.json`

对应 end-to-end eval：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3200_stage1ipfcondmix_maskw_a05_seed2_stage2seed2_20260327T012200Z/metrics/coarse_to_fine_summary.json`

#### 13.34.1 teacher-forced Stage2：gain 明显放大

相对旧的 weighted Stage2 seed2：

- `teacher_forced_stage2.tvd_joint_projected`
  - `0.2872320 -> 0.2838438`
- `inactive_mass_raw`
  - `0.0144124 -> 0.0143309`

相对 13.33 的单一 `stage1ipfcond`：

- `teacher_forced_stage2.tvd_joint_projected`
  - `0.2869531 -> 0.2838438`

也就是说：

- 真正有效的不是“完全用 predicted coarse 替代 true coarse”
- 而是让 Stage2 同时暴露在：
  - clean condition
  - imperfect condition

#### 13.34.2 non-oracle end-to-end：seed2 首次压到 `IPF` 以下

固定同一个 `Stage1 epoch3200`：

- `stage1_coarse_tvd_ipf`
  - 完全不变：
  - `0.0565750 -> 0.0565750`
- `pipeline_stage1_raw_tvd_joint`
  - `0.1280187 -> 0.1255749`
- `pipeline_stage1_coarse_ipf_tvd_joint`
  - `0.1280600 -> 0.1257494`
- `oracle_stage2_tvd_joint`
  - `0.1118985 -> 0.1097530`

同时：

- `IPF`
  - `0.1269883`

这意味着：

- 在保持同一个 Stage1 不变的前提下
- mixed-condition Stage2 单独就把 seed2 的 non-oracle pipeline
  - 从 `IPF + 9.02e-04`
  - 推到了 `IPF - 1.24e-03`

#### 13.34.3 这轮结果揭示了什么

这轮给出的新结论是结构性的：

- residual gap 不只是 `Stage2` 看没看过 predicted coarse
- 更关键的是：
  - Stage2 需要在训练时同时保留 clean 与 noisy coarse 条件
  - 才能学到足够宽的 refinement support

所以当前最值得推进的主线，已经从：

- `stage1ipfcond`

进一步收敛到了：

- `stage1ipfcondmix`

而且这次不是边角收益：

- 相对 seed2 baseline
  - `pipeline_stage1_coarse_ipf_tvd_joint`
    - `-2.31e-03`

这个量级已经足以让单 seed 真正过线。

#### 13.34.4 下一步

为了判断这是不是主线级突破，而不是 seed2 特例，我已经在 `wsa` 上启动了 `seed0/seed1` 的同口径批任务：

- batch log：
  - `/home/jinlin/projects/Synthetic_City/outputs/_stage2_mix_seed01_batch_20260327T0138Z.log`
- seed0 mixed dataset dir：
  - `/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed0_stage1ipfcondmix`
- seed1 mixed dataset dir：
  - `/home/jinlin/data/geoexplicit_data/synthetic_city/data/us/processed/external_c2f/mainline_gate50_selcoarse_ep3000_seed1_stage1ipfcondmix`

当前 batch 已确认在正常推进：

- 进程：
  - `/tmp/run_c2f_stage2_mix_seed01.sh`
- 当前阶段：
  - 正在构建 `seed0` 的 mixed-condition teacher dataset

### 13.35 `stage1ipfcondmix` 扩到多 seed：均值已略低于 `IPF`，但还没有拉开余量

13.34 之后，我把同一条 `mixed-condition Stage2` 主线继续扩到了 `seed0/seed1`。

对应新增 runs：

- `seed0 teacher`
  - `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed0_maskw_a05_20260327T013800Z/run_summary.json`
- `seed0 eval`
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_stage1ipfcondmix_maskw_a05_seed0_stage2seed0_20260327T013800Z/metrics/coarse_to_fine_summary.json`
- `seed1 teacher`
  - `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed1_maskw_a05_20260327T013801Z/run_summary.json`
- `seed1 eval`
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_stage1ipfcondmix_maskw_a05_seed1_stage2seed1_20260327T013801Z/metrics/coarse_to_fine_summary.json`

#### 13.35.1 seed-level non-oracle 结果

- `seed0`
  - baseline:
    - `0.1303026`
  - mixed:
    - `0.1285491`
  - delta:
    - `-0.0017535`
- `seed1`
  - baseline:
    - `0.1319123`
  - mixed:
    - `0.1264879`
  - delta:
    - `-0.0054245`
- `seed2`
  - baseline:
    - `0.1281040`
  - mixed:
    - `0.1257494`
  - delta:
    - `-0.0023545`

因此当前 best-available 3-run aggregate 已经变成：

- `pipeline_stage1_coarse_ipf_tvd_joint`
  - `0.1269288 ± 0.0011847`

相对 `IPF = 0.1269883`：

- 均值低了：
  - `5.95e-05`

这轮最准确的判断是：

- `mixed-condition Stage2`
  - 已经把主线从“明显落后 IPF”
  - 推到了“均值略低于 IPF”
- 但它还不能支持：
  - “显著稳定优于 IPF”

因为：

- 只有 `2/3` seed 真正过线
- `seed0` 仍高于 `IPF`
  - `+0.0015608`

#### 13.35.2 为什么当前 residual gap 更像 Stage2 seed variance

mixed 之后三组的

- `pipeline_stage1_coarse_ipf_tvd_joint - oracle_stage2_tvd_joint`

几乎是常数：

- `seed0`
  - `0.1285491 - 0.1125498 = 0.0159993`
- `seed1`
  - `0.1264879 - 0.1103983 = 0.0160896`
- `seed2`
  - `0.1257494 - 0.1097530 = 0.0159965`

这说明：

- train-test mismatch 基本已经被 mixed-condition 吸收了
- 当前 residual gap 更像：
  - `Stage2 oracle quality` 的 seed 方差
- 也就是说：
  - `seed0` 没过线
  - 更像是它自己的 Stage2 还不够强
  - 而不是 Stage1 特别差

### 13.36 `seed0 ns256`：更稳的 Stage2 推理有帮助，但不足以单独把 `seed0` 推过线

为了先判断 residual gap 里有多少来自推理方差，我先没有重训模型，而是只把 `seed0 mixed Stage2` 的 eval sampling 从：

- `n_eval_joint_samples = 64`

提到：

- `256`

对应 run：

- `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_stage1ipfcondmix_maskw_a05_seed0_stage2seed0_ns256_20260327T020000Z/metrics/coarse_to_fine_summary.json`

结果：

- `seed0 pipeline_stage1_coarse_ipf_tvd_joint`
  - `0.1285491 -> 0.1278349`
- delta:
  - `-7.1422e-04`

这说明：

- 更稳的 Monte Carlo averaging
  - 确实有帮助
- 但帮助还不够大

因为：

- `seed0` 仍高于 `IPF`
  - `0.1278349 - 0.1269883 = 8.47e-04`

因此 13.36 的结论是：

- 推理更稳是正确方向
- 但它不是终解
- 要把 `seed0` 也稳稳压过去
  - 仍需要继续降低 `Stage2` 本体的 oracle error

### 13.37 `seed0 bestsel50`：开始测试“Stage2 best checkpoint selection”是否能继续压低 oracle

既然当前 residual 更像：

- `Stage2 oracle quality` 的 seed 方差

那么比继续改 Stage1 更直接的一步是：

- 不只保留 `final.pt`
- 而是在 Stage2 训练中周期评估
- 按 `teacher_forced_stage2.tvd_joint_projected` 保存 `best.pt`

我已经给 Stage2 trainer 加了：

- `--save_best_model`
- `--eval_every_epochs`

当前正在跑：

- `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed0_maskw_a05_bestsel50_20260327T020700Z`

配套日志：

- `/home/jinlin/projects/Synthetic_City/outputs/_seed0_mix_bestsel50_20260327T020700Z.log`

当前已经看到的中间评估：

- `epoch=50`
  - `tvd_joint_projected = 0.289910`
- `epoch=100`
  - `tvd_joint_projected = 0.285574`

这说明：

- bestsel 逻辑已经正常工作
- 但到 `epoch100` 为止
  - 还没有明显超过当前 mixed final seed0 的 `0.2855256`

因此这条线仍在继续观察中。

### 13.38 `Stage2 clean-head + diffusion`：`seed0` 已经明显低于 `IPF`

既然：

- `bestsel50` 这条线增益有限
- `diffusion` 又不能推翻

那么更合理的结构改动不是替换 diffusion，而是：

- 保留 diffusion 作为主生成分支
- 额外给 `Stage2` 加一个 `clean local head`
- 让它作为 noising-free anchor
- 再用 `head <-> diffusion consistency` 稳定条件表征

本次采用的配置是：

- `clean_head_weight = 1.0`
- `consistency_weight = 0.5`
- `aux_t_gate = 50`
- `predict_mode = blend`
- `blend_alpha = 0.25`
- 其余仍保持
  - `mixed-condition Stage2`
  - `cell-weighted epsilon loss`

对应训练结果：

- teacher run:
  - `outputs/_us_puma_external_c2f_full_earn_teacher_stage1ipfcondmix_mainline_gate50_selcoarse_ep3000_seed0_maskw_a05_cleanheadw1_cons05_gate50_blend25_bestsel50_20260327T023511Z/run_summary.json`
- eval run:
  - `outputs/_us_puma_external_c2f_full_earn_eval_mainline_gate50_selcoarse_ep3000_stage1ipfcondmix_cleanheadw1_cons05_gate50_blend25_maskw_a05_seed0_stage2seed0_20260327T023511Z/metrics/coarse_to_fine_summary.json`

关键数字：

- `seed0 old mixed`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.1285491`
- `seed0 new clean-head+diffusion`
  - `pipeline_stage1_coarse_ipf_tvd_joint = 0.1211049`
- `IPF`
  - `0.1269883`

delta：

- 相对旧 `seed0 mixed`
  - `-0.0074442`
- 相对 `IPF`
  - `-0.0058834`

这说明：

- 这次不是擦线
- 而是 `seed0` 已经被明显压到 `IPF` 以下

同时它的 `oracle_stage2_tvd_joint` 也到了：

- `0.1038834`

而对应 teacher-forced best checkpoint 是：

- `best_projected_tvd_joint = 0.2690819`
- `best_epoch = 250`

因此 13.38 的结论是：

- `Stage2 clean-head + diffusion` 是当前最有效的新方向
- 它没有推翻 diffusion
- 而是在保留 diffusion 主线的前提下
  - 明显降低了 `Stage2` error
  - 并把最难的 `seed0` 直接推到 `IPF` 以下

### 13.39 `seed1/seed2 clean-head + diffusion`：已启动新的顺序批任务

为了判断这个结构改动是否能把 `true3` 均值也明显拉开，我已经按相同配置继续扩到：

- `seed1`
  - 仍使用 `ep3000` 的最强 `Stage1`
- `seed2`
  - 仍使用当前最强的 `epoch3200` `Stage1` snapshot

当前批任务：

- batch log:
  - `/home/jinlin/projects/Synthetic_City/outputs/_stage2_cleanhead_seed12_batch_20260327T040359Z.log`

当前状态：

- `seed1` 训练已经正常进入
- 后续会自动顺序执行：
  - `seed1 teacher`
  - `seed1 eval`
  - `seed2 teacher`
  - `seed2 eval`

因此下一步最关键的判断已经很清楚：

- 不是再看中间指标
- 而是等 `seed1/seed2` 跑完后
  - 直接计算新的 `true3 mean±std`
  - 看它是否能稳定、明确地低于 `IPF = 0.1269883`

### 14.1 `full_income v2 seed0 c2f`：`best.pt` 与更稳推理都是真收益

在 `full_income v2 seed0 c2f` 第一次完整跑通后，原始 end-to-end 结果是：

- `pipeline_stage1_coarse_ipf.tvd_joint = 0.1773901`
- 对应 `IPF = 0.1757722`

这说明：

- `c2f` 已经把 `seed0` 从 one-shot 的极差工作区间拉回到 `IPF` 附近
- 但第一次结果还没有真正过线

我随后只做了两步低成本重评估，不改训练本体：

1. 把 `Stage2 checkpoint` 从 `final.pt` 改成 `best.pt`
2. 在此基础上把 `Stage2` 的 eval sampling 从 `64` 提到 `256`

对应结果是：

- 原始 `final.pt + ns64`
  - `0.1773901`
- `best.pt + ns64`
  - `0.1762777`
- `best.pt + ns256`
  - `0.1755801`

对照：

- `IPF = 0.1757722`
- `one-shot = 0.7826781`

delta：

- 相对原始 `final.pt + ns64`
  - `-0.0018100`
- 相对 `IPF`
  - `-0.0001921`

这一步的结论是：

- `teacher best checkpoint` 不是边角细节，而是当前 `full_income c2f` 的必要选择
- 更稳的 `Stage2` sampling 也不是纯噪声处理，而是足以把 `seed0` 从略高于 `IPF` 推到略低于 `IPF`
- `full_income` 当前 strongest `seed0` 配方应当更新为：
  - `Stage1: full_income v2 ep1200 strongest full-model`
  - `Stage2: clean-head + diffusion`
  - `Stage2 checkpoint: best.pt`
  - `Stage2 eval samples: 256`

同时，`oracle_stage2_true_coarse.tvd_joint` 也同步从：

- `0.1565911`

降到：

- `0.1551003`

这说明这次收益并不只是 pipeline 偶然抖动，而是 `Stage2` 本体在当前配置下确实更强。

### 14.2 `full_income v2 seed1 Stage1`：已按 strongest recipe 启动

当前 `full_income v2` 还没有 `seed1/seed2` 的对应资产，因此下一步最有信息量的动作不是继续抠 `seed0`，而是直接扩到第二个 seed。

我已在 `WSA` 上启动：

- `tmux session`
  - `full_income_v2_seed1_ep1200`

对应 run：

- `/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_income_v2_weighted_a05_detach_gate50_selcoarse_masked_ep1200_seed1_20260328T092900Z`

配置与当前 `seed0` strongest one-shot 完全对齐：

- `weighted_a05`
- `detach`
- `gate50`
- `selection_metric = val_tvd_coarse_head`
- `support_mask_mode = dataset_nonzero`
- `epochs = 1200`

当前已确认正常进入训练，首轮日志为：

- `epoch=1 loss=4.730572`

因此现在 `full_income` 主线的判断已经清楚：

- `seed0` 上，`c2f` 已经被推到略低于 `IPF`
- 下一步不再是继续改 `seed0`
- 而是尽快拿到 `seed1`，判断这条 strongest recipe 是否具备跨 seed 稳定性

### 14.3 `full_income PUMA` 主线继续推进：`seed1 c2f` 已转入 teacher，`seed2` 已补齐并接上 `c2f`

这轮推进的目标不是再改配方，而是把 `PUMA` 主线从单个 `seed0` 扩成真正的多 seed 验证链。

首先，`seed1 c2f` 已经不再停留在排队状态，而是实质进入了 `Stage2 teacher` 训练。当前链路状态为：

- `builder` 已完成，产物已写出：
  - `/home/jinlin/projects/Synthetic_City/data/us/processed/external_c2f/full_income_v2_seed1/extc2f_full_income_v2_stage1ipfcondmix_pums_2023_puma_us_wide.csv`
- `teacher` 已开始训练，当前首个有效评估点为：
  - `epoch=50`
  - `new_best_tvd_joint_projected=0.282351`
- `eval` 会话已就位，等待 `teacher best.pt` 后自动接续

这一步的意义不是数值本身，而是：

- `seed1 c2f` 全链已经真正跑通到 `Stage2` 学习阶段
- 当前不再存在“只证明 `seed0` 单点可行”的问题

其次，`seed2 Stage1` 也已经补齐完成，结果进入和 `seed0/seed1` 同一工作区间：

- run:
  - `/home/jinlin/projects/Synthetic_City/outputs/_us_puma_external_joint_hier_diffusion_full_income_v2_weighted_a05_detach_gate50_selcoarse_masked_ep1200_seed2_20260328T105100Z`
- `tvd_joint = 0.7871428`
- `tvd_coarse_head = 0.0820989`
- `IPF = 0.1757580`
- `best_epoch = 1200`

对照前两个 seed：

- `seed0 tvd_coarse_head = 0.0790298`
- `seed1 tvd_coarse_head = 0.0797947`
- `seed2 tvd_coarse_head = 0.0820989`

这说明：

- `seed2` 不是坏种子
- 它的 `Stage1 coarse` 质量已经足够接入 `c2f`
- 因此继续向 `seed2 c2f` 推进是有信息量的，不是盲目铺实验

基于这个判断，我已经把 `seed2 c2f` 三段会话也排上：

- `full_income_v2_c2f_build_seed2`
- `full_income_v2_c2f_teacher_seed2`
- `full_income_v2_c2f_eval_seed2`

其中 `build_seed2` 已确认真实启动，占用第二个轻量 Python 进程；也就是说当前 GPU/远端资源利用已经从“单任务串行”推进到了：

- `seed1 teacher`
- `seed2 c2f build`

同时并行推进。

所以这轮的结论很清楚：

- 是有效推进，而且不是表面推进
- `seed1 c2f` 已经进入真正会决定成败的 `Stage2` 阶段
- `seed2` 也已经不再停在 `Stage1`，而是正式接入 `c2f` 主线

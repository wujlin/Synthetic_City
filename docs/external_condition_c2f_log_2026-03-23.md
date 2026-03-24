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

# 5-way Full-Joint Diffusion Review Note

## 1. 当前要 review 的对象是什么

当前这条线要解决的对象是：

\[
p(\text{AGEP\_bin}, \text{SEX}, \text{SCHL\_allpop}, \text{ESR\_allpop}, \text{EARN\_16p\_bin} \mid \text{region})
\]

这里的五个变量分别是：

- `AGEP_bin`
- `SEX`
- `SCHL_allpop`
- `ESR_allpop`
- `EARN_16p_bin`

其中：

- `EARN_16p_bin` 不是旧的 `PINCP`
- external 端来自 `ACS B20001`
- target 端来自 `PUMS PERNP`

所以这条 `5-way` 线已经不是“income proxy 对不齐”的问题，而是一个真正的 `5-variable full joint` 问题。

---

## 2. 相关代码路径

### 2.1 earnings schema

- `tools/external_earn_v1_schema.py`

这个文件定义：

- `EARN_LABELS`
- `B20001` 到 coarse bins 的分组
- `bin_earn_allpop(age, earn)`

当前 full earnings bins 是 6 类：

1. `not_in_earnings_universe`
2. `lt_25k`
3. `25k_50k`
4. `50k_75k`
5. `75k_100k`
6. `ge_100k`

其中 `not_in_earnings_universe` 的定义在代码里非常具体：

- `age >= 16`
- `earn > 0`

这两个条件同时满足，才会落入 5 个 earnings bins；否则落到第 0 类。

### 2.2 external condition 构造

- `tools/build_external_condition_earn_v1_acs_puma.py`
- `tools/merge_external_condition_v1_with_earn.py`
- `tools/train_us_puma_external_v1_diffusion.py`

这三步的分工是：

1. `build_external_condition_earn_v1_acs_puma.py`
   - 从 ACS API 拉 `B20001` 和 `B01001`
   - 生成 PUMA-level `EARN_16p_bin`
   - 关键逻辑是：
     - `not_in_earnings_universe = total population - 16+ with earnings`

2. `merge_external_condition_v1_with_earn.py`
   - 把 base 4-attr condition 和 earnings condition 拼起来
   - 最终得到：
     - `extcond_v1_earn_v1_acs5_2022_puma_us.csv`

3. `train_us_puma_external_v1_diffusion.py`
   - `_load_external_condition_matrix()`
   - 把 long-format condition 按 `variable -> category` 展成 `cond_raw`
   - 当前 `5-way` 的 `cond_raw_dim = 28`

当前 28 维 block 是：

- age: `10`
- sex: `2`
- education: `5`
- employment: `5`
- earnings: `6`

### 2.3 conditional earnings target

- `tools/build_external_target_earn_conditional_v1_us.py`

这个文件构造的是：

\[
p(\text{EARN\_16p\_bin} \mid \text{AGEP\_bin}, \text{SEX}, \text{SCHL\_allpop}, \text{ESR\_allpop}, \text{PUMA})
\]

也就是说，它先按 4-attr cell 聚合 PUMS，再在每个 cell 内统计 earnings 条件分布。

### 2.4 5-way full target

- `tools/build_external_target_v1_full_earn.py`

这个文件不是直接从原始 PUMS 一次性数出 3000-cell table，而是：

\[
p(\text{5-way} \mid \text{region})
=
p(\text{4-attr} \mid \text{region})
\cdot
p(\text{earn} \mid \text{4-attr}, \text{region})
\]

代码里最关键的地方是：

- `tab[ai, si, qi, ei, :] += cell_prob * p_earn`

这一步把：

- `cell_prob`
- `p_earn`

乘起来，展开成 5-way `tab`。

这意味着：

- 当前 `5-way target` 是从同一个 PUMS 源精确展开出来的
- 不是新的近似 target
- 但它的正确性依赖于：
  - 4-attr cell indexing 没错
  - `p_earn` 条件分布没错

### 2.5 5-way trainer specialization

- `tools/train_external_joint_hier_diffusion_full_earn.py`

这个文件做两件事：

1. 把 base trainer 的 fine shape 改成：
   - `(10, 2, 5, 5, 6)`
   - `K = 3000`

2. 把 coarse auxiliary 改成：
   - `(4, 2, 3, 3, 4)`
   - `K = 288`

这里最值得 review 的是：

- `EARN_LITE_LABELS`
- `EARN_TO_LITE`
- `_build_fine_to_coarse_matrix_full_earn()`

因为它决定了：

- 6 个 full earnings bins
- 如何折叠到 4 个 coarse earnings bins

### 2.6 训练主逻辑

- `tools/train_external_joint_hier_diffusion_full.py`

这是最核心的文件。主要看 5 个地方：

1. `SharedLatentHierarchicalDiffusion`
2. `step()`
3. `sample_latent_conditioned()`
4. `_evaluate_joint_distribution()`
5. `main()`

### 2.7 运行与汇总

- `tools/run_external_joint_hier_diffusion_full_earn.sh`
- `tools/run_external_joint_hier_diffusion_full_earn_batch.sh`
- `tools/summarize_external_joint_hier_diffusion_runs.py`

这些文件决定：

- 当前默认超参
- batch 输出目录
- summary 里到底汇总哪些指标

---

## 3. 计算逻辑

### 3.1 condition 是怎么进模型的

`condition_csv` 是 long-format 的：

- `puma_uid`
- `variable`
- `category`
- `target`

`_load_external_condition_matrix()` 会把它拼成：

- `cond_raw.shape = (n_puma, 28)`

block 顺序来自 schema 的 `variable_order`。当前 `5-way` 情况下就是：

1. `AGEP_bin`
2. `SEX`
3. `SCHL_allpop`
4. `ESR_allpop`
5. `EARN_16p_bin`

### 3.2 target 是怎么进模型的

`joint_wide_csv` 里有：

- `p_joint_000 ... p_joint_2999`

`_load_joint_wide()` 读进来之后，会归一化成：

- `p_fine.shape = (n_puma, 3000)`

然后再乘上 fine-to-coarse mapping，得到：

- `p_coarse.shape = (n_puma, 288)`

### 3.3 模型内部 forward 逻辑

`SharedLatentHierarchicalDiffusion` 的 forward 没有单独写成 `forward()`，实际逻辑分散在 `step()`、`predict_coarse()` 和 `sample_latent_conditioned()` 里。

核心链条是：

1. `cond_raw -> encoder -> z`
2. `z -> coarse_feature -> coarse_out -> coarse_prob`
3. `x_t, t, z -> denoiser -> eps_pred`
4. `eps_pred + x_t -> x0_pred`
5. `x0_pred -> denormalized logp -> softmax -> fine_prob`

也就是说，当前 fine branch 还是标准的 continuous DDPM：

- 训练目标是 `epsilon MSE`
- 最终 joint 不是直接输出的概率，而是
  - 先输出 `x0_pred`
  - 再反标准化成 `logp`
  - 再做 `softmax`

### 3.4 当前 loss 由哪几项组成

在 `step()` 里，当前总损失是：

\[
L =
L_{diff}
+
\lambda_c L_{coarse}
+
\lambda_{cons} L_{cons}
+
\lambda_m L_{marg}
\]

具体对应：

1. `loss_diffusion`
   - `eps_pred` 对 `noise` 的 MSE

2. `loss_coarse`
   - coarse head 对 `p_coarse_true` 的交叉熵

3. `loss_consistency`
   - `fine_prob @ agg_mat`
   - 与 `coarse_prob`
   - 做 TVD 风格一致性约束

4. `loss_marginal`
   - 从 `fine_prob` 上边缘化出 5 个 block marginals
   - 与 `cond_raw` 对应 block 做 TVD 风格约束

### 3.5 当前采样和评估逻辑

`_evaluate_joint_distribution()` 的顺序非常关键：

1. 用 `predict_coarse()` 先算 coarse head
2. 用 `encoder(cond_eval)` 得到 `z_eval`
3. 用 `sample_latent_conditioned()` 采 `x_samples`
4. `x_samples -> denormalized logp`
5. `logp -> softmax -> fine_pred_raw`
6. 再对 `fine_pred_raw` 做 IPF，得到 `fine_pred`

所以当前始终有两套指标：

- `raw`
- `post-IPF`

这点很重要，因为当前最明显的问题就在：

- coarse head 好
- post-IPF 指标还能被拉回一部分
- 但 `raw` 经常很差

---

## 4. 当前最值得和 PI 一起查的点

### 4.1 5-way target 的定义是不是你们真正想要的 target

当前 `5-way target` 不是直接从原始 PUMS 一次性数 `3000-cell` 频率，而是：

- 先构 4-attr joint
- 再乘 conditional earnings

这件事本身没错，但要确认两件事：

1. 这是不是你们接受的 target 定义
2. 这一步是否在数值上等价于直接从 PUMS 聚合 5-way counts

如果 PI 想查“这里会不会引入额外误差”，这个文件最关键：

- `tools/build_external_target_v1_full_earn.py`

### 4.2 fine-to-coarse mapping 是否合理

当前 `5-way` 训练稳定器很依赖：

- `_build_fine_to_coarse_matrix_full_earn()`

这里的关键问题是：

- 为什么 earnings lite 是 4 档，而不是 3 档或 2 档
- `lt_25k` 和 `25k_50k` 都并到 `lt_50k`
- `50k_75k` 和 `75k_100k` 都并到 `50k_100k`

这一步如果 coarse 设计不合理，会直接影响：

- coarse consistency 能不能真正约束新增 earnings 轴

### 4.3 当前 drift 最像发生在 `x0_pred -> logp -> softmax`

现在最明显的现象是：

- coarse head 越训越好
- 但 raw fine sample 越训越坏

这说明 drift 更像发生在：

1. `x0_pred` 的数值尺度
2. 反标准化之后的 `logp`
3. `softmax` 对极端 logp 的放大

如果 PI 要查“为什么 raw 会整体漂”，最值得看的是：

- `tools/train_external_joint_hier_diffusion_full.py`
  - `step()`
  - `sample_latent_conditioned()`
  - `_evaluate_joint_distribution()`

### 4.4 当前 best checkpoint 的选择口径本身会影响结论

原始版本只按：

- `val_tvd_joint`

选 best checkpoint。后来为了抑制 raw drift，才新增：

- `val_tvd_joint_raw`
- `val_combo`

所以如果 PI 要查：

- 之前为什么会“训得越久越坏”
- 是不是只是 checkpoint 选错了

那就要一起看：

- `selection_metric`
- `eval_every`
- `best_epoch`

### 4.5 当前 5-way 失败，不是 coarse state 学不到

这个是一个很重要的排除结论：

- 旧版 5-way diffusion:
  - `tvd_joint ≈ 0.5808`
  - `tvd_coarse_head ≈ 0.0394`

这说明：

- `regional context`
- coarse branch

其实没坏。真正坏的是：

- 3000-cell raw diffusion sample

所以不要把问题归咎成：

- condition 不够
- earnings 定义不对
- encoder 不会

---

## 5. 截至当前最有用的实验信号

### 5.1 旧版 5-way diffusion 是明显失败的

文件：

- `outputs/_us_puma_external_joint_hier_diffusion_full_earn_batch_.json`

结果：

- `tvd_joint = 0.58083 ± 0.00446`
- `5-way IPF = 0.12699`

但 coarse 没坏：

- `tvd_coarse_head = 0.03938 ± 0.00021`

### 5.2 v2 修复方向是有效的

v2 做了两件事：

1. `72-cell` coarse head 扩成 `288-cell`
2. 加 `marginal consistency loss`

single-seed pilot 从：

- `0.57877 -> 0.52594`

说明训练口径修复本身是对症的。

### 5.3 加宽 denoiser 继续有效，但还不够

`v2 + diffusion_hidden_dims=1024,1024`

single-seed 到了：

- `tvd_joint = 0.43957`

说明容量是下一阶段瓶颈，但这条线还没接近可用。

### 5.4 单纯继续加长训练没有解决问题

`wide1000` 的 3 seeds：

- `tvd_joint = 0.48580 ± 0.03288`

而且三个 seed 的 `best_epoch` 都是 `400`。  
这说明：

- 继续 `400 -> 1000`
- 不是当前主矛盾

### 5.5 当前最强的新信号来自 raw-aware selection + logp clipping

single-seed `rawclip`：

- `tvd_joint = 0.15583`
- `tvd_joint_raw = 0.24401`

相对于 `v2 + wide` 的 `0.43957`，这是非常大的改善。  
这说明当前问题确实和：

- raw sample drift
- 极端 `logp` 被 `softmax` 放大

直接相关。

---

## 6. 建议 review 顺序

如果你和 PI 要一起 review，我建议按这个顺序走：

1. `tools/build_external_target_v1_full_earn.py`
   - 先确认 5-way target 的定义是不是你们想要的

2. `tools/train_external_joint_hier_diffusion_full_earn.py`
   - 再确认 fine-to-coarse 映射和 coarse shape 合不合理

3. `tools/train_external_joint_hier_diffusion_full.py`
   - 重点看：
     - `step()`
     - `sample_latent_conditioned()`
     - `_evaluate_joint_distribution()`

4. `tools/run_external_joint_hier_diffusion_full_earn.sh`
   - 最后核当前超参、selection、clipping 是否和结果一致

这个顺序的好处是：

- 先查 target 定义
- 再查训练约束
- 最后查采样与选权重

不容易一上来就陷进 DDPM 细节里。

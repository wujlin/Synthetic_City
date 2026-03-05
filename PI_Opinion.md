给 Partner 的三个实验设计
实验 1: Output Example 可视化
目标：让读者直观看到模型输出是什么——一个概率向量，和真实分布的对比。

选择标准：从 Michigan test set（leave-MI-out split）挑 4 个 PUMA，覆盖人口类型光谱：

A 大学城（如 Ann Arbor 区域）：高 TVD-to-global，年轻 + 低收入 + 高教育
B 制造业走廊（如 Flint/Detroit 工业区）：壮年 + 收入极化
C 退休/农村（如 Upper Peninsula）：老龄 + 中等收入
D 典型郊区（TVD 近中位数）：接近国家平均
选法：按 TVD-to-global 的分位数（>90th, ~75th, ~50th, ~25th）筛选，再用人口学特征确认。

**具体操作**：
1. 加载 leave-MI-out 实验的 Michigan 68 PUMAs 的 TVD-to-global 值（已有数据，来自 fig1_heterogeneity 的计算脚本）
2. 按 TVD 分位数筛选 4 个 PUMA GEOID
3. 对每个 PUMA，加载 true joint p（32 cells）和 diffusion output p̂（从已训练的 5-var K=32 pairwise model 推理，128 draws average + post-hoc IPF）
4. 用 PUMA 的 marginal profile（5 个属性的 1D 分布）确认其人口类型标签

计算内容（5-var K=32, pairwise-conditioned）：

True joint p（32 cells）
Diffusion output p̂（128 draws average + post-hoc IPF）
Cell-wise residual p̂_k − p_k
Per-PUMA TVD

可视化：2×4 subplot grid

Top row (a–d)：paired bar chart，32 cells 按 true probability 降序排列，blue=true, orange=diffusion
Bottom row (e–h)：residual plot，同 x 轴，y = p̂_k − p_k，虚线标 ±0.01
每个子图标题：PUMA 名 + 人口类型 + TVD 值
输出文件：figures/fig_output_examples.pdf

**代码入口**：基于现有推理脚本（sample 阶段），只需加载已有 model checkpoint 和 Michigan test set，跑推理 + 可视化。不需要重新训练。

---

实验 2: 训练收敛性
目标：证明 10,000 epochs 的训练预算足够，结果不是 early stopping 或 overfitting 的产物。

操作（5-var K=32, pairwise）：

在 epoch = [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000] 保存 checkpoint
每个 checkpoint 跑 Michigan 68 PUMAs 推理（128 draws, average, post-hoc IPF）
计算 mean TVD ± std across 68 PUMAs
可视化：mean TVD vs. epoch（带 ±1 std 阴影带），标注 plateau 区域和论文使用的 training budget

**代码入口**：修改训练脚本，在指定 epoch list 保存 checkpoint（加一个 `--save_epochs` 参数）。推理脚本保持不变，循环加载各 checkpoint 即可。

**注意**：这需要重新跑一次训练（约 10k epochs），因为之前没有保存中间 checkpoint。

---

实验 3: Monte Carlo 稳定性
目标：证明 128 draws averaging 足够，输出可复现。

操作（同一模型，即 5-var K=32 pairwise 的最终 checkpoint）：

Draw count ∈ [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]，每个 count 计算 mean TVD across 68 MI PUMAs
固定 128 draws，用 10 个不同 random seed 重复，计算 TVD 的 run-to-run std
可视化：TVD vs. draw count（log-scale x 轴），标注 128-draw 使用点，error bar = ±1 std across 10 seeds

**代码入口**：修改推理脚本的 `--n_samples` 参数，循环跑不同 draw count。seed 变化通过 `--seed` 参数控制。不需要重新训练。

---

fig_verification.pdf 的 Panel 分配

将上述三个实验的结果合成一张 3-panel 图：

- Panel (a)：Marginal consistency — 5 个属性的 raw marginal TVD bar chart（数据已有，来自现有实验的 raw output，不需要新实验）
- Panel (b)：Training convergence — mean TVD vs. epoch（实验 2 的输出）
- Panel (c)：Monte Carlo stability — TVD vs. draw count（实验 3 的输出）

布局：1×3 横排，共享 y 轴标签 "TVD"。

---

Figure 1 拆分

当前 fig1_heterogeneity.pdf 是 4-panel 合一图。拆分为：

- **Fig 1**（放在 Introduction 正文中）：只放 panel (a) choropleth 地图，单独成图 → `figures/fig1_map.pdf`
- **Panels (b,c,d)** → 移到 SI 作为补充图 → `figures/fig_S2_heterogeneity_stats.pdf`

Partner 需要做的：把原始绑图脚本拆成两个输出文件。

---

Methods 框架图设计
现有 schematic 只展示 "IPF vs. diffusion" 两管线对比。新的研究框架图应包含完整管线：

ACS PUMS → 分箱 + Laplace smoothing → 联合概率表 p
    → log(p+ε) → z-score → z₀
        → DDPM forward (加噪) + condition c 注入 → ε-prediction loss (训练)
        → reverse denoising → inverse z-score → softmax → p_raw (推理)
            → 128 draws average → post-hoc IPF → final p̂
                → 与 ground truth 比较 → TVD

条件分支：published marginals / cross-tabs → condition vector c → concat at denoiser input

当前 schematic 可以作为子图嵌入（作为 IPF baseline 对比部分），但整体框架图需要展示从数据到评估的完整链路。建议 partner 用 draw.io 或 TikZ 制作。文件名：figures/fig_framework.pdf
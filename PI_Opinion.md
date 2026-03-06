给 Partner 的实验与图表设计

---

## 实验 1（重新设计）: Output Examples — 展示合成人口产品

### 设计理念

当前的 fig_output_examples.pdf 只展示了 32-cell 概率向量的 paired bar + residual。
**问题**：读者（城市规划者、交通建模者）无法直观理解模型输出是什么。
**目标**：把 §2.1 的图重新设计为"产品展示"——让读者看到：
1. 这个地区的人口长什么样（demographic profile）
2. 模型正确捕获了该地区特有的属性关联模式（cross-tabulation）
3. 在完整的 32-cell joint 上，生成 ≈ 真实

### 选取 PUMA

从 Michigan 68 个测试 PUMA 中选 **2 个对比鲜明的地区**，展示模型对不同类型地区都有效：

| 行 | PUMA UID | 类型 | TVD-to-global 分位 | 选取理由 |
|---|---|---|---|---|
| 上排 | 2602903 | 大学城型（高异质性） | ~0.90 | 与全国均值偏离最大，最能展示"region-specific"能力 |
| 下排 | 2601100 | 典型郊区型（中位数） | ~0.50 | 接近平均水平，代表普通场景 |

（选取方法同现有脚本 `_select_representatives`，按 TVD-to-global 分位数）

### 图表布局

**fig_output_examples.pdf**
- 2 行 × 3 列 = 6 panels
- 尺寸：约 170mm × 100mm（单栏或双栏均可）

```
       Column (a)              Column (b)                  Column (c)
       人口画像                 关键交叉表                   完整联合分布

Row 1  [大学城 PUMA 2602903]    [education × income 2×2]    [32-cell paired bars]
       "N = xx,xxx 人"          True | Generated             true vs gen, person counts

Row 2  [郊区 PUMA 2601100]     [education × income 2×2]    [32-cell paired bars]
       "N = xx,xxx 人"          True | Generated             true vs gen, person counts
```

### 各 Panel 详细规格

#### Column (a): 人口画像 — "这个地区长什么样"

- **形式**：水平堆叠条形图，5 个属性各一行
- **每个属性**：2 段（bin0 | bin1），标注人数
- **两套颜色**：真实（蓝色系）vs 生成（橙色系），紧邻排列
- **标注**：
  - 左上角：PUMA 名称 + 人口类型标签（如 "University Town" / "Typical Suburb"）
  - 右上角：总人口 N = xxx,xxx（从 `total_person_weight` 获取）
- **人数计算**：`count = N × p_marginal`

具体标签示例（以 PUMA 2601100 为例）：
```
Age:        [Young ████ 56%  | Old ███ 44%]
Sex:        [Male ████ 48%   | Female ████ 52%]
Income:     [Low █████ 81%   | High ██ 19%]
Education:  [Low ████ 60%    | High ███ 40%]
Employment: [Employed ███ 46%| Not-Empl ████ 54%]
```

**关键点**：post-hoc IPF 保证边际精确匹配，所以蓝色和橙色应该几乎完全重合。这恰好传达了信息："边际一致性由设计保证。"

#### Column (b): 关键交叉表 — "属性间的关联是否被捕获"

- **变量对**：education × income（2×2 = 4 cells），最能体现 copula 差异
- **形式**：2 个并排的 2×2 annotated heatmap
  - 左边标 "True"，右边标 "Generated"
  - 每个 cell 标注**人数**和**占比**
- **色标**：连续色标（如 YlOrRd），两个 heatmap 共享同一色标范围
- **替代方案**：如果 heatmap 视觉效果不好（因为只有 4 cells），可用 **grouped bar chart**（4 组 × 2 bars）

交叉表计算：
```python
# 从 32-cell joint 提取 education × income 二维边际
tab = p_joint.reshape(shape)  # shape = (2,2,2,2,2) → (age, sex, income, schl, esr)
# income 是 axis=2, schl 是 axis=3
cross_tab = tab.sum(axis=(0,1,4))  # sum over age, sex, esr → shape (2,2)
# cross_tab[i,j] = P(income=i, schl=j)
# 人数 = N × cross_tab[i,j]
```

**注意 axis 顺序**：shape=(age, sex, income, schl, esr)，所以 income=axis2, schl=axis3。

**为什么选 education × income**：
- 这是最直观的 copula pair——大学城里"高学历+低收入"（学生）比例高，郊区里"高学历→高收入"是正常模式
- 两个 PUMA 的交叉表模式应该明显不同，直观展示"region-specific dependence"

#### Column (c): 完整联合分布 — "32 个 cell 的完整产品"

- **形式**：Paired bar chart，32 cells 按真实概率降序排列
  - 蓝色 = True，橙色 = Generated
  - y 轴 = **人数**（= N × p_k），不是概率
- **标注**：
  - 右上角：TVD = 0.xxx
  - x 轴标签：cell index（1–32），不需要标注属性组合（太挤）
  - 可在前 3-5 个最大 cell 上方标注属性组合缩写（如 "Y-M-L-L-E"）
- **辅助元素**：在 0 线附近加一条 residual subplot（小面积），或直接在 bar 上标色差

### 数据来源与脚本

```
输入文件：
  joint_wide_csv: outputs/_us_puma_5var_joint_k32_20260220T172157Z/puma_5var_joint_wide.csv
  checkpoint: outputs/_exp2_convergence_train_k32_20260306T032039Z/checkpoints/pairwise/leave_mi_out/final.pt

推理参数（与主结果一致）：
  condition: pairwise
  eval_mode: leave_mi_out
  n_eval_joint_samples: 128
  posthoc_ipf: true
  ipf_iters: 200

代码入口：
  修改 tools/essay/exp1_output_examples.py 或新建 tools/essay/exp1_output_examples_v2.py
  复用 _eval_5var_common.py 的 load_eval_data() 和 infer_one_region()

输出文件：
  figures/fig_output_examples.pdf (替换现有文件)
  输出 JSON 中额外记录：
    - total_population (N) for each PUMA
    - cross_tab_true 和 cross_tab_gen (education × income 2×2)
    - cell_counts_true 和 cell_counts_gen (32 cells)
```

### 配色

沿用 Okabe-Ito 色盲友好配色：
- True: OKABE_ITO["blue"] (#0072B2)
- Generated: OKABE_ITO["vermillion"] (#D55E00)
- 或者用 True=深灰, Generated=主题色（如果期刊要求灰度可读）

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
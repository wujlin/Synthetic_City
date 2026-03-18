# 可视化风格规范（Nature 级期刊标准）

本规范用于在不同项目/脚本/论文之间统一制图风格，以 Nature 系列期刊的 figure guidelines 为基准。

**唯一真源（source of truth）**：`src/visualization/plot_style.py`（`src/plot_style.py` 提供同名转发，便于短 import）。
本文档是其"人类可读版"说明，补充语义约定与检查清单。

> **参考标准**：[Nature Research Figure Guide](https://research-figure-guide.nature.com/)、[Springer Nature artwork guidelines](https://www.springernature.com/gp/authors/campaigns/writing-a-manuscript/figure-preparation)

---

## 1. 尺寸（物理尺寸 = 最终印刷尺寸）

Nature 的核心原则：**figure 提交时的物理尺寸就是最终印刷尺寸**，不会再被缩放。因此脚本中的 `figsize` 必须与目标版面宽度一致。

| 类型 | 宽度 | figsize 建议 | 备注 |
|---|---|---|---|
| **单栏** | 89 mm (3.5 in) | `(3.5, 2.6)` | 最常用，适合单面板图 |
| **1.5 栏** | 120 mm (4.7 in) | `(4.7, 3.5)` | 适合 2 面板并排 |
| **双栏/全宽** | 183 mm (7.2 in) | `(7.2, 4.5)` | 适合多面板复杂图 |

高度不超过 240 mm (9.4 in)，含 caption 不超过整页。

```python
# src/visualization/plot_style.py 中的尺寸常量
FIGSIZE_HALF = (3.5, 2.6)   # 单栏
FIGSIZE_FULL = (7.2, 4.5)   # 双栏/全宽
```

---

## 2. 字体

### 字体族

Nature 要求 **sans-serif**（Helvetica、Arial 为首选）。

| 优先级 | 字体 | 说明 |
|---|---|---|
| 1 | Helvetica | Nature 标准，macOS 内置 |
| 2 | Arial | Windows/WSL 常见替代 |
| 3 | DejaVu Sans | matplotlib 内置兜底 |

数学字体：`mathtext.fontset = "dejavusans"` 或 `"stixsans"`，保证公式与正文视觉协调。

### 字号（以最终印刷尺寸校准）

Nature 硬性要求：**最终印刷时任何文字不小于 5 pt，推荐 6–8 pt**。

由于我们的 figsize 已等于最终印刷尺寸，脚本中的字号就是最终字号：

| 元素 | 字号 | 说明 |
|---|---|---|
| 轴标签 | 8 pt | `axes.labelsize` |
| tick 标签 | 7 pt | `xtick/ytick.labelsize` |
| 图例 | 7 pt | `legend.fontsize` |
| 面板标签 (a/b/c) | 8 pt, **bold** | 见 `add_panel_label()` |
| 标题（若有） | 8 pt | `axes.titlesize`；通常不加 title，用 caption |

> **注意**：如果使用旧的 `FIGSIZE_FULL = (6.5, 4.0)` 等非 Nature 尺寸，字号需相应调整。
> 原则是：**缩到最终印刷宽度后，最小字号 ≥ 5 pt**。

---

## 3. 线宽与 Marker

Nature 要求：**最终印刷时线宽不小于 0.25 pt**（推荐 0.5–1.5 pt）。

| 元素 | 值 | 说明 |
|---|---|---|
| 数据线 | 1.0–1.5 pt | `lines.linewidth`；粗于 2 pt 在单栏图中会显得笨重 |
| 坐标轴 | 0.8 pt | `axes.linewidth` |
| tick | major.size=3.5, width=0.8 | |
| marker | 4–5 pt | `lines.markersize`；不宜过大，避免遮挡数据 |
| 参考线 | 0.6–0.8 pt, `linestyle=':'` | 灰色虚线/点线 |

---

## 4. 配色

### 4.1 色盲友好（强制要求）

Nature 明确要求 figure 对色觉障碍者友好。推荐使用 **Okabe–Ito** 调色板：

| 名称 | Hex | 用途建议 |
|---|---|---|
| `blue` | `#0072B2` | 主曲线 / 主方法 |
| `vermillion` | `#D55E00` | 对照 / baseline |
| `bluish_green` | `#009E73` | 第三组数据 |
| `orange` | `#E69F00` | 强调（谨慎使用） |
| `sky_blue` | `#56B4E9` | 浅色背景 / 辅助 |
| `reddish_purple` | `#CC79A7` | 第六组数据 |
| `gray` | `#777777` | 参考线 / 次要元素 |
| `black` | `#000000` | 文本 / 坐标轴 |

### 4.2 配色原则

- **不依赖颜色传递信息**：同时使用线型（实线/虚线/点线）或 marker 形状作为区分
- **避免红绿对比**：红绿色盲无法区分。用 blue vs vermillion 替代
- **避免彩虹色图**（jet / rainbow）：用 `viridis`、`cividis` 等感知均匀色图
- **语义映射全篇一致**：确定了"蓝=方法 A，红=方法 B"后，全文所有图保持一致

---

## 5. 面板标签（a, b, c, d）

Nature 要求：**小写粗体字母**，放在面板**左上角外侧**。

```python
from src.plot_style import add_panel_label
add_panel_label(ax, "a")  # 默认锚点 (0,1)，向左偏移 42pt
```

- 字体：加粗、黑色
- 位置：轴域外（不遮挡数据），通过固定 pt 偏移避免与 tick label 冲突
- 所有面板标签的位置和字号必须统一

---

## 6. 图例（Legend）

原则：**简洁，不遮挡数据**。

- 优先放在轴外（下方居中或右侧），避免覆盖数据区域
- `frameon=False`（不画边框）
- 每条图例项用最少的文字，避免长句
- 多面板共享同一组图例时，只放一份（通常在底部居中）

```python
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
)
```

---

## 7. 不确定性（置信区间 / 误差）

优先使用 `fill_between` 画 **shaded band**（而非 errorbar）：

- `alpha ≈ 0.15–0.25`
- `linewidth=0`（band 边界不画线）
- 与主线同色系
- 当 CI 很窄时 band 可能不显眼，这是正常现象（说明结果稳定）

Errorbar 仅在数据点稀疏时使用（如 bar chart），cap 长度适中。

---

## 8. 坐标轴与网格

- **默认不画网格**：`axes.grid = False`（Nature 风格偏简洁）
- **去除上/右边框**：`despine(ax)` 保留左/下边框
- **轴标签用 sentence case**：`"Arrival rate (%)"`，不用 Title Case
- **数学符号用 LaTeX**：`r"Polarization $|Q|$"`
- **负号**：`axes.unicode_minus = False`

---

## 9. 导出与文件格式

### 格式优先级

| 格式 | 用途 | 说明 |
|---|---|---|
| **PDF** | 投稿首选 | 矢量图，可缩放无损 |
| **EPS** | 部分期刊要求 | 矢量，兼容性好 |
| **TIFF** | 含照片/显微图像时 | 光栅，需高 DPI |
| **PNG** | 预览/内部交流 | 不用于正式投稿 |

### DPI 要求

| 内容类型 | 最低 DPI |
|---|---|
| 线条图（line art） | 1000 |
| 组合图（线条+灰度） | 600 |
| 照片/灰度图 | 300 |
| 纯矢量 (PDF/EPS) | 不适用 |

> **实际操作**：优先导出 PDF（矢量），无需关心 DPI。仅在必须光栅化时使用 TIFF 600+。

### 关键设置

```python
"pdf.fonttype": 42,       # 嵌入 TrueType，避免 Type3 字体
"ps.fonttype": 42,
"savefig.dpi": 300,       # PNG 预览用
```

### 禁止事项

- **不使用** `bbox_inches="tight"`：会导致不同文本元素引起 bounding box 抖动，多面板组图时子图错位
- **不使用** JPEG：有损压缩，不适合科学图表
- 统一用 `save_figure(fig, path)` 保存

---

## 10. Python 用法模板

```python
import matplotlib.pyplot as plt

from src.plot_style import (
    OKABE_ITO,
    FIGSIZE_HALF,
    FIGSIZE_FULL,
    paper_style,
    add_panel_label,
    save_figure,
    despine,
)

with paper_style():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    ax = axes[0]
    ax.plot(x, y, color=OKABE_ITO["blue"], lw=1.2, label="Method A")
    ax.fill_between(x, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, lw=0)
    ax.set_xlabel("Control parameter")
    ax.set_ylabel(r"Response $R$")
    despine(ax)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.plot(x, z, color=OKABE_ITO["vermillion"], lw=1.2, label="Baseline")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    despine(ax)
    add_panel_label(ax, "b")

    fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=False,
    )
    save_figure(fig, "figures/example.pdf")
```

---

## 11. 出图检查清单（交付前必过）

### 技术规范
- [ ] 物理尺寸匹配目标版面（单栏 89mm / 双栏 183mm）
- [ ] 最终印刷时最小字号 ≥ 5 pt（推荐 ≥ 7 pt）
- [ ] 最终印刷时最小线宽 ≥ 0.25 pt（推荐 ≥ 0.5 pt）
- [ ] PDF 导出无 Type3 字体（`pdf.fonttype = 42`）
- [ ] 未使用 `bbox_inches="tight"`

### 可读性
- [ ] 面板标签（a/b/c）小写加粗，在轴外，位置统一
- [ ] 图例不遮挡数据，无边框，字号一致
- [ ] 坐标轴标签 sentence case，数学符号用 LaTeX
- [ ] 去除上/右边框（除非有特殊理由保留）

### 色彩与可及性
- [ ] 配色色盲友好（Okabe–Ito 或等效）
- [ ] 不仅靠颜色区分数据——同时使用线型/marker
- [ ] 语义配色全文一致（同一方法在所有图中同色）
- [ ] 未使用 rainbow/jet 色图

### 数据完整性
- [ ] CI/errorband 使用 shaded band，`alpha` 适中
- [ ] 所有图中的数字与实验产物文件一致
- [ ] 多面板图的坐标轴范围合理对齐（共享轴时一致）

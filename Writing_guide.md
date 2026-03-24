这份文档总结了我们在打磨本项目主文过程中反复暴露出的写作问题。它不是泛泛而谈的“写作建议”，而是已经在 `main.tex` 中真实踩过、逐条修过的坑。后续所有写作与改稿，请优先按这份文档执行，避免重复返工。

# 本项目写作避坑指南与协作规范

## 0. 核心目标

写作的目标不是“把做过的步骤写全”，而是把以下三件事讲清楚：

1. **核心发现是什么**
2. **证据链如何逐层支撑这个发现**
3. **哪些地方是样本限制、方法边界或补充性分析**

如果一段话不能明确服务于这三件事，就应当重写或删除。

---

## 1. 全文主线必须稳定

### 1.1 Introduction、Methods、Results、Discussion 必须回答不同问题

- **Introduction**：提出问题，说明为什么已有研究还没有回答“为什么不同事件恢复快慢不同”
- **Methods**：定义数据、参数、检验设计，说明“如何回答这个问题”
- **Results**：报告证据，不要在这里重新解释方法动机
- **Discussion**：解释这些结果意味着什么，哪些是机制，哪些是边界

### 1.2 不能把不同部分写成同一种文体

- **Methods** 不能写成结果预告
- **Results** 不能写成方法说明书
- **Discussion** 不能只是把 Results 换个说法再复述一遍

---

## 2. 样本口径必须一次讲清，之后少重复

这是本轮最重要的教训之一。

### 2.1 不同样本数不是原罪，没解释清楚才是问题

主文里出现过 `18 / 16 / 15 / 13`，之所以危险，不是因为数字不同，而是因为读者会怀疑你在事后筛选。以后必须遵守：

- **18**：只能表示 `full study cohort`
- **16（event-level）**：主文事件级推断子集
- **16（geo-unit）**：geo-unit 主分析子集
- **14**：事件级与 geo-unit 分析的 overlap 子集

### 2.2 同样都是 16，也要说明是两个不同的 16

本项目里两个 `16-event subset` 不是完全同一批事件。以后必须写清：

- `the 16-event event-level fit subset`
- `the separate 16-event geo-unit fit subset`

绝不能偷懒写成一个模糊的 `the 16-event subset`，否则读者会误以为所有分析都建立在同一批事件上。

### 2.3 样本缩减必须归因于“可估性”，不能写成“我们挑了更干净的结果”

必须使用这类表述：

- `defined by estimability`
- `with usable event-level decay fits`
- `with sufficient unit-level fits`
- `overlap between the event-level and geo-unit subsets`

避免这类表述：

- `isolating a cohort`
- `strict quality filters`
- `selected the most reliable events`

这些写法会直接引发 cherry-picking 联想。

### 2.4 没有硬方法理由的 hazard-type exclusion 不要放进主文主叙事

本轮已经确认：`15 non-earthquake events` 不是数据管线天然约束，而是早期绘图脚本的人为展示口径。今后规则如下：

- 如果一个子样本是因为**灾害类型**被排除，必须给出明确、事先定义的方法理由
- 如果没有明确方法理由，就不能让它承担主文核心结论
- 这种口径最多作为补充性/敏感性分析，而不是主图主结果

---

## 3. Methods 的写法规则

### 3.1 Methods 里只能写“动作”和“设计”，不能写“结果”

以后在 Methods 里优先使用：

- `test`
- `examine`
- `compare`
- `estimate`
- `construct`
- `fit`
- `evaluate`

避免使用：

- `show`
- `demonstrate`
- `confirm`
- `prove`
- `establish`

原因很简单：这些动词更像 Results 或 Discussion 的结论语气。

### 3.2 少用 `To ..., we ...`，更不能把它写成假逻辑

`To test ..., we show ...` 这种写法是错误的，因为后半句不是方法动作，而是结果动作。以后如果用 `To ..., we ...` 句式，必须保证后半句真的是设计动作，比如：

- `To test whether diffusion underlies the observed association, we compare the empirical result against three null models.`

### 3.3 不要使用 run-in 小标题式写法

本轮已清理掉这类写法：

- `\emph{Counterfactual experiments.}\quad`
- `\textbf{Post-peak recovery rate.}`

原因：

- 会把主文写成实验报告提纲
- 破坏连续叙事
- 让 Methods 和 Results 看起来碎片化

如果真要分段，靠自然段首句完成，不靠人工强调词。

### 3.4 公式前必须先给物理图像

不能一上来就丢公式定义。必须遵守：

1. 先说参数在物理上代表什么
2. 再说它回答什么科学问题
3. 最后再给公式

例如：

- 先解释 `\dnear` 是“向外逃散还是向内聚拢的近场几何”
- 再给 `\dnear = \langle \delta(r) \rangle_{r < 50 km}`

### 3.5 Methods 可以交叉引用 Introduction，但目的必须明确

允许使用：

- `As discussed in the Introduction, ...`

但这个交叉引用只能用来说明**分析目标或问题动机**，不能和数据集 citation 混在同一句里，否则会造成功能混乱。

正确分工是：

- 一句回指 Introduction，交代当前模块要回答什么问题
- 下一句单独介绍数据来源或方法输入

---

## 4. Results 的写法规则

### 4.1 每一段只做一件事

本轮 Results 最大的问题之一，就是经常在同一段里同时做三件事：

- 讲图像直觉
- 报统计结果
- 解释物理机制

以后每段必须分工明确：

- 一段讲现象
- 一段讲定量关联
- 一段讲稳健性或排除性检验

### 4.2 不要在 Results 段首用空泛抽象句

避免：

- `This empirical shape--rate association demonstrates high statistical stability.`
- `A gradient-based mechanism also implies a separation across scales.`

更好的写法是直接说你发现了什么，或者这个段落要检验什么。

### 4.3 避免抽象名词堆叠

以下表达都在本轮被判定为不够自然或不够清楚：

- `empirical association`
- `genuine property`
- `mechanistic ingredients`
- `density anomaly`
- `common temporal sequence`
- `fall within the distribution`

原则是：**能写具体现象，就不要写抽象名词。**

### 4.4 Results 不要过度防御

避免：

- `strict quality filters`
- `genuine physical regularity`
- `definitive empirical proof`
- `perfectly partitions`

这些词会让读者提高警惕，像在“提前辩护”。结果段应该报告证据，不应该预设审稿人要反驳什么。

---

## 5. Caption 的严格规则

这是本轮反复修改最多的部分之一。

### 5.1 Caption 负责“读图”，正文负责“论证”

Caption 要回答的是：

- 每个 panel 画了什么
- 关键颜色/线型/marker 代表什么
- 必要的样本背景是什么

正文要回答的是：

- 效应大小
- 统计显著性
- 机制解释
- 对竞争解释的排除

### 5.2 Caption 不要重复正文里的统计结果

如果正文已经给了：

- `\rho`
- `p`
- `n`
- `R^2`
- ICC
- 相关方程

Caption 就不要再把同一组数字重复一遍，除非这些数字是**读图时必须知道**的。

### 5.3 Caption 的标题句不能直接写成主结论

避免：

- `Post-disaster displacement follows a universal power-law decay...`
- `Spatial diffusion explains ...`

更合适的是：

- `Post-disaster displacement and recovery dynamics across the study cohort.`
- `Synthetic, counterfactual, and cross-scale tests of the spatial-diffusion mechanism.`

### 5.4 Caption 里只保留必要数字

优先保留：

- 图的样本规模
- bootstrap 次数
- marker/line 含义

优先移到正文：

- 主效应 `p` 值
- 相关系数
- 回归系数
- 解释性句子

---

## 6. 术语、缩写与分析单位必须和实现一致

### 6.1 缩写只定义一次

例如：

- `Facebook Disaster Maps (FBDM)`：正文首次定义一次，后面统一用 `FBDM`
- `partial differential equation (PDE)`：正文首次定义一次，后面不要重复展开

### 6.2 不能把分析单位写错

本轮已经发现过的错误：

- 把 `geo-unit` 写成 `tiles`
- 把 `quadkey-based geo-units` 写成 `administrative subregions`

如果实现里用的是 `Level-10 quadkey geo-units`，正文就必须如实写：

- `geo-unit`
- `quadkey-based geo-unit`

不要为了听起来更“人类可读”就擅自改成 municipality / district，除非代码里真的做了行政区匹配。

### 6.3 术语要前后一致

全文中同一个概念不要反复切换叫法。例如：

- `spatial geometry`
- `spatial shape`
- `spatial structure`

三者如果都存在，必须有清楚的分工；否则就统一成一个主术语，避免读者误以为你在说三个不同概念。

---

## 7. 补充材料引用必须精确到位置

本轮已经明确：这类引用无效，必须禁止：

- `see in Supplementary Information`
- `for details, see Supplementary Information`

以后统一改成：

- `Supplementary~\S6.3`
- `Supplementary Fig.~S6`
- `Supplementary Table~S1`

规则：

1. 指明是节、图还是表
2. 最好指到具体编号
3. 不要把“整份 SI”当作一个引用对象

---

## 8. Discussion 的写法规则

### 8.1 Discussion 不能碎成一堆小标题

本轮已经删除了这类结构：

- `\paragraph{Spatial diffusion as a recovery mechanism.}`
- `\paragraph{Cross-scale universality.}`
- `\paragraph{What determines recovery beyond geometry?}`

原因：

- 会把 Discussion 写成答辩提纲
- 每段显得像独立备注，而不是一条连续论证

Discussion 更适合写成连续的 4 段逻辑：

1. 核心发现是什么
2. 为什么这些证据支持这个机制
3. 这个机制解释什么、不解释什么
4. 局限性和外推边界在哪里

### 8.2 Discussion 要明确“机制解释到哪一步为止”

本轮一个关键改进是明确区分：

- 事件尺度上，空间几何解释恢复快慢
- geo-unit 尺度上，局部恢复仍然高度异质

因此必须写清：

- 这是 `first-order predictor of event-level recovery`
- 不是 `complete explanation of geo-unit-level outcomes`

### 8.3 局限性必须具体，不要空泛

本轮确认应该明确写出的局限包括：

- FBDM 用户偏向数字可观测人群，可能更接近恢复“上界”
- 观察窗口只覆盖急性恢复，不覆盖长期重建
- diffuse hazards 的中心定义存在几何不确定性
- 海岸线、山脉等不对称边界会改变有效边界条件
- 全局 `k, \Ds` 是最简 null model，不是事件级精确拟合模型

这些局限要写成**具体约束**，而不是抽象的 `future work is needed`

---

## 9. 常见表达雷区

以下表达在本轮被反复判定为不合适，今后默认避免：

### 9.1 方法段不合适的动词

- `show`
- `demonstrate`
- `prove`
- `confirm`

### 9.2 容易引发防御感的词

- `strict`
- `genuine`
- `definitive`
- `perfectly`
- `isolate a cohort`

### 9.3 容易造成指代或逻辑不清的词

- `its`
- `drivers`
- `ingredients`
- `link`
- `anomaly`

### 9.4 可以优先考虑的替换

- `drives the empirical association` → `underlies the observed association`
- `breaks the link` → `disrupts the event-specific correspondence`
- `genuine property` → `intrinsic feature` 或直接写清具体含义
- `evaluate whether` → `test whether` / `examine whether`
- `if ... must ...`（Methods）→ `To test whether ...`

---

## 10. 图文与结果口径协同规则

### 10.1 主图口径必须和正文一致

如果正文改了样本口径，图和 caption 必须同时改。不能出现：

- 正文写 16，图注写 15
- 正文写 overlap 14，图注还写 13
- 图里排除了某类事件，正文却没说

### 10.2 figure 编号、文件名、引用名必须统一

本轮已经出现过：

- framework 图文件名不一致
- fig3 / fig4 旧文件残留
- PDF/JPG 混用

以后统一要求：

- 主文图统一按 `fig1_... / fig2_... / fig3_... / fig4_...`
- `main.tex` 的 `\includegraphics{}` 与实际文件名完全一致
- 优先使用 PDF 主图；PNG 仅作预览

### 10.3 每个 panel 都必须提供独立信息

本轮一个反复出现的问题是：某些 panel 只是把正文已经说过的话再画一遍，信息量不足，却会占用图面、分散注意力。

以后统一要求：

- 如果一个 panel 只是对其他 panel 或正文的压缩总结，而没有新增证据，优先删掉
- panel 数量宁少勿滥，宁可保留 `2` 个有独立功能的 panel，也不要硬凑成 `3` 或 `4`
- 在决定是否保留某个 panel 时，先问：**读者如果只看这一格，会不会获得正文和其他 panel 中没有的新信息？**

### 10.4 图的视觉风格必须服务信息，而不是抢信息

本轮已经明确，过饱和、过廉价、过演示稿化的配色会削弱学术图的可信度。以后默认遵守：

- 优先使用柔和、克制的科学配色，而不是高饱和撞色
- 同一类比较优先用**单色 + 不同 marker**，不要无必要地用多种强色
- 图的颜色层级应服务于主信息：主比较保留对比，辅助线、参考线、背景元素一律弱化
- caption 和正文负责解释，图面本身只保留必要视觉元素，避免“图像化装饰”

---

## 11. LaTeX 源码排版规范

### 11.1 不要把源码按短语硬断行

本轮很多段落都存在这种问题：

- 一句话被切成 6~8 行
- 一行只放一个短语
- 审阅和 diff 都非常难看

以后统一规则：

- 正常自然段，按句子或语义块换行
- 不要为了“视觉对齐”把一整段切碎
- caption 也一样，不要按短语硬断

### 11.2 初稿阶段不要使用强调格式

默认禁止：

- `\emph{}`
- `\textbf{}`

除非确实是公式符号、变量或期刊格式要求，否则不要用它们替代逻辑结构。

---

## 12. Synthetic City 项目新增写作规则

### 12.1 区分科学对象和计算对象

本项目里最容易混淆的是 `copula` 和 `joint distribution`。

- **Introduction / 问题定义层** 可以强调 `region-specific copula`
- **Methods / Results / 指标层** 必须以 `regional joint distribution` 为主，因为训练目标、条件构造和 TVD 评估都是围绕 joint distribution 展开
- 只有在需要连接两者时，才使用这类句子：
  - `the joint distribution encodes the regional copula`

禁止在同一段里反复来回切换：

- `copula`
- `joint distribution`
- `dependence structure`

除非三者分工非常明确，否则读者会误以为你在说三个不同对象。

### 12.2 条件信号和目标向量的来源必须写清楚

本轮已经明确：当前实验里，`PUMS` 同时提供了 target vector 和 condition，但二者的角色不同。

- `PUMA`：地理单元
- `PUMS`：当前实验的数据来源
- `condition vector c`：从 `PUMS` 构造的 `PUMA-level joint distribution` 再边缘化得到

以后必须写清：

- 当前实验中的 `condition` 是 **PUMA-level**
- 但它的数值是 **PUMS-derived**
- 当前实验不是直接从另一份外部 aggregate census table 读取 `condition`

否则读者会误以为当前论文已经完成了“外源 census constraints 驱动生成”的部署场景。

### 12.3 权重的层级不能写错

本轮的一个关键术语教训是：**不要写 `PUMA survey weight`。**

正确写法是：

- `survey-weighted aggregation of PUMS records within each PUMA`

因为：

- survey weights 属于 `PUMS` 记录
- `PUMA` 只是聚合层级

同时必须区分：

- `survey-weighted`：PUMA 内部从样本记录恢复总体分布
- `population-weighted average across training PUMAs`：PUMA 之间构造 global seed

这两类 `weight` 不能混写。

### 12.4 变量选择必须有总括性理由，不能只靠表格分散解释

本轮已经确认：当正文使用 `AGEP, SEX, PINCP, SCHL, ESR` 这五个变量时，不能只在变量表里分散说明它们的作用，而要在 Methods 正文里给出一句总括性 justification。

以后优先使用这类表述：

- 覆盖核心 demographic 和 socioeconomic 维度
- 存在有意义的 dependence structure
- 在 `PUMS` 和 aggregate summaries 中稳定可得
- 构成一个 tractable but demanding 的 joint reconstruction task

不要让读者觉得这五个变量只是“随手挑的 benchmark”。

### 12.5 Results 必须先稳住对象，再解释层次

本轮在 case-study results 中反复暴露出的写法问题包括：

- 把 `expected counts` 写成 `sampled discrete individuals`
- 用 `profiles` 指代明确的 `marginal distributions`
- 用 `generated ... are close to observed ones` 这类句子造成比较对象歧义

以后写 Results 时，优先遵守：

1. 先写清**比较对象是谁**
2. 再写清**比较发生在哪个层次**
3. 最后再给解释

例如：

- 先区分 `marginal / pairwise / full-joint`
- 再区分 `generated distribution` 是和 `its observed counterpart` 比，而不是和另一个 region 比

### 12.6 比较性结果必须放在比较性 subsection 里

本轮已经确认，`Michigan 5-fold diffusion vs. IPF` 这类结果不应放在 `Verification and validation` 中，而应放在 `Comparison with IPF` 里。

以后统一分工：

- `Verification`：模型自身是否稳定、是否收敛、采样波动是否可忽略
- `Comparison`：与 baseline 相比谁更好、优势是否稳定、cross-validation 是否仍成立

不要把：

- intrinsic quality
- benchmark advantage

混在同一个 subsection 里。

### 12.7 Supplementary Information 不能写成承诺清单

本轮 SI 最大的问题不是语言，而是“承诺了很多，实际没有给”。

以后禁止出现：

- `This SI also provides ...`
- `for submission, include ...`
- 空的 `Run Artifact Index`
- 空的 `Reproducibility Notes`

规则是：

- 只写已经实际放进 SI 的内容
- 如果某部分还没准备好，就先删掉，不要用占位句占着
- SI 的功能是补充主文证据链，不是展示一个未来可能完成的目录

---

## 13. Partner 提交前自检清单

每次提交新写的段落前，请至少自查以下问题：

- [ ] 这一段的首句是否直接说清了本段 claim？
- [ ] 这一段是在讲发现，还是在堆步骤？
- [ ] 这一段是否把方法、结果、讨论混写了？
- [ ] 是否出现了没有解释的样本数字？
- [ ] 这些样本数字是否会让人怀疑是事后筛选？
- [ ] 是否把 caption 写成了“第二套 Results”？
- [ ] 是否存在 `see Supplementary Information` 这类无效引用？
- [ ] 是否把分析单位写成了比实际实现更“好听”的名字？
- [ ] 是否使用了过强、过主观、过防御的词？
- [ ] 是否出现了 `\emph{}`、`\textbf{}`、run-in 小标题？
- [ ] LaTeX 源码是否被硬切成很多碎行？

---

## 14. 一句话总原则

**主文要像一条清楚的科学论证链，而不是一份实验记录、图注合集或防御性说明书。**

只要发现某段在回答“我们做了什么”，而不是“我们发现了什么、这意味着什么”，就说明那段还没写好。

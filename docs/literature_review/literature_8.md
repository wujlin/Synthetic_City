下面这份补充文献清单，是围绕你们的核心叙事线（“IPF 的全局 seed = 全局 copula 假设”→ 空间异质性被否定 → distribution-level conditional diffusion 从边际恢复异质性 copula）来“补地基 + 补谱系 + 补技术对标 + 补评估先例”的。优先级严格按你给的：任务2 > 任务1 > 任务3 > 任务4。

我先说一个对写 Related Work/Discussion 最关键的“抓手句式”：在经典 synthetic reconstruction / IPF 系谱里，作者会非常直白地讲——合成时要“保留 seed 的 correlation / interaction structure（交互结构）”，而 IPF 本质上是在满足边际约束的集合里，找一个“与初始表（seed/starting table）最相似”的解（最小相对熵 / 保 odds ratio / cross-product ratios）。这几句话几乎等价于你们要写的：“seed 决定了边际之外的依赖结构（离散意义下可视为 copula/interaction）”。你们可以把它写成一段“IPF = KL projection onto marginal constraints; seed encodes dependence/coupling beyond marginals; hence seed implies a global copula assumption”。

---

任务 2：Copula 在 population synthesis 中的角色（你们叙事地基）

[1] Templ, Meindl, Alfons, Dupriez (2017). Simulation of Synthetic Complex Data: The R Package simPop. Journal of Statistical Software.

* 一句话摘要：把 synthetic population 的主流路线（synthetic reconstruction / combinatorial optimization / model-based）系统总结，并明确写出“估计 joint distribution 时要保留 seed 的 correlation structure”，同时讨论 IPF 的相对熵最小化与 odds ratio / cross-product ratio（交互结构）保留等性质。
* 和我们的关系：support（直接给你们“seed=dependence/correlation structure”的权威措辞；也能作为“IPF 隐式固定交互结构”的引用锚点）
* 建议放在 essay 的位置：Related Work（population synthesis 谱系骨架）+ Discussion（seed = global copula assumption 的论据段落） ([ResearchGate][1])

[2] Naszodi (2023). Odds ratios preservation and the parameterization of Iterative Proportional Fitting. arXiv.

* 一句话摘要：从参数化与统计性质角度讨论 IPF 的 odds ratio（交互/关联结构）保留特性，并把 IPF 看作在约束下对初始表的“结构性调整”。
* 和我们的关系：support（把“IPF 保护了哪些依赖结构”说得更理论化，能帮你们把“copula/interaction”说得更严谨）
* 建议放在 essay 的位置：Discussion（为什么 IPF 的 seed 假设等价于固定 dependence/coupling；以及为什么这会在空间异质性下失效） ([arXiv][2])

[3] Csiszár (1975). I-divergence geometry of probability distributions and minimization problems. The Annals of Probability.

* 一句话摘要：奠定 I-divergence（KL）几何与投影（I-projection）的理论基础，是把“在满足边际约束的分布集合里，找与某个参考分布最近的解”形式化的经典来源。
* 和我们的关系：support（你们写“IPF/矩阵配平是对 seed 的 KL 投影，因此 seed 充当 dependence prior/coupling prior”时，用它做理论背书最稳）
* 建议放在 essay 的位置：Related Work（理论背景一两句）或 Discussion（把 IPF 的隐式先验说清楚）

[4] Lovelace et al. (2015). From open data to spatial microsimulation: harnessing open data and GUIs for generating synthetic populations. Journal of Artificial Societies and Social Simulation.

* 一句话摘要：面向 spatial microsimulation / synthetic populations 的综述与工具化实践总结，强调典型流程是用 seed microdata 的重加权/匹配去满足小区尺度边际。
* 和我们的关系：support（用于把“IPF/重加权是空间人口合成的主流”写成很稳的领域共识背景；并自然引出 seed 假设的局限）
* 建议放在 essay 的位置：Introduction（领域背景）+ Related Work（spatial microsimulation / reweighting 主线） ([JASSS][3])

[5] Krupskii, Huser, Genton (2018). Factor Copula Models for Replicated Spatial Data. Journal of the American Statistical Association.

* 一句话摘要：提出用于 replicated spatial data 的 factor copula 框架，专门为“空间数据的依赖结构（尤其尾部依赖/不对称）”提供比 Gaussian copula 更灵活的建模。
* 和我们的关系：support（强力支撑“copula 作为 dependence 载体在空间数据中是自然语言”；也能当你们“空间依赖结构复杂且需要更灵活模型”的外部证据）
* 建议放在 essay 的位置：Introduction/Related Work（copula 的空间建模动机）+ Discussion（空间依赖异质性/非高斯性） ([Marc G. Genton][4])

[6] Mondal, Krupskii, Genton (2024). A non-stationary factor copula model for non-Gaussian spatial data. Stat.

* 一句话摘要：提出“非平稳（non-stationary）”的 factor copula，用于让空间依赖结构随空间位置变化，面向非高斯空间数据。
* 和我们的关系：support（你们叙事线第一步“copula 空间异质性存在”的外部强证据：统计学界有专门的 non-stationary spatial copula 建模方向）
* 建议放在 essay 的位置：Introduction（copula heterogeneity 的外部证据）或 Related Work（spatially varying dependence） ([Wiley Online Library][5])

[7] Kudryashova et al. (2022). Parametric Copula-GP model for analyzing multidimensional neuronal and behavioral relationships. PLoS Computational Biology.

* 一句话摘要：用 Gaussian Process 让 copula 参数随外部变量（文中讨论可为 temporal 或 spatial context）变化，从而刻画“依赖关系随上下文变化”的现象。
* 和我们的关系：extend（虽然应用域不同，但方法论上给你们一个很贴的类比：把 dependence/coupling 当作可随空间变化的对象去学；你们是从边际约束学习 copula，而他们是从观测对学习参数）
* 建议放在 essay 的位置：Related Work（spatially/ context-varying copula）或 Discussion（“依赖结构可变”是跨领域现象） ([PLOS][6])

[8] Vo, Kim, Bansal (2025). A novel data fusion method to leverage passively-collected mobility data in generating spatially-heterogeneous synthetic population. Transportation Research Part B: Methodological.

* 一句话摘要：提出 cluster-based data fusion，把被动采集的移动数据与 HTS 结合，目标就是生成“空间异质性更强”的 synthetic population，并做分布一致性验证。
* 和我们的关系：support（非常贴你们要反驳的“全局 seed 假设”：交通/城市计算社区已经在主动追求 spatial heterogeneity，并用 cluster/分区来做“局部化”）
* 建议放在 essay 的位置：Related Work（region-specific/cluster-based 的局部化思路，作为“已有尝试但仍非 distribution-level copula learning”的对比） ([IDEAS/RePEc][7])

[9] Hawkins, Habib (2022/2023). A multi-source data fusion framework for joint population, expenditure, and time use synthesis. Transportation (Springer).

* 一句话摘要：提出多源数据融合工作流，把多个 national datasets 融合来补齐变量（expenditure、time use 等），再与 population synthesis 结合用于城市模拟。
* 和我们的关系：support（作为“多源融合提升属性维度完整性”的先例；也可用来写“信息质量驱动而非模型容量”的叙事铺垫：缺什么变量就会决定上限）
* 建议放在 essay 的位置：Related Work（multi-source fusion / enrichment） ([IDEAS/RePEc][8])

[10] Lee, Bansal, Vo, Kim (2025). Collaborative generative adversarial networks for fusing household travel survey and smart card data to generate heterogeneous activity schedules in urban digital twins. Transportation Research Part C.

* 一句话摘要：用 CollaGAN 结构把 HTS 与 smart card 数据在 latent space 里对齐融合，并强调保留小样本的细粒度信息与大规模数据的异质性模式。
* 和我们的关系：support/contrast（support：多源融合 + 生成模型在交通领域落地；contrast：他们是 record/trajectory-level GAN，你们是 distribution-level diffusion，从边际约束恢复 dependence）
* 建议放在 essay 的位置：Related Work（DL + fusion + heterogeneity） ([科学直接][9])

补充一个你们可以明确写的“gap/novelty”句式（建议 Discussion 或 Related Work 末尾）：在上述 population synthesis 与 data fusion / GAN 工作中，训练对象主要仍是“个体记录或活动记录”；而你们的核心训练对象是“区域层面的联合分布向量（distribution-level sample）”，并从边际约束学习空间异质 copula。就我这轮针对性检索与对比，上述 population synthesis 论文里没有看到把“每个样本=一个分布向量”作为生成建模单位、并用 diffusion 做条件生成来恢复局部 dependence 的直接先例；最接近的是下面任务3里的“simplex diffusion / Dirichlet diffusion”在概率单纯形上生成分布向量，但它们不是为人口合成与边际约束而设计。这个“closest prior art”对你们写 novelty claim 很有利。([arXiv][10])

---

任务 1：Population Synthesis 方法谱系（Related Work 骨架补齐，含 copula estimation）

下面按你表格里的类别来给“骨架+定位”。其中不少条目同时也服务任务2/3。

IPF 经典 / matrix fitting

[11] Deming, Stephan (1940). On a Least Squares Adjustment of a Sampled Frequency Table When the Expected Marginal Totals are Known. Annals of Mathematical Statistics.

* 一句话摘要：提出 raking / IPF 的经典起点，用迭代方式调整表使其满足给定边际。
* 和我们的关系：support（作为 IPF 系谱的源头；你们要“替代 IPF 的全局 seed 假设”，必须把它作为 baseline 背景）
* 建议放在 essay 的位置：Related Work（IPF 经典段落开头）
* 注：simPop 综述里把 IPF/矩阵配平作为常用方法并回溯到该脉络。 ([ResearchGate][1])

[12] Ireland, Kullback (1968). Contingency Tables with Given Marginals. （经典 IPF 理论性质来源之一，常被用来说明相对熵最小/最相似解）

* 一句话摘要：讨论给定边际下的表调整问题，常被后续工作用来说明 IPF 的“最小相对熵/最相似于初始表”的性质。
* 和我们的关系：support（你们要把“seed = dependence prior / copula prior”写严谨，这类结果是 IPF 的理论支点）
* 建议放在 essay 的位置：Related Work 或 Discussion（“IPF 是 KL projection，因此 seed=隐式依赖假设”） ([ResearchGate][1])

（你们已有 Bishop 1975、Choupani 2016，这里不重复展开。）

Combinatorial Optimization（SA/GA 等）

[13] Templ et al. (2017). Simulation of Synthetic Complex Data: The R Package simPop. Journal of Statistical Software.

* 一句话摘要：把 combinatorial optimization（例如 simulated annealing 校准权重/匹配边际）作为合成人口的重要分支，并在工具层面落地。
* 和我们的关系：contrast（你们方法属于“从边际约束恢复局部依赖”的 distribution-level diffusion；CO 路线多是直接在组合空间里找一个满足约束的样本集合，不显式学习区域 copula）
* 建议放在 essay 的位置：Related Work（Combinatorial Optimization 小节） ([ResearchGate][1])

[14] simPop CRAN documentation (版本见 CRAN 文档). simPop: Simulation of Complex Synthetic Data（软件文档，含 simulated annealing 校准接口描述）

* 一句话摘要：给出 SA 校准 synthetic population 的具体接口与目标描述（在可接受误差意义下匹配边际）。
* 和我们的关系：support（作为 CO baseline 的可复现工具引用；你们若要对标 SA/CO，可以用它做实现/复现入口）
* 建议放在 essay 的位置：Related Work（实现层面补充一句）或 Appendix（baseline 实现细节） ([CRAN][11])

Bayesian / MCMC（你们已有 Farooq 2013，这里补“相关但更贴你们 copula/空间依赖”的概率建模文献）

[15] Krupskii et al. (2018). Factor Copula Models for Replicated Spatial Data. JASA.

* 一句话摘要：用显式 copula 作为 spatial dependence 的建模组件，并用似然推断；强调“空间数据的依赖结构不是随便一个 exchangeable copula 能搞定的”。
* 和我们的关系：support（把“copula 是 dependence 的对象”写得很正统；你们把它推进到“从边际约束学习 copula”，是很自然的延伸）
* 建议放在 essay 的位置：Related Work（Copula/Dependence 建模） ([Marc G. Genton][4])

深度生成模型做表格数据（GAN/VAE/Diffusion）

[16] Kotelnikov et al. (2023). TabDDPM: Modelling Tabular Data with Diffusion Models. ICML (PMLR).

* 一句话摘要：系统讨论 tabular 的连续/离散混合特征建模，并证明 diffusion 在多套 tabular benchmark 上优于 GAN/VAE。
* 和我们的关系：support（你们用 diffusion 是主叙事之一；TabDDPM 是“tabular diffusion”最主流的技术对标；也可用来论证 standardization/特征尺度处理的工程常识）
* 建议放在 essay 的位置：Related Work（tabular diffusion）+ Methods（说明你们为什么选 diffusion / 与 TabDDPM 的区别：你们生成的是“分布向量”而不是 record） ([Proceedings of Machine Learning Research][12])

[17] Villaizán-Vallelado et al. (2025). Diffusion Models for Tabular Data Imputation and Synthetic Data Generation. ACM Computing Surveys（或 ACM 综述型期刊页面对应条目）。

* 一句话摘要：在 tabular 场景系统化讨论 diffusion 用于缺失值填补与合成数据生成，并以 TabDDPM 为关键基线之一。
* 和我们的关系：support（你们 Methods/Related Work 里想把 diffusion 的 tabular 技术脉络写“更像综述”，这篇很适合作为“综述型总引”）
* 建议放在 essay 的位置：Related Work（tabular diffusion/评估方法综述引用） ([ACM数字图书馆][13])

深度生成模型做 population synthesis（DL for population synthesis / heterogeneity / multi-source）

[18] “Enhancing Diversity and Feasibility: Joint Population Synthesis from Multi-source Data Using Generative Models” (2026). arXiv.

* 一句话摘要：提出 multi-source GAN 框架联合学习 census 与 travel survey，并强调 sampling zeros / structural zeros 与统一评估指标。
* 和我们的关系：contrast（他们是 record-level GAN；你们是 distribution-level diffusion，并把“边际约束→局部 copula”作为核心科学问题）
* 建议放在 essay 的位置：Related Work（DL for population synthesis 的最新进展） ([arXiv][14])

[19] Vo et al. (2025). A novel data fusion method to leverage passively-collected mobility data in generating spatially-heterogeneous synthetic population. TRB-B.

* 一句话摘要：用 cluster-based fusion 显式追求 spatial heterogeneity，并做分布一致性校验。
* 和我们的关系：support（证明“空间异质性是被关注且能提升”的研究方向；也可作为你们“conditional information quality 是瓶颈”的现实证据：他们通过引入 mobility 这种条件信息改善 spatial heterogeneity）
* 建议放在 essay 的位置：Related Work（heterogeneity / multi-source / alternative data） ([IDEAS/RePEc][7])

Copula 估计（你们当前为 0，这里给最经典的 4+1）

[20] Sklar (1959). Fonctions de Répartition à n Dimensions et Leurs Marges. Publications de l’Institut de Statistique de l’Université de Paris.

* 一句话摘要：Sklar 定理：把 joint 分解为 marginals + copula（依赖结构），是你们整篇论文“marginal constraints vs dependence/coupling”叙事的数学原点。
* 和我们的关系：support（Intro/Methods 里定义 copula 与“空间异质 copula”的合法性必须引用）
* 建议放在 essay 的位置：Introduction（定义 copula）或 Methods（符号与理论背景） ([科学与教育出版][15])

[21] Nelsen (2006, 2nd ed.). An Introduction to Copulas. Springer.

* 一句话摘要：最常用的 copula 教科书级引用，覆盖 copula 基本性质、依赖度量、族与应用。
* 和我们的关系：support（让你们“copula 是标准 dependence language”这一点站得很稳；也方便你们引用 tail dependence、Kendall’s tau 等）
* 建议放在 essay 的位置：Related Work（copula estimation 背景）或 Methods（依赖度量） ([施普林格][16])

[22] Joe (2014). Dependence Modeling with Copulas. Chapman & Hall/CRC.

* 一句话摘要：高维依赖建模与 vine copula 等体系化总结（更偏高维/结构化）。
* 和我们的关系：support（你们说“我们学习的是空间异质 copula/coupling”，Joe 这本书可作为高维依赖建模的标准引文）
* 建议放在 essay 的位置：Related Work（高维 copula / vine / dependence structure） ([Taylor & Francis][17])

[23] Aas, Czado, Frigessi, Bakken (2009). Pair-copula constructions of multiple dependence. Insurance: Mathematics and Economics.

* 一句话摘要：vine copula / pair-copula 构造的代表性经典论文之一，为“高维 copula 可分解/可构造”提供具体方法与案例。
* 和我们的关系：support（如果你们想在 Discussion 里对比“传统 copula 建模 vs 你们从边际学 copula”，这篇可作为高维 copula 的代表路线）
* 建议放在 essay 的位置：Related Work（copula estimation / high-dimensional copulas） ([科学直接][18])

[24] Patton (2006). （conditional copula / copula parameters vary with covariates 的经典经济计量路线之一）

* 一句话摘要：把 copula 参数设为随外生变量/时间变化的函数，是“dependence 不是全局常数”的经典建模思想来源。
* 和我们的关系：support/extend（你们可以把“空间位置/区域特征”视为 covariates；你们的关键创新是：不是用观测对去拟合 copula，而是从边际约束与分布级样本学习）
* 建议放在 essay 的位置：Related Work（time-/covariate-varying copulas）
* 备注：Krupskii et al. 的引言列举了 Patton(2006) 作为 copula 应用与依赖建模的代表引用之一，可作为你们引用 Patton 的间接支撑。 ([Marc G. Genton][4])

---

任务 3：Diffusion Model 技术对标（conditional diffusion / simplex diffusion / standardization）

[25] Ho, Salimans (2022). Classifier-Free Diffusion Guidance. arXiv / OpenReview.

* 一句话摘要：提出 classifier-free guidance，通过同时训练 conditional 与 unconditional 模型并线性组合 score 实现条件引导，是当代条件扩散最常引用的基础技术之一。
* 和我们的关系：support（你们 Methods 里需要一个“条件扩散怎么做”的基础引用；尤其当你们强调“条件信息质量驱动性能”时，CFG 也是“条件强度可调”的经典对照）
* 建议放在 essay 的位置：Methods（conditional diffusion 技术背景） ([arXiv][19])

[26] Floto et al. (2023). Diffusion on the Probability Simplex. arXiv.

* 一句话摘要：直接把 diffusion 定义在概率单纯形上，把点解释为“类别分布/概率向量”，属于你们“distribution-level diffusion”最接近的外部先例之一。
* 和我们的关系：support/contrast（support：证明在 simplex 上做 diffusion 是已有路线；contrast：他们是通用分布向量生成，你们是“从边际约束恢复空间异质 copula”的条件生成，并带有明确的合成人口目标与评估）
* 建议放在 essay 的位置：Methods（simplex/log-space diffusion 的 related work）+ Related Work（distribution-level generation 先例） ([arXiv][10])

[27] Avdeyev et al. (2023). Dirichlet Diffusion Score Model for Biological Sequence Generation. ICML (PMLR).

* 一句话摘要：在概率单纯形上用 Dirichlet 作为稳态分布来做 score-based diffusion，面向离散序列生成；并展示可加入 hard constraints（例如 Sudoku 约束）的例子。
* 和我们的关系：support（两点很贴：1）“simplex/Dirichlet diffusion”给你们在概率空间加噪与反演的技术参照；2）他们展示了“硬约束可融入 diffusion”的可行性，你们可借鉴到“边际约束/可行域投影”等机制讨论）
* 建议放在 essay 的位置：Methods（simplex diffusion / constraints with diffusion） ([Proceedings of Machine Learning Research][20])

[28] Kotelnikov et al. (2023). TabDDPM: Modelling Tabular Data with Diffusion Models. ICML (PMLR).

* 一句话摘要：tabular diffusion 代表作，包含连续特征标准化/处理与离散特征建模的工程细节与系统评测。
* 和我们的关系：support（你们问的“standardization / z-score normalization 对 diffusion 的作用”，tabular diffusion 文献通常会把特征缩放作为默认预处理；TabDDPM 是最合适的引用锚点）
* 建议放在 essay 的位置：Methods（数据预处理与噪声调度匹配的说明） ([Proceedings of Machine Learning Research][12])

---

任务 4：评估方法 precedent（TVD / spatial holdout / sign test）

[29] Demšar (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. Journal of Machine Learning Research.

* 一句话摘要：系统讨论在多数据集上比较多个算法时应使用的稳健非参数检验（如 Wilcoxon signed-rank、Friedman + post-hoc），是 ML 论文里“统计显著性比较”的标准引用。
* 和我们的关系：support（你们使用 sign test / 非参数检验来比较模型时，可用它做方法论背书；即便你们最终用 sign test，也可以写成“遵循非参数检验的推荐实践”并引用该文）
* 建议放在 essay 的位置：Methods（统计检验）或 Appendix（显著性检验说明） ([jmlr.org][21])

[30] García, Herrera (2008). An Extension on “Statistical Comparisons of Classifiers over Multiple Data Sets”. Journal of Machine Learning Research.

* 一句话摘要：在 Demšar 框架上补充与扩展多算法比较的统计程序与实践建议。
* 和我们的关系：support（如果审稿人对统计检验方案较挑剔，这篇可作为补强引用）
* 建议放在 essay 的位置：Methods/Appendix（统计检验） ([jmlr.org][22])

关于 TVD 与 spatial holdout：这两项在“population synthesis”论文里常见的写法分别是（a）用 L1/TV 距离或其变体衡量合成分布与真实分布差异，（b）用 leave-one-region-out 的空间外推评估检验条件信息可迁移性。你们可以在 Related Work 里把它描述为“distributional fidelity + spatial generalization”的常规组合，然后在 Methods 里定义 TVD（TVD = 0.5 * L1）并把 holdout 设计与现实用例（跨州/跨区域迁移）对齐；这部分如果你希望我再补“在合成人口/合成表格数据里明确用 TVD 的论文”，需要再做一次专门检索把条目补齐（目前我这里更确定的是你们的 diffusion/tabular 对标与统计检验引用）。([Proceedings of Machine Learning Research][12])

---

统一 BibTeX（把上面条目直接丢进 references.bib）

```bibtex
@article{Templ2017simPop,
  title        = {Simulation of Synthetic Complex Data: The R Package simPop},
  author       = {Templ, Matthias and Meindl, Bernhard and Alfons, Andreas and Dupriez, Olivier},
  journal      = {Journal of Statistical Software},
  year         = {2017},
  url          = {https://www.jstatsoft.org/article/view/v079i10/1142}
}

@misc{Naszodi2023IPFOddsRatio,
  title         = {Odds ratios preservation and the parameterization of Iterative Proportional Fitting},
  author        = {Naszodi, Mauro},
  year          = {2023},
  eprint        = {2301.?????},
  archivePrefix = {arXiv},
  note          = {arXiv preprint},
  url           = {https://arxiv.org/abs/2301.?????}
}

@article{Csiszar1975IDivergence,
  title   = {I-divergence geometry of probability distributions and minimization problems},
  author  = {Csisz{\'a}r, Imre},
  journal = {The Annals of Probability},
  year    = {1975},
  volume  = {3},
  number  = {1},
  pages   = {146--158}
}

@article{Lovelace2015OpenDataSpatialMicrosimulation,
  title   = {From open data to spatial microsimulation: harnessing open data and GUIs for generating synthetic populations},
  author  = {Lovelace, Robin and others},
  journal = {Journal of Artificial Societies and Social Simulation},
  year    = {2015},
  url     = {https://www.jasss.org/18/2/14.html}
}

@inproceedings{Konduri2016EnhancedPopGen,
  title     = {Enhanced Synthetic Population Generator That Uses Multilevel Controls},
  author    = {Konduri, Keerthi Chandra and You, Dongho and Pendyala, Ram M.},
  booktitle = {Transportation Research Record},
  year      = {2016},
  url       = {https://www.semanticscholar.org/paper/Enhanced-Synthetic-Population-Generator-That-Uses-Konduri-You/7c123b0f889da3d0c8a4bfac62d1994e82150144}
}

@article{MorenoMoeckel2018MultiLevelGeography,
  title   = {Population synthesis handling three geographical resolutions},
  author  = {Moreno, A. Torres and Moeckel, R.},
  journal = {Transportation Research Record},
  year    = {2018},
  url     = {https://www.semanticscholar.org/paper/Population-synthesis-handling-three-geographical-Moreno-Moeckel/42d321abe57eea2b7fbd8c6a315b14a37f872e30}
}

@article{Vo2025ClusterDataFusion,
  title   = {A novel data fusion method to leverage passively-collected mobility data in generating spatially-heterogeneous synthetic population},
  author  = {Vo, Khoa D. and Kim, Eui-Jin and Bansal, Prateek},
  journal = {Transportation Research Part B: Methodological},
  year    = {2025},
  volume  = {191},
  pages   = {103128},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0191261524002522}
}

@article{HawkinsHabib2022MultiSourceFusion,
  title   = {A multi-source data fusion framework for joint population, expenditure, and time use synthesis},
  author  = {Hawkins, Jason and Habib, Khandker Nurul},
  journal = {Transportation},
  year    = {2023},
  volume  = {50},
  number  = {4},
  doi     = {10.1007/s11116-022-10279-8},
  url     = {https://ideas.repec.org/a/kap/transp/v50y2023i4d10.1007_s11116-022-10279-8.html}
}

@article{Lee2025CollaGAN,
  title   = {Collaborative generative adversarial networks for fusing household travel survey and smart card data to generate heterogeneous activity schedules in urban digital twins},
  author  = {Lee, Huichang and Bansal, Prateek and Vo, Khoa D. and Kim, Eui-Jin},
  journal = {Transportation Research Part C: Emerging Technologies},
  year    = {2025},
  pages   = {105125},
  url     = {https://www.sciencedirect.com/science/article/pii/S0968090X25001299}
}

@article{Krupskii2018FactorCopula,
  title   = {Factor Copula Models for Replicated Spatial Data},
  author  = {Krupskii, Pavel and Huser, Rapha{\"e}l and Genton, Marc G.},
  journal = {Journal of the American Statistical Association},
  year    = {2018},
  volume  = {113},
  number  = {521},
  pages   = {467--479},
  doi     = {10.1080/01621459.2016.1261712},
  url     = {https://marcgenton.github.io/2018.KHG.JASA.pdf}
}

@article{Mondal2024NonStationaryFactorCopula,
  title   = {A non-stationary factor copula model for non-Gaussian spatial data},
  author  = {Mondal, Sagnik and Krupskii, Pavel and Genton, Marc G.},
  journal = {Stat},
  year    = {2024},
  volume  = {13},
  number  = {3},
  pages   = {e715},
  url     = {https://onlinelibrary.wiley.com/toc/20491573/2024/13/3}
}

@article{Kudryashova2022CopulaGP,
  title   = {Parametric Copula-GP model for analyzing multidimensional neuronal and behavioral relationships},
  author  = {Kudryashova, Nina and Amvrosiadis, Theoklitos and Dupuy, Nathalie and Rochefort, Nathalie and Onken, Arno},
  journal = {PLOS Computational Biology},
  year    = {2022},
  volume  = {18},
  number  = {1},
  pages   = {e1009799},
  doi     = {10.1371/journal.pcbi.1009799},
  url     = {https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009799}
}

@article{Sklar1959,
  title   = {Fonctions de r{\'e}partition {\`a} n dimensions et leurs marges},
  author  = {Sklar, Abe},
  journal = {Publications de l'Institut de Statistique de l'Universit{\'e} de Paris},
  year    = {1959},
  volume  = {8},
  pages   = {229--231}
}

@book{Nelsen2006Copulas,
  title     = {An Introduction to Copulas},
  author    = {Nelsen, Roger B.},
  publisher = {Springer},
  year      = {2006},
  edition   = {2},
  doi       = {10.1007/0-387-28678-0},
  url       = {https://link.springer.com/book/10.1007/0-387-28678-0}
}

@book{Joe2014Copulas,
  title     = {Dependence Modeling with Copulas},
  author    = {Joe, Harry},
  publisher = {Chapman and Hall/CRC},
  year      = {2014},
  doi       = {10.1201/b17116},
  url       = {https://www.taylorfrancis.com/books/mono/10.1201/b17116/dependence-modeling-copulas-harry-joe}
}

@article{Aas2009PairCopula,
  title   = {Pair-copula constructions of multiple dependence},
  author  = {Aas, Kjersti and Czado, Claudia and Frigessi, Arnoldo and Bakken, Henrik},
  journal = {Insurance: Mathematics and Economics},
  year    = {2009},
  volume  = {44},
  number  = {2},
  pages   = {182--198},
  doi     = {10.1016/j.insmatheco.2007.02.001},
  url     = {https://www.sciencedirect.com/science/article/pii/S0167668707000194}
}

@misc{HoSalimans2022CFG,
  title         = {Classifier-Free Diffusion Guidance},
  author        = {Ho, Jonathan and Salimans, Tim},
  year          = {2022},
  archivePrefix = {arXiv},
  eprint        = {2207.12598},
  url           = {https://arxiv.org/abs/2207.12598}
}

@misc{Floto2023SimplexDiffusion,
  title         = {Diffusion on the Probability Simplex},
  author        = {Floto, Griffin and Jonsson, Thorsteinn and Nica, Mihai and Sanner, Scott and Zhu, Eric Zhengyu},
  year          = {2023},
  archivePrefix = {arXiv},
  eprint        = {2309.02530},
  url           = {https://arxiv.org/abs/2309.02530}
}

@inproceedings{Avdeyev2023DDSM,
  title     = {Dirichlet Diffusion Score Model for Biological Sequence Generation},
  author    = {Avdeyev, Pavel and others},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/avdeyev23a.html}
}

@inproceedings{Kotelnikov2023TabDDPM,
  title     = {TabDDPM: Modelling Tabular Data with Diffusion Models},
  author    = {Kotelnikov, Akim and Baranchuk, Dmitry and Rubachev, Ivan and Babenko, Artem},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v202/kotelnikov23a.html}
}

@article{VillaizanVallelado2025TabularDiffusionSurvey,
  title   = {Diffusion Models for Tabular Data Imputation and Synthetic Data Generation},
  author  = {Villaiz{\'a}n-Vallelado, M. and others},
  journal = {ACM Computing Surveys},
  year    = {2025},
  url     = {https://dl.acm.org/doi/full/10.1145/3742435}
}

@article{Demsar2006,
  title   = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  author  = {Dem{\v{s}}ar, Janez},
  journal = {Journal of Machine Learning Research},
  year    = {2006},
  volume  = {7},
  pages   = {1--30},
  url     = {https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf}
}

@article{Garcia2008ExtensionDemsar,
  title   = {An Extension on “Statistical Comparisons of Classifiers over Multiple Data Sets”},
  author  = {Garc{\'\i}a, Salvador and Herrera, Francisco},
  journal = {Journal of Machine Learning Research},
  year    = {2008},
  volume  = {9},
  pages   = {2677--2694},
  url     = {https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf}
}

@misc{JointPopSynthesis2026MultiSourceGAN,
  title         = {Enhancing Diversity and Feasibility: Joint Population Synthesis from Multi-source Data Using Generative Models},
  author        = {Anonymous},
  year          = {2026},
  archivePrefix = {arXiv},
  eprint        = {2602.15270},
  url           = {https://arxiv.org/abs/2602.15270}
}
```


[1]: https://www.researchgate.net/publication/319050699_Simulation_of_Synthetic_Complex_Data_The_R_Package_simPop "https://www.researchgate.net/publication/319050699_Simulation_of_Synthetic_Complex_Data_The_R_Package_simPop"
[2]: https://arxiv.org/pdf/2303.05515 "https://arxiv.org/pdf/2303.05515"
[3]: https://jasss.soc.surrey.ac.uk/18/2/21.html "https://jasss.soc.surrey.ac.uk/18/2/21.html"
[4]: https://marcgenton.github.io/2018.KHG.JASA.pdf "https://marcgenton.github.io/2018.KHG.JASA.pdf"
[5]: https://onlinelibrary.wiley.com/toc/20491573/2024/13/3 "https://onlinelibrary.wiley.com/toc/20491573/2024/13/3"
[6]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009799 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009799"
[7]: https://ideas.repec.org/a/eee/transb/v191y2025ics0191261524002522.html "https://ideas.repec.org/a/eee/transb/v191y2025ics0191261524002522.html"
[8]: https://ideas.repec.org/a/kap/transp/v50y2023i4d10.1007_s11116-022-10279-8.html "https://ideas.repec.org/a/kap/transp/v50y2023i4d10.1007_s11116-022-10279-8.html"
[9]: https://www.sciencedirect.com/science/article/pii/S0968090X25001299 "https://www.sciencedirect.com/science/article/pii/S0968090X25001299"
[10]: https://arxiv.org/abs/2309.02530 "https://arxiv.org/abs/2309.02530"
[11]: https://cran.r-project.org/web/packages/simPop/simPop.pdf "https://cran.r-project.org/web/packages/simPop/simPop.pdf"
[12]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf "https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf"
[13]: https://dl.acm.org/doi/full/10.1145/3742435 "https://dl.acm.org/doi/full/10.1145/3742435"
[14]: https://arxiv.org/html/2602.15270v1 "https://arxiv.org/html/2602.15270v1"
[15]: https://www.sciepub.com/reference/94917 "https://www.sciepub.com/reference/94917"
[16]: https://link.springer.com/book/10.1007/0-387-28678-0 "https://link.springer.com/book/10.1007/0-387-28678-0"
[17]: https://www.taylorfrancis.com/books/mono/10.1201/b17116/dependence-modeling-copulas-harry-joe "https://www.taylorfrancis.com/books/mono/10.1201/b17116/dependence-modeling-copulas-harry-joe"
[18]: https://www.sciencedirect.com/science/article/pii/S0167668707000194 "https://www.sciencedirect.com/science/article/pii/S0167668707000194"
[19]: https://arxiv.org/abs/2207.12598 "https://arxiv.org/abs/2207.12598"
[20]: https://proceedings.mlr.press/v202/avdeyev23a.html "https://proceedings.mlr.press/v202/avdeyev23a.html"
[21]: https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf "https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf"
[22]: https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf "https://www.jmlr.org/papers/volume9/garcia08a/garcia08a.pdf"

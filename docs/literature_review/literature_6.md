你这个“方法论演进 + 理论框架”的写法，我建议在申请书里用一条主线把 6 个方向串起来：**人口空间化/合成人口本质上都在做“从聚合到微观”的逆问题**，传统范式要么做“空间下推的点估计”（回归/加权再分配），要么做“边际约束下的重加权”（IPF/IPU/CO），它们共同的短板是：**很难在高维下稳定地恢复/表达联合分布与不确定性**；而深度生成模型（尤其扩散）提供了一个更自然的语言：**学习/表示联合分布，并在采样中纳入多源条件与聚合约束**。

下面按你给的 6 个调研方向，每个方向给 3–5 篇“可以放进基金本子里”的代表性工作。每篇都按：引用信息 → 核心贡献 → 方法论本质 → 与你们研究的关联 来写。

---

一段可以放在申请书“方法论对比/问题本质”里的小对照表（用来引出“从回归/匹配到生成”）

| 范式                                      | 典型代表                                                                 | 目标输出                 | 方法论本质                          | 典型局限（和你们论点的连接点）                                                                                   |
| --------------------------------------- | -------------------------------------------------------------------- | -------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| 空间下推/空间化（Spatialization/Disaggregation） | WorldPop / LandScan / building-footprint 下推                          | grid/建筑尺度人口“强度/计数”面  | 多源加权再分配 +（越来越多）机器学习回归预测权重      | 多为**点估计**与**质量守恒下推**，通常不直接生成个体层联合结构；对“谁住哪”的微观联合分布表达有限。([Nature][1])                               |
| 合成人口（IPF/IPU/CO 等）                      | IPF/IPU、多层重加权、组合优化                                                   | 个体/家庭微观样本            | **边际匹配**与约束优化（重加权/选样）          | 高维下会遇到**零单元/采样零**、多层约束不一致、相关结构受 seed 限制；更像“拟合边际”而不是“学习联合分布”。([ResearchGate][2])                   |
| 联合分布生成（Generative）                      | CTGAN / TabDDPM / TabSyn /（开始出现）diffusion-based population synthesis | 个体-建筑/空间的联合样本（可多次采样） | 显式/隐式**分布学习**（生成模型）+ 条件生成/采样引导 | 能把“多源条件 + 不确定性”纳入统一概率框架；关键在于如何把聚合约束与规则可行性内生到生成/采样。([Proceedings of Machine Learning Research][3]) |

---

## 方向1：人口空间化（Population Spatialization / Disaggregation）的方法演进

（从空间插值 → 多源加权 → 机器学习/深度学习）

1. Andrew J. Tatem et al. (2017). *WorldPop, open data for spatial demography*. **Scientific Data**. ([PubMed][4])
   核心贡献：系统阐述了 WorldPop 提供高分辨率人口栅格/属性数据的动机、数据产品与总体建模思路，强调“透明方法 + 开放数据”的空间人口学基础设施意义。
   方法论本质：典型是**基于协变量的统计学习/回归（常见是 RF 等）去预测栅格权重/密度，再做质量守恒的再分配（dasymetric downscaling）**。
   与你们研究关联：可用于论证“空间化主流路线是做**人口强度面/栅格面**（回归/加权），并不等价于学习建筑尺度的‘个体属性—居住单元’联合生成”。

2. Viswadeep Lebakula et al. (2025). *LandScan Global 30 Arcsecond Annual Global Gridded Population Datasets from 2000 to 2022*. **Scientific Data**. ([Nature][1])
   核心贡献：给出 2000–2022 的年度全球“环境人口（24 小时平均分布）”栅格产品，并说明其利用多源辅助数据在行政区内进行再分配的流程。
   方法论本质：典型的**多变量 dasymetric mapping（多源权重再分配）**，并包含专家知识/规则与验证流程。
   与你们研究关联：LandScan 代表“多源加权下推”的工程化路线：很强的空间产品能力，但仍主要在“聚合 → 栅格/单元强度”的层面；你们要做的“建筑物尺度人口画像（属性+空间联合）”可以据此强调是更进一步的微观联合建模问题。

3. Xiaojun Huang et al. (2021). *A 100 m population grid in the CONUS by disaggregating census data using Microsoft building footprints*.（常见归类为 Big Earth Data/相关地学期刊版本）([Scholar Commons][5])
   核心贡献：展示了利用大规模建筑物轮廓数据（building footprints）与土地利用信息，将美国人口统计在更高分辨率上进行下推的可行性框架。
   方法论本质：依然是**“行政区人口总量守恒”约束下的空间分配/再分配**，建筑物轮廓与用地更多扮演“约束与权重”的角色（可理解为回归/规则加权的组合）。
   与你们研究关联：这是“往建筑尺度走”的关键证据：领域在分辨率上已逼近建筑层，但大多仍停留在“把人**数量**分下去”，而不是生成“人—楼”的属性联合样本；这正好支撑你们提出“从空间下推到联合生成”的必要性。

4. Elias Pajares et al. (2021). *Population Disaggregation on the Building Level Based on Outdated Census Data*. **ISPRS International Journal of Geo-Information**. ([MDPI][6])
   核心贡献：提出在普遍存在“普查滞后/过时”的情况下，如何结合可获得的开放空间数据实现建筑/出入口层级的人口下推，并报告了建筑层与栅格层的拟合效果。
   方法论本质：**混合式下推（hybrid disaggregation）**：仍以守恒分配为核心，用辅助信息改善权重/可居住性判别；整体仍是“分配/匹配”范式，而非联合分布生成。
   与你们研究关联：非常贴合你们“建筑物尺度人口画像”的动机：它能说明“建筑级空间化在地学/规划界可做、也有人做”，但主流仍更像“后验分配/修正”，因此你们用扩散把“属性—空间—条件”放进统一生成过程，会显得更有方法论升级意味。

5. Yong Qiu et al. (2022). *Disaggregating population data for assessing progress of SDGs: methods and applications*. **International Journal of Digital Earth**. ([ResearchGate][7])
   核心贡献：从方法、辅助数据与产品角度系统总结人口下推/空间化研究脉络，并讨论其在 SDGs 等评估任务中的作用。
   方法论本质：把谱系讲得很清楚：从插值/守恒平滑 → dasymetric 权重 → 引入机器学习的权重/密度预测；但总体仍围绕“聚合统计守恒的空间重分配”。
   与你们研究关联：这是你们写“方法论演进”的关键综述引文：用它能很自然地说清楚“空间化的主流范式本质是回归/加权下推”，从而为“为什么要转向联合分布生成（尤其是人-楼联合）”做铺垫。

---

## 方向2：合成人口方法的局限性讨论

（IPF/IPU/CO 的理论基础与“只匹配边际、不学联合”的问题）

1. Kristy La et al. (2025). *Population synthesis: a problem-based review*. **Transport Reviews**. ([ResearchGate][2])
   核心贡献：以“问题驱动”的方式总结人口合成在数据限制、异质性、维度增长（维度灾难）与适应性等方面的挑战与机会。
   方法论本质：这是“领域自我反思”的综述，明确人口合成并非单纯算法问题，而是被“可获得数据的粒度/偏差/稀疏性”深刻限制。
   与你们研究关联：可以直接引用来支撑你们的论点：传统边际匹配方法在高维与多源条件下会遇到结构性瓶颈，因此需要更强的联合分布建模范式（为扩散/生成方法铺路）。

2. Xin Ye et al. (2022). *On Iterative Proportional Updating (IPU) and Solution Existence of the Household Synthesis Problem*. **IEEE Transactions on Cybernetics**. ([PubMed][8])
   核心贡献：从理论上讨论 IPU 在家庭合成问题上的“解存在性/收敛性”与局限，并提出基于双层优化的改进框架。
   方法论本质：非常典型的“**边际约束下的重加权/拟合**”视角，并明确指出在多层（个体+家庭）约束同时存在时，传统 IPU 可能无法保证找到全局最优或甚至可行解。
   与你们研究关联：这是你要的“直接讨论传统方法局限”的硬证据之一：你们可以据此论证——当约束更复杂（例如再加入建筑容量、用地规则、空间偏好）时，纯粹的 IPU/IPF 会更不稳定，必须考虑能表达联合结构与可行性空间的生成式模型。

3. Nicholas Fournier et al. (2021). *Integrated population synthesis and workplace assignment using an efficient optimization-based person-household matching method*. **Transportation**. ([ResearchGate][9])
   核心贡献：把“合成人口”与“工作地分配”更紧密地耦合，减少“先合成再分配”导致的解不一致，并用优化式匹配方法提升可扩展性。
   方法论本质：依然属于**约束优化/匹配**路线（把多个模块合并求解），并强调在更大矩阵、更复杂 joint fitting 下的计算可行性。
   与你们研究关联：这篇非常适合用来说明“领域确实意识到‘生成与分配分离’会引入误差，并有人尝试做集成”，从而顺势提出：扩散式联合生成是比“更大规模的优化匹配”更具表达力的下一代路径。

4. Jaewoong Kang et al. (2023). *Generating Population Synthesis Using a Diffusion Model*. **Winter Simulation Conference (WSC 2023)**. ([INFORMS Simulation Society][10])
   核心贡献：把 DDPM 引入人口合成，展示扩散模型在合成微观样本方面的可行性，并强调对“采样零（sampling zeros）”等问题的潜在优势。
   方法论本质：从“边际拟合/重加权”转向“**分布学习（生成式）**”；尽管它的工程处理（把个体表格转成矩阵等）不一定是最终形态，但它是领域内明确的扩散先行工作。
   与你们研究关联：可作为“扩散用于人口合成已有先例”的关键引文；你们的创新点就可以进一步聚焦到“建筑物尺度 + 多源条件 + 聚合约束引导”的系统化框架，而不只是“把扩散用在 tabular 合成”。

5. Min Tang et al. (2025). *Generating Feasible and Diverse Synthetic Populations*（arXiv 预印本）. ([arXiv][11])
   核心贡献：明确把人口合成中的核心张力表述为“**可行性（feasibility） vs 多样性（diversity）**”，并讨论如何在生成过程中恢复缺失的 sampling zeros、同时尽量避免产生 structural zeros。
   方法论本质：典型的“生成式建模 + 约束/可行性控制”的视角（更贴近你们想讲的“约束内生”叙事）。
   与你们研究关联：它几乎可以直接当作你们申请书里“为什么需要联合分布生成范式”的最新支撑：传统方法在零单元与高维相关结构上受限，而扩散方法开始把“联合分布恢复 + 约束控制”当成中心问题来研究。

---

## 方向3：生态推断问题（Ecological Inference）

（从聚合数据推断个体行为：与人口空间化/合成的关系）

1. Gary King (1997). *A Solution to the Ecological Inference Problem: Reconstructing Individual Behavior from Aggregate Data*. Cambridge University Press. ([GKing][12])
   核心贡献：提出了经典 EI 框架，系统讨论“从聚合到个体”的不可识别性、约束边界与统计建模策略，在社会科学与计量中影响很大。
   方法论本质：严格的**跨层推断（cross-level inference）**问题建模：你只能看到聚合表，但你关心的是个体层关联结构。
   与你们研究关联：你们的“从 ACS tract 边际推断建筑/个体画像”本质上就是 EI 的变体。引用 King 可以把问题“抬升”为一个经典逆问题，从理论上说明：如果只做边际匹配或回归点估计，很难表征不确定性与多解性，因此需要概率生成式框架。

2. Ben Fishman et al. (2021). *Deep Ecological Inference*.（ICLR 2021, OpenReview）([ResearchGate][13])
   核心贡献：把 EI 形式化为深度学习可处理的推断/生成问题，面向 RxC（多类别）生态表等更一般情形。
   方法论本质：从传统 EI 的统计建模转向“**神经网络参数化的分布/推断器**”，本质是在学习“与聚合观测一致的微观生成机制/后验”。
   与你们研究关联：这是“EI 的深度学习版本”里非常贴近你们叙事的一篇：你们可以用它说明——把“聚合约束”当作学习/生成的一部分（而非后处理）在方法论上是合理且已有 ML 先例的。

3. Lukasz McCartan & Shiro Kuriwaki (2025). *Identification and Semiparametric Estimation of Conditional Means from Aggregated Data*（arXiv）. ([Cory McCartan][14])
   核心贡献：聚焦“只有聚合数据时，想识别/估计个体层条件均值”这一类目标，讨论识别条件、敏感性与半参数估计。
   方法论本质：更偏“识别理论 + 估计”，强调在聚合观测下推断微观量需要额外假设，并讨论可检验/不可检验部分。
   与你们研究关联：这篇可用来写“理论风险声明”：为什么你们必须引入多源条件（建筑、POI、移动行为）作为识别信息，且用生成模型输出分布而非单点——因为聚合到个体天然存在不可识别区域，需要用信息与先验去收缩解空间。

4. Jose Maria Pavía & Rafael Romero (2023). *Ecological inference forecasting for RxC contingency tables: A comparison of existing methods*（SORT）. ([Semantic Scholar][15])
   核心贡献：比较 RxC 表生态推断的多种方法（统计/贝叶斯/计算策略），面向实际预测任务给出经验结论。
   方法论本质：强调 EI 在多类别情形下的计算复杂度与方法差异。
   与你们研究关联：可以用来支持你们“tract × 多属性边际约束”的实际复杂性：一旦属性维度上去，传统方法的计算与稳定性都会成为瓶颈，进而为“生成式 + 采样”的路线提供论据。

---

## 方向4：深度生成模型在表格/地理数据中的应用

（GAN/VAE/Diffusion 用于 tabular；是否用于人口合成/地理生成）

1. Leyan Xu et al. (2019). *Modeling Tabular Data using Conditional GAN*（CTGAN）. **NeurIPS 2019**. ([NeurIPS Papers][16])
   核心贡献：针对表格数据混合类型、类别不平衡等问题，提出 CTGAN 的条件机制与训练技巧，成为 tabular 合成的经典深度基线。
   方法论本质：**生成对抗网络的分布学习**（conditional generation）。
   与你们研究关联：在基金本子里它适合当“传统深度生成（GAN）代表”，用来对比说明：近年来扩散在 tabular 上逐渐占优，且更便于做条件控制与采样引导（为你们选 Diffusion 作为核心方法做铺垫）。

2. Akim Kotelnikov et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. **ICML 2023 (PMLR)**. ([Proceedings of Machine Learning Research][3])
   核心贡献：把 DDPM 系统性落到 tabular 数据（混合连续/离散处理）上，在多数据集基准上展示强性能，并讨论隐私场景适配。
   方法论本质：**扩散模型的分布学习**（score/denoise 视角）。
   与你们研究关联：TabDDPM 可以作为“扩散做表格联合分布学习”的权威代表，引出你们把人口属性（tabular）与建筑特征（tabular/geospatial embedding）放进同一联合生成框架的合理性。

3. Hengrui Zhang et al. (2024). *Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space*（TabSyn，ICLR 2024 体系）. ([OpenReview][17])
   核心贡献：提出“VAE 表示学习 + latent diffusion”的两阶段方案，增强混合类型 tabular 的生成质量。
   方法论本质：**表示学习（AE/VAE） + 扩散的联合分布学习**。
   与你们研究关联：TabSyn 这类“latent diffusion + 结构化条件”对你们很有启发：建筑/地理信息往往需要先编码成合适的潜变量，再与人口属性一起做联合生成；这条路线在写作上也更像“多源大数据融合”的技术范式。

4. Aoxiang Wang et al. (2024). *Challenges and opportunities of generative models on tabular data synthesis*（综述/评测向）. ([ScienceDirect][18])
   核心贡献：系统讨论 tabular 生成模型的评价、挑战（异质类型、相关结构、可控性、可用性等）与机会。
   方法论本质：站在更高层面对“从传统统计/回归到生成模型”的迁移进行梳理（至少在 tabular 合成领域是显性转向）。
   与你们研究关联：这是你问的“有没有直接讨论从回归到生成的转向”里，最接近“可直接引用”的材料之一：你们可以用它把“生成式学习联合分布”的必要性写得更像主流 ML 叙事，而非仅仅是工程选择。

5. Kang et al. (2023). *Generating Population Synthesis Using a Diffusion Model*（WSC 2023）. ([INFORMS Simulation Society][10])
   核心贡献/本质/关联：见方向2第4条。之所以在这一节再出现，是因为它把“tabular diffusion → synthetic population”这条链条补齐了：你们不仅是“借鉴 tabular diffusion”，而是把它进一步推进到“人—楼联合画像”的新任务形态。

---

## 方向5：移动大数据与人口属性推断

（手机信令/GPS/POI 行为与 demographic 的关联）

1. John Jay et al. (2020). *Neighbourhood income and physical distancing during the COVID-19 pandemic in the United States*. **Nature Human Behaviour**. ([Nature][19])
   核心贡献：用大规模手机移动性数据刻画疫情期间的居家/出行行为差异，发现与社区收入存在显著梯度关系。
   方法论本质：**统计分析 + 回归/关联推断**（行为数据 → 社经变量差异）。
   与你们研究关联：这类工作可以支撑你们“多源行为数据能为人口属性提供强信息”的论点：移动行为与社会经济属性并非弱相关，从而为你们把“mobility/POI/建成环境”作为条件输入（或弱监督约束）提供经验依据。

2. Esteban Moro et al. (2021). *Mobility patterns are associated with experienced income segregation in large US cities*. **Nature Communications**. ([Nature][20])
   核心贡献：强调“经历到的隔离（experienced segregation）”不仅与居住分布有关，也与个体移动行为模式强相关。
   方法论本质：**网络/行为特征建模 + 关联分析**。
   与你们研究关联：非常适合写进“联合分布重要性”的段落：如果只在居住地边际上匹配，可能错过由行为驱动的交互结构；你们做“建筑尺度画像”若能引入行为或 POI 访问先验，会更像在恢复真实联合机制。

3. Emily Aiken et al. (2022). *Machine learning and phone data can improve targeting of humanitarian aid*. **Nature**. ([Nature][21])
   核心贡献：展示移动通信数据可与调查数据结合，用机器学习识别贫困/脆弱群体，从而提升援助投放的 targeting。
   方法论本质：**监督学习/预测**（把移动特征映射到人口属性/福利指标）。
   与你们研究关联：这是“移动数据 → demographic/贫困属性”在顶刊上非常硬的证据。你们可以用它说明：多源大数据确实携带人口属性信号，因此“在生成模型里把这些信号作为条件”不是凭空假设。

4. Ekin Uğurel et al. (2025). *On Predicting Sociodemographics from Mobility Signals*（arXiv）. ([arXiv][22])
   核心贡献：更系统地讨论从 mobility 信号预测年龄/性别/收入等人口属性的特征构造、泛化与不确定性诊断。
   方法论本质：**特征工程/表示学习 + 多任务预测 + 不确定性评估**。
   与你们研究关联：两点很有用：第一，它让“从行为推断属性”更方法论化；第二，它强调不确定性与校准，这与生成式框架天然契合——你们可以把“预测不确定性”升级为“生成分布的不确定性”。

---

## 方向6：城市系统的联合分布/协变结构研究

（为什么“联合分布/尾部/协变”比“边际/均值”更关键）

1. Martin Hendrick et al. (2025). *A stochastic theory of urban metabolism*. **Proceedings of the National Academy of Sciences (PNAS)**. ([PubMed][23])
   核心贡献：批判传统城市尺度律对“城市边界定义”与“忽略城市内部变异”的偏差，转向更随机/概率化的城市代谢理论表述。
   方法论本质：强调用**随机过程/概率分布（包含城市内部变量的边际与关联结构）**来理解城市指标，而不是只看汇总量或均值关系。
   与你们研究关联：这篇非常适合支撑你们“为什么要从边际到联合分布”的宏观论证：城市系统里很多规律来自**内部异质性与协变结构**，如果方法只对齐边际，很可能在机制层面失真。

2. Moa Arvidsson et al. (2023). *Urban scaling laws arise from within-city inequalities*. **Nature Human Behaviour**. ([Nature][24])
   核心贡献：指出许多城市尺度律现象在很大程度上由城市内部的“重尾分布/不平等尾部”驱动，不能只用平均增长解释。
   方法论本质：把关注点从“边际均值/总量的幂律”转向“**分布形态（尤其尾部）随规模变化**”。
   与你们研究关联：你们想强调的“联合分布与协变结构”在这里有非常直观的城市科学证据：真实世界的结构性差异往往在分布尾部和变量耦合里，边际匹配容易把这些抹平。

3. Andres Gomez-Lievano et al. (2012). *The Statistics of Urban Scaling and Their Connection to Zipf’s Law*. **PLOS ONE**. ([PLOS][25])
   核心贡献：构建了更自洽的统计框架来刻画城市指标与城市规模的关系，讨论条件分布统计与尺度律、Zipf 定律之间的联系。
   方法论本质：显式讨论**条件分布/统计框架**，比“线性回归一条幂律线”更接近你们想要的“联合分布视角”。
   与你们研究关联：用作经典理论背书很合适：你们可以说，“城市规律研究本身就需要用分布而非单点关系来表述；人口画像生成同理，应以联合分布生成替代单纯边际对齐”。

---

## 对你们申请书写作的具体建议（把 6 条线索收束成“转向联合生成”的理论框架）

第一，把问题定义成“生态推断 + 约束生成”的统一逆问题。写法上可以先引用 King（1997）把“聚合→微观”提升为经典 EI 问题，再接 Deep Ecological Inference（Fishman et al., 2021）说明 ML 已经在把“聚合约束”纳入可学习的生成/推断框架。([GKing][12])

第二，用两段话分别“温和但明确”地指出空间化与合成人口的范式边界：

* 空间化（WorldPop/LandScan/建筑轮廓下推）强在多源空间特征与高分辨率产品，但主要输出强度面/计数面，方法论仍是回归/加权再分配。([Nature][1])
* 合成人口（IPF/IPU/CO）强在边际一致性与可控性，但在多层约束与高维相关结构上有解存在性与稳定性瓶颈，且容易被 sampling zeros/结构稀疏性困住。([PubMed][8])

第三，把“从回归/匹配到生成”写成一种必然的工程与科学升级，而不是“跟风深度学习”。最好用三组证据支撑：

* Tabular 合成领域：扩散（TabDDPM/TabSyn）已成为强基线，综述也在总结生成模型的机会与挑战；说明这是一个成熟方向而非偶然选择。([Proceedings of Machine Learning Research][3])
* 人口合成领域：已经出现直接用扩散做人群合成与讨论可行性/多样性的工作（Kang 2023；Tang 2025），说明“联合分布生成”正在成为人口合成的新语言。([INFORMS Simulation Society][10])
* 城市科学：从均值到分布、从边际到协变结构的转向同样存在（Arvidsson 2023；Hendrick 2025），给你们“联合分布的重要性”一个更宏观的学理支撑。([Nature][24])

第四，把“多源大数据融合”写成“识别信息”而不是“更多特征”。用 Aiken（2022）这类顶刊证据说清：移动/行为数据确实携带贫困/人口属性信号；因此你们把 POI、mobility、建成环境作为条件或弱监督，是在为 EI 的不可识别性提供额外信息，缩小可行解空间。([Nature][21])

最后，建议在“创新点”里用一句话钉住你们最核心的转向：
**从“边际一致的空间下推/重加权”转向“面向建筑物尺度的（人-楼-条件）联合分布生成，并在生成过程中内生满足聚合约束与可行性规则”。**
你们内部的架构材料里其实已经把“叙事需要联合生成、约束需要内生满足”写得很直白了，可以直接抽象成申请书的理论框架表述。

---




[1]: https://www.nature.com/articles/s41597-025-04817-z "https://www.nature.com/articles/s41597-025-04817-z"
[2]: https://www.researchgate.net/publication/389527876_Population_synthesis_a_problem-based_review "https://www.researchgate.net/publication/389527876_Population_synthesis_a_problem-based_review"
[3]: https://proceedings.mlr.press/v202/kotelnikov23a.html "https://proceedings.mlr.press/v202/kotelnikov23a.html"
[4]: https://pubmed.ncbi.nlm.nih.gov/28140397/ "https://pubmed.ncbi.nlm.nih.gov/28140397/"
[5]: https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1238&context=geog_facpub "https://scholarcommons.sc.edu/cgi/viewcontent.cgi?article=1238&context=geog_facpub"
[6]: https://www.mdpi.com/2220-9964/10/10/662 "https://www.mdpi.com/2220-9964/10/10/662"
[7]: https://www.researchgate.net/publication/358048149_Disaggregating_population_data_for_assessing_progress_of_SDGs_methods_and_applications "https://www.researchgate.net/publication/358048149_Disaggregating_population_data_for_assessing_progress_of_SDGs_methods_and_applications"
[8]: https://pubmed.ncbi.nlm.nih.gov/32479409/ "https://pubmed.ncbi.nlm.nih.gov/32479409/"
[9]: https://www.researchgate.net/publication/339342455_Integrated_population_synthesis_and_workplace_assignment_using_an_efficient_optimization-based_person-household_matching_method "https://www.researchgate.net/publication/339342455_Integrated_population_synthesis_and_workplace_assignment_using_an_efficient_optimization-based_person-household_matching_method"
[10]: https://informs-sim.org/wsc23papers/247.pdf "https://informs-sim.org/wsc23papers/247.pdf"
[11]: https://www.arxiv.org/abs/2508.09164 "https://www.arxiv.org/abs/2508.09164"
[12]: https://gking.harvard.edu/files/gking/files/part1.pdf "https://gking.harvard.edu/files/gking/files/part1.pdf"
[13]: https://www.researchgate.net/publication/394386747_Ecological_Inference_for_Electoral_Analysis_A_Computational_Perspective_on_Human_Decision-Making "https://www.researchgate.net/publication/394386747_Ecological_Inference_for_Electoral_Analysis_A_Computational_Perspective_on_Human_Decision-Making"
[14]: https://corymccartan.com/seine/articles/seine.html "https://corymccartan.com/seine/articles/seine.html"
[15]: https://www.semanticscholar.org/paper/Identification-and-Semiparametric-Estimation-of-McCartan-Kuriwaki/ed0170522586c06bb246caedf3dfb979e15b4b09 "https://www.semanticscholar.org/paper/Identification-and-Semiparametric-Estimation-of-McCartan-Kuriwaki/ed0170522586c06bb246caedf3dfb979e15b4b09"
[16]: https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan "https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan"
[17]: https://openreview.net/forum?id=4Ay23yeuz0 "https://openreview.net/forum?id=4Ay23yeuz0"
[18]: https://www.sciencedirect.com/science/article/pii/S1568494624009979 "https://www.sciencedirect.com/science/article/pii/S1568494624009979"
[19]: https://www.nature.com/articles/s41562-020-00998-2 "https://www.nature.com/articles/s41562-020-00998-2"
[20]: https://www.nature.com/articles/s41467-021-24899-8 "https://www.nature.com/articles/s41467-021-24899-8"
[21]: https://www.nature.com/articles/s41586-022-04484-9 "https://www.nature.com/articles/s41586-022-04484-9"
[22]: https://arxiv.org/abs/2511.03924 "https://arxiv.org/abs/2511.03924"
[23]: https://pubmed.ncbi.nlm.nih.gov/40789025/?utm_source=chatgpt.com "A stochastic theory of urban metabolism"
[24]: https://www.nature.com/articles/s41562-022-01509-1?utm_source=chatgpt.com "Urban scaling laws arise from within-city inequalities"
[25]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0040393&utm_source=chatgpt.com "The Statistics of Urban Scaling and Their Connection to Zipf's ..."

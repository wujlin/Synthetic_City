# 调研报告：[任务 L1] 合成人口方法中的“空间分配”处理方式

## 核心发现（3-5 句话）

经典合成人口（synthetic population）方法（IPF/IPU/CO 等）通常把“空间单元（zone/tract/TAZ）”当作生成时的外层索引：先在每个空间单元内拟合边际分布并抽样/重加权得到该单元的人口与家庭，从而天然完成“个体→空间单元”的分配。([STRC][1])
当目标是更细粒度（parcel/building/address）的落点时，领域里更常见的是把“生成（属性与家庭结构）”和“落位（location/allocation）”分离：用容量、土地利用、重力模型或离散选择模型做后处理落位，而不是把精确地址直接内嵌到主合成器里。([Taylor & Francis Online][2])
确实存在“把空间分配更深度内嵌”的方向，典型是把居住地/就业地匹配、跨尺度一致性（multi-resolution controls）一起纳入优化或联合建模，但往往依赖更强的辅助数据与更复杂的校准。([prapare.org][3])
因此，你们的“分离生成与分配（先生成属性/家庭，再做空间落位）”总体符合交通/城市微观仿真与空间微观模拟（spatial microsimulation）中的主流工程范式，尤其在建筑级数据不完备时更合理。([STRC][1])

## 详细内容

### 子问题1：经典方法（IPF/IPU/CO）如何处理“个体→空间单元”的分配问题？

1. IPF / Synthetic Reconstruction（合成重建）的典型流程是：对每个空间单元（如 tract/TAZ）用边际约束拟合权重，再从样本微数据（seed）抽取/整数化，得到该空间单元内的家庭与个体列表；在这个意义上，“分配到空间单元”是生成过程的一部分，而不是额外一步。([STRC][1])

2. IPU（Iterative Proportional Updating）的动机是同时控制 household-level 与 person-level 的边际/联合特征。Ye et al. 的路线是给 seed household/individual 赋权，使得在目标区域内同时匹配家庭边际与人口边际，从而得到“某空间单元内”的合成人口。([JASSS][4])

3. CO（Combinatorial Optimisation）通常把“拟合”写成一个搜索/优化问题（以误差指标如 TAE 等为目标），在小区域上直接选择/复制微数据记录，逼近目标表；它同样默认“每个区域各自拟合”，因此空间单元级别的分配也是内生的。([Wiley Online Library][5])

### 子问题2：是否存在将空间分配内嵌到生成过程的方法？还是普遍采用后处理？

更细粒度空间（parcel/building/address）层面，文献与实践里常见两条路径：

A) 更“内嵌”的做法（把空间落点或空间关系纳入联合问题）

* 多尺度一致性：在不同地理尺度同时施加控制（例如既匹配 tract 又匹配更粗/更细尺度的约束），让合成结果在尺度间一致，从而减少后续再分配的自由度与不稳定性。([ResearchGate][6])
* 联合居住地/就业地或 OD 关系：把 workplace assignment 或通勤关系与人口合成一起做（而不是先合成人再另行分配工作地），属于更强的“空间关系内嵌”。([prapare.org][3])

B) 更普遍的做法（生成与落位分离，落位作为后处理/下游模块）

* 在交通与城市仿真中，一个常见工程结构是：先得到“每个 zone 的 households/persons”，再用土地利用/建筑/容量/吸引力等信息，把这些 agents 进一步落到更细空间（parcel/building/grid）。以 Microsoft building footprints + ACS 做细网格人口分解就是这类“生成（总量/属性）+空间分解（落位）”范式的代表之一。([Taylor & Francis Online][2])
* LODES 这种工作地—居住地（block-level OD）数据也常被当作“空间分配/校准模块”的外部锚点，而不是合成人口器本身的一部分。([lehd.ces.census.gov][7])

总体判断：**空间单元（tract/TAZ）级别“内嵌”，建筑/地址级别多为“后处理或下游模块”**，除非你具备非常强的建筑级容量/用途/可达性/价格等数据并愿意做离散选择/约束优化的系统校准。([Taylor & Francis Online][2])

### 子问题3：如果是后处理，常用的分配规则有哪些？（capacity-based, utility-based, gravity model...）

这里把“把已生成的 agents 落到更细空间单元 s（building/parcel/grid）”抽象成：对每个 agent i，在候选集合 S(i) 上选一个落点，并满足一些约束。

常用规则可以分四类（可以混合）：

1. Capacity-based（容量约束/装箱式）

* 每个空间单元 s 有容量 cap(s)（住房单元数、可居住面积、住宅建筑数、地址点数等），按随机或加权随机把 households 填入，直到容量用尽；必要时做迭代再平衡。
* 优点：实现简单，和“住宅容量/空置率”这类数据天然兼容；缺点：很难保证更丰富的空间相关属性（如收入与地段）匹配。
  （这类做法与“基于建筑足迹把 tract/block group 人口分解到细网格/建筑”的思路高度一致。）([Taylor & Francis Online][2])

2. Utility-based / discrete choice（效用模型/离散选择落位）

* 给每个 (i,s) 定义效用 U(i,s)=β·x(i,s)+…，按 logit/probit 采样；可加入容量约束形成 capacitated choice。
* 适合结合地价、可达性、学区、建筑属性等“解释变量”，代价是需要估计/校准。

3. Gravity / friction（重力模型/摩擦因子）

* 常用于“工作地/学校地”分配或 OD 关系生成：吸引力 A(s) 与距离衰减 f(d) 共同决定概率。
* 在把居住人口从粗区分配到细网格时，也常把“住宅可能性”作为 A(s)，把到中心/道路的距离作为修正。([lehd.ces.census.gov][7])

4. Rule-based + constraint satisfaction（规则+约束满足）

* 例如“有孩家庭优先落到住宅用地且容量足够的地块”“老年人更靠近医疗设施”等；常与上面三类叠加，先筛候选再抽样。

## 对我们项目的建议

1. 如果你们的主目标是“Detroit/Wayne County 的 tract/TAZ 级人口一致 + 后续可做建筑级落位”，建议继续采用**两段式**：
   第一段用 IPU/IPF/CO 生成 tract/TAZ 内的 households/persons；第二段用 building footprints/parcel/地址点做 capacity/land-use 引导的落位。这样与领域常见做法一致，且更易调试。([STRC][1])

2. 若你们非常关心“居住地—工作地”一致性（通勤 OD、就业空间分布），可以把 LODES 当作空间锚点：要么在后处理里做 workplace assignment（重力/约束匹配），要么走“联合合成+工作地分配”的更内嵌路线（复杂度更高）。([Census.gov][8])

3. 落位后验证要尽量使用“容量一致性（住房单元数/住宅建筑密度）+宏观边际一致性（ACS）+（如有）外部微观锚点（选民/房产）”的多源交叉验证框架，避免只看边际误差。([Taylor & Francis Online][2])

## 参考文献

* Müller, K., & Axhausen, K. W. Population synthesis for microsimulation: State of the art（working paper, 2010；常被作为 2010/2011 “state of the art” 引用）。([研究收藏][9])
* Ye, X., Konduri, K., Pendyala, R. M., Sana, B., & Waddell, P. Methodology to match distributions of both household and person attributes…（IPU）。([JASSS][4])
* Pritchard, D., & Miller, E. J. Advances in population synthesis… Transportation 39(3), 2012。([IDEAS/RePEc][10])
* Voas, D., & Williamson, P. An evaluation of the combinatorial optimisation approach to the creation of synthetic microdata, 2000。([Wiley Online Library][5])
* Huang, X. et al. 用 Microsoft building footprints + ACS 做 CONUS 100m 人口网格分解（2020/2021）。([Taylor & Francis Online][2])
* LODES/LEHD（block-level OD 数据说明与下载结构）。([lehd.ces.census.gov][7])

---

# 调研报告：[任务 L2] 扩散模型在表格数据/半监督场景的应用

## 核心发现（3-5 句话）

TabDDPM、STaSy、TabSyn 等工作把扩散/score-based 生成带入表格数据，核心挑战是“混合数据类型（连续+离散）”与“小数据集稳定训练”。([Proceedings of Machine Learning Research][11])
离散变量处理出现了两条主线：一是把 categorical 当作连续（one-hot/embedding）一起做 score matching（STaSy 的典型路线），二是显式用离散扩散（multinomial / discrete-state diffusion，如 TabDDPM 的 categorical 分支、D3PM 的通用离散扩散框架）。([arXiv][12])
“部分标签缺失”可视作“条件缺失/部分条件可用”的条件生成问题：训练时做 condition dropout（classifier-free 思路）或把 label 当作待补全字段联合生成；近期表格扩散在“缺失值补全/动态掩码”上已有较系统方案，可自然迁移到缺失标签。([arXiv][13])
关于“多粒度/层次化条件”（sample-level 与 group-level 约束并存），已有工作开始讨论可控生成与关系/跨元组约束，但对“层次化条件=多尺度约束”仍偏早期，更多是可作为你们研究切入点的空白。([dbgroup.cs.tsinghua.edu.cn][14])

## 详细内容

### 子问题1：技术方案对比（TabDDPM vs STaSy vs D3PM vs TabSyn）

下面表格聚焦“噪声空间/离散处理/条件机制/优缺点”，便于你直接映射到你们的半监督与约束场景。

| 方法                 | 核心生成空间与噪声过程                                                          | 离散变量处理                                           | 条件生成/缺失条件                                       | 主要优缺点                                                                                          |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| TabDDPM (2023)     | 以 DDPM 思路做表格建模；连续特征走高斯扩散分支                                           | categorical 走“离散/多项式（multinomial）”扩散建模（与连续分支并行）  | 可做条件生成（把条件列拼接/输入网络）；缺失条件可用“mask/丢弃条件”技巧扩展       | 优：结构相对直接、可显式处理离散；缺：不同数据集稳定性可能波动，且小数据/高基数类别仍挑战 ([Proceedings of Machine Learning Research][11]) |
| STaSy (2022)       | score-based（SDE）范式，在连续空间学习 score                                     | 常见实现是把 one-hot/embedding 当作连续特征一起做 score（更“连续化”） | 条件机制可通过拼接与条件网络实现；缺失条件可通过 condition dropout/掩码处理 | 优：score-based 训练技巧（如自步训练等）有助稳定；缺：把类别当连续可能带来语义不稳/边界问题 ([arXiv][15])                             |
| D3PM (2021)        | 通用离散状态空间的扩散（structured denoising diffusion in discrete state-spaces） | 原生支持离散 token（适合纯离散或把列离散化后建模）                     | 条件生成可通过条件转移/条件网络扩展                              | 优：理论上更“正统”的离散扩散；缺：混合连续变量时需与连续分支/混合框架结合 ([arXiv][16])                                           |
| TabSyn (ICLR 2024) | 先把混合表格编码到 latent space，在 latent 上做 score-based diffusion，再解码回数据空间    | 通过 latent 表示弱化“类别连续化”的直接问题；更强调 mixed-type 的结构化处理 | 支持条件生成；也更容易把“掩码/条件注意力”放到 latent 生成里             | 优：对 mixed-type 往往更稳、更强；缺：引入编码器/解码器，训练与失真控制更复杂 ([assets.amazon.science][17])                    |

补充：2025 以后也出现把“扩散（连续）+（非）自回归/Transformer（离散）”更紧耦合的框架，例如 TabNAT 试图把连续与离散的生成方式做成一个统一系统（并强调缺失数据补全场景）。([OpenReview][18])

### 子问题2：TabDDPM 之后有哪些改进工作？特别是处理离散变量的方法

从你们关心的“离散变量”出发，近两年的改进大致分三类：

1. 连续-离散联合框架
   TabNAT 明确把异构表格分成连续与离散，并使用 masked Transformer 学 conditional embedding，再分别用条件扩散与类别分布学习处理两类变量。([OpenReview][18])

2. 更强的条件建模与动态掩码（对缺失/半监督很关键）
   有工作在表格扩散中加入 conditioning attention、encoder-decoder Transformer 作为去噪网络、以及 dynamic masking，以提升“条件-生成变量”的耦合与缺失补全能力。([arXiv][13])

3. 可控生成与约束（从 sample-level 条件走向更复杂约束）
   SIGMOD 2024 的工作把“可控表格合成”作为核心问题，讨论了用户指定条件，并引入控制器去满足 intra-table / inter-table 条件（更接近你们想要的“多粒度约束”方向）。([dbgroup.cs.tsinghua.edu.cn][14])

### 子问题3：扩散模型如何处理“部分标签缺失”的场景？

这里给你一个“可落地的三种范式”，并说明各自的优缺点（不局限于图像，表格同理）：

范式 A：把 label 当作条件 c，做 classifier-free / condition dropout
训练时随机丢弃条件，让同一个模型学到 p(x|c) 与 p(x) 两种模式；推断时：

* label 已知：用条件分支；
* label 缺失：退化为无条件生成或对 label 做枚举/采样再生成。
  这是处理“缺失标签=缺失条件”的最直接办法。

范式 B：把 label 当作“需要补全的一列”，做联合生成/补全
把 (x, y) 合成一个表，训练模型对被 mask 的字段做去噪/补全；当 y 缺失时，把 y 置为 mask，让模型同时补全 y 并生成/补全 x。
表格扩散里“缺失值补全 + 动态掩码”的设计，天然支持把“缺失标签”当作特殊缺失值来处理。([arXiv][13])

范式 C：半监督学习中的生成式伪标签/增强
在半监督设置中，可以用扩散模型生成增强样本或生成伪标签并迭代训练；这在更一般的半监督扩散框架里已有系统化讨论（可借鉴其训练/置信度过滤策略）。

### 子问题4：条件扩散模型中，条件信息粒度选择（sample-level vs group-level）有什么讨论？

你们的“层次化条件/多粒度约束”可以拆成两种条件：

* Sample-level（行级条件）：例如类别标签、是否某人群、是否某区等，适合直接拼接到去噪网络输入或用条件注意力。([arXiv][13])
* Group-level（组级/集合级条件）：例如“某个 tract 的年龄边际/收入边际必须满足”“某个 group 的总量约束”，本质是对一批样本的联合约束。

已有进展主要在“可控生成/关系约束”：用控制器或引导项把生成拉向满足某些条件的集合，甚至处理跨表（inter-table）条件。([dbgroup.cs.tsinghua.edu.cn][14])
但严格意义上把“多尺度边际约束（group-level）”做成扩散模型的原生条件接口（而非后验筛选/重采样）仍是空白偏多的方向——这点反而对你们是机会：你们可以把“多尺度约束”作为一种 structured condition embedding，并结合投影/拉格朗日引导或可微约束项。([Emergent Mind][19])

## 对我们项目的建议

1. 如果你们的半监督场景是“部分样本缺 label / label 缺失但特征齐全”，优先考虑 **范式 A（condition dropout）+ 范式 B（mask 补全）** 的组合：训练时随机 mask 标签列与若干特征列，让模型习得“补全/生成”的统一能力。([arXiv][13])

2. 若你们还要满足“区域级（group-level）边际/约束”，建议把它视为“控制问题”而不是单纯条件输入：可以参考可控表格合成里“控制器+生成器”的分工，把约束满足放到显式模块中（比如 guidance/控制器），减少主扩散模型背负的硬约束压力。([dbgroup.cs.tsinghua.edu.cn][14])

3. 离散变量很多/高基数类别多时，优先选 **显式离散建模（TabDDPM 的 multinomial 分支、或 D3PM 思路）** 或 **latent-space（TabSyn）**，尽量避免简单把 one-hot 当连续导致的边界问题。([Proceedings of Machine Learning Research][11])

## 参考文献

* Kotelnikov et al. TabDDPM: Modelling Tabular Data with Diffusion Models (PMLR 2023).([Proceedings of Machine Learning Research][11])
* Kim et al. STaSy: Score-based Tabular data Synthesis (arXiv 2022).([arXiv][15])
* Austin et al. Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM, 2021).([arXiv][16])
* Zhang et al. Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space (TabSyn, ICLR 2024).([assets.amazon.science][17])
* Ho & Salimans. Classifier-Free Diffusion Guidance (2022).
* Liu et al. Controllable Tabular Data Synthesis Using Diffusion Models (SIGMOD 2024).([dbgroup.cs.tsinghua.edu.cn][14])
* TabNAT (2025).([OpenReview][18])

---

# 调研报告：[任务 L3] 建筑级/地址级人口数据的学术应用

## 核心发现（3-5 句话）

“建筑级人口数据”在学术上更常以两种形式出现：一是直接估计 building-level/细网格人口（由建筑足迹、遥感、调查或 census/ACS 聚合数据下推得到），二是把建筑/地址作为承载体，用作“把 tract/block group 人口空间分解”的落位载体。([Taylor & Francis Online][2])
在美国语境下，严格可公开获得的“地址级个体微观数据”较少，更多是行政/商业数据（选民名册、房产登记等）或隐私处理后的记录级数据（如 HMDA 到 tract、LODES 到 block 且为聚合表）。([密歇根州政府][20])
Voter files、HMDA、LODES 这三类数据分别更适合验证：居住地址分布（成人登记人口）、住房/借贷相关社会经济切片（到 tract）、以及工作地与通勤 OD（到 block 的聚合）。([密歇根州政府][20])
对你们项目而言，建筑级/地址级数据更现实的用法是“作为空间落位与外部验证锚点”，而不是取代 ACS 作为全属性 ground truth。([Taylor & Francis Online][2])

## 详细内容

### 子问题1：有哪些研究使用了建筑级别的人口数据？数据来源是什么？

下面给出更贴近“建筑/足迹/地址”粒度的代表性研究类型与例子：

1. 用建筑足迹把 census/ACS 聚合人口下推到细网格（近似建筑级）

* 研究示例：在 CONUS 范围利用 Microsoft building footprints，把 ACS 等聚合人口分解到 100m 网格（本质依赖建筑足迹作为居住承载体）。([Taylor & Francis Online][2])
* 研究示例：在洪水暴露评估中，利用国家级建筑足迹把 ACS block group 人口分解到更细尺度以捕捉微观异质性。([ScienceDirect][21])

2. 建筑足迹 + 调查/贝叶斯层次模型：估计更高分辨率人口（可能输出到建筑或近建筑尺度）

* 研究示例：用 household survey 与 building footprints 构建贝叶斯层次模型，生成高分辨率人口估计（强调“足迹+调查”融合）。([PubMed][22])

3. 建筑级人口估计（以建筑属性/体量等推断）

* 这类在城市尺度与遥感/3D 建筑数据语境中较常见（不一定美国），用于从建筑体量、用途推断居住人口。([isprs-archives.copernicus.org][23])

数据来源维度总结：

* 建筑足迹（building footprints）：公开足迹（如 Microsoft、OSM、地方 GIS）+ 住宅筛选/土地利用。([Taylor & Francis Online][2])
* 聚合人口约束：ACS 5-year（block group/tract）、Decennial Census。([Census.gov][24])
* 额外校准数据：household survey、行政记录或专题调查。([PubMed][22])

### 子问题2：美国学术界常用的地址级微观数据有哪些？（Voter files, HMDA, LODES 的具体使用案例）

这里按“空间精度—属性覆盖—获取难度”来讲清楚三类数据“能做什么/不能做什么”。

A) Voter files（选民登记名册，地址级个体记录，通常需购买/申请）

* 空间精度：通常到登记住址（street address），可地理编码到点/街段；但存在 PO Box、地址保密项目等例外。([密歇根州政府][20])
* 属性覆盖：姓名、地址、登记状态、（通常有）投票历史；对收入/就业/家庭结构覆盖很弱。([国家立法会议][25])
* 学术应用方式（典型）：作为“可地理编码的成人个体样本”，用于验证居住分布、迁移、投票行为与邻里效应等；对你们项目更现实的价值是**验证合成人口在空间上的成人分布/地址落位合理性**（不是全人口）。([国家立法会议][25])

B) HMDA（房贷申请记录，记录级公开但位置通常到 tract）

* 空间精度：公开数据通常到 census tract（而非具体地址）。([Consumer Financial Protection Bureau][26])
* 属性覆盖：贷款/申请人收入、部分人口统计属性、贷款条款等，适合做住房金融与公平信贷相关分析，但不代表全体人口。([Consumer Financial Protection Bureau][26])
* 学术应用方式（典型）：用作“住房市场/借贷行为的微观切片”，可为你们的收入分布、房主相关变量提供外部一致性检查（到 tract）。([Consumer Financial Protection Bureau][26])

C) LODES/LEHD（就业与通勤 OD：到 block 的细空间，但以聚合表发布）

* 空间精度：基础地理单元是 census block（居住与工作 geocode 都是 block code），OD/就业分布非常细，但不是“个体记录”，而是带有保密处理的聚合计数表。([lehd.ces.census.gov][7])
* 属性覆盖：可包含行业、收入分组、年龄段等维度的分组统计，适合验证 work location 分布与 OD 结构。([lehd.ces.census.gov][7])
* 学术应用方式（典型）：交通可达性、就业空间分布、通勤流分析与模型校准；对你们项目最直接用途是**验证合成人口的工作地分布/通勤 OD（如果你们要生成/分配工作地）**。([lehd.ces.census.gov][7])

### 期望输出1：数据源对比表（数据源、属性覆盖度、空间精度、获取难度）

| 数据源                                  | 记录粒度       | 空间精度           | 主要属性覆盖            | 获取难度/限制     | 对你们的主要价值                                                            |
| ------------------------------------ | ---------- | -------------- | ----------------- | ----------- | ------------------------------------------------------------------- |
| Michigan Voter Registration / QVF 导出 | 个体（登记选民）   | 地址级（可地理编码）     | 登记与投票相关，人口属性有限    | 需申请/费用/用途限制 | 验证成人居住空间分布、地址落位质量 ([密歇根州政府][20])                                    |
| HMDA                                 | 记录级（申请/贷款） | 多为 tract       | 收入、贷款、部分人口统计      | 免费公开（隐私处理）  | 验证住房金融切片与空间模式（到 tract） ([Consumer Financial Protection Bureau][26]) |
| LODES/LEHD                           | 聚合计数（多维分组） | block（OD）      | 行业/收入段/年龄段等分组就业统计 | 免费公开        | 验证工作地分布、OD 与通勤结构 ([lehd.ces.census.gov][7])                         |
| 建筑足迹 + ACS 下推的人口表面                   | 网格/近建筑     | 10–100m（取决于方法） | 多为总量或少数变量         | 足迹多免费；下推需建模 | 作为建筑级落位底座与空间异质性刻画 ([Taylor & Francis Online][2])                    |
| USPS DSF / 实地核查（研究/政府项目）             | 地址/投递点     | 地址级            | 占用/空置等            | 通常不完全公开     | 可用于检验 Census/ACS 占用与落位（案例：Detroit 核查） ([福特公共政策学校][27])              |

### 期望输出2：3-5 篇使用建筑级/地址级数据的研究案例

* Huang 等：利用 Microsoft building footprints 把 ACS 等聚合人口下推，生成 CONUS 100m 人口网格（建筑足迹作为关键承载体）。([Taylor & Francis Online][2])
* Huang 等：在洪水暴露评估中用建筑足迹把 block group 人口分解到更细尺度，以捕捉微观暴露差异。([ScienceDirect][21])
* Boo 等：结合 household survey 与 building footprints 的贝叶斯层次模型，做高分辨率人口估计。([PubMed][22])
* Cooney 等（Detroit 案例）：用 USPS 数据与现场核查审计 Detroit block group 的占用/人口计数偏差（“地址级行政/运营数据用于人口核验”的典型路径）。([福特公共政策学校][27])
  -（可补充方向）LODES 用于 block-level 通勤 OD 与工作地分布研究与模型校准（虽然是聚合表，但空间粒度非常细）。([lehd.ces.census.gov][7])

## 对我们项目的建议

1. 如果你们的目标是“建筑/地址级落位 + 可验证”，建议把建筑足迹/parcel/地址点视为 **allocation substrate**：先在 tract/TAZ 合成，再用足迹/住宅筛选+容量把人口下推，最后用外部数据验证（voter/LODES/HMDA 各验证一块）。([Taylor & Francis Online][2])

2. 不要把 voter file/HMDA 当作“全人口真值”，而应把它们当作“偏样本但高空间精度”的锚点：更适合检验空间落位/某些子人群切片，而不是替代 ACS 边际。([密歇根州政府][20])

## 参考文献

* Huang, X. 等：CONUS 100m population grid（building footprints + ACS 下推）。([Taylor & Francis Online][2])
* Boo, G. 等：building footprints + survey 的高分辨率人口估计。([PubMed][22])
* Huang, X. 等：建筑足迹用于洪水暴露人口细分解。([ScienceDirect][21])
* Cooney, P. 等：Detroit Census 2020 审计（USPS + 实地核查）。([福特公共政策学校][27])
* LEHD/LODES 官方说明与数据结构。([lehd.ces.census.gov][7])

---

# 调研报告：[任务 L4] 合成人口验证方法综述

## 核心发现（3-5 句话）

合成人口验证在学术界通常分三层：宏观（边际/总量）、中观（变量相关结构/联合分布）、微观（规则一致性与可用性），并强调 internal validation（用未参与拟合的约束或留出表做检验）与 external validation（对外部数据源的对比）。([STRC][1])
常见宏观拟合指标包括 SRMSE、相对误差（RE）、R²/相关系数、以及基于绝对误差的统计量；很多实践会对每个约束变量报告这些指标。([PMC][28])
在缺乏个体级 ground truth 时，主流做法是：留出部分边际/交叉表不参与合成作为“外部约束”，或引入外部行政数据（如 LODES）验证特定维度（工作地/通勤）。([PMC][28])
最小可复现的报告集建议至少覆盖：关键边际的拟合误差（按地理层级）、若干关键二元关系的拟合（相关/列联强度）、以及基本逻辑一致性规则；这比只报边际更能发现“分离生成与分配”带来的空间结构偏差。([PMC][28])

## 详细内容

### 子问题1：学术界验证合成人口质量的标准指标有哪些？（指标清单+计算方法）

把指标按层级整理（你们写论文时也更顺）：

A) 宏观层（macro / marginal fidelity）

* Relative Error (RE)：RE = (synthetic − target) / target。([PMC][28])
* SRMSE（Standardised Root Mean Square Error）：对多个区域或多个类别的误差做 RMSE 并标准化（常见做法是除以均值或目标量级）。([PMC][28])
* R² / Pearson correlation：把所有区域的 synthetic 与 target 向量做相关或回归拟合优度。([PMC][28])
* TAE/SAE（Total/Standardised Absolute Error）：对绝对误差求和或标准化（在 CO 文献里也常作为拟合目标/评估）。([pcwww.liv.ac.uk][29])

B) 中观层（meso / association structure）

* 列联表相似度：对关键二元/三元交叉表（如 age×sex、income×tenure）比较 cell-level 差异或用 Cramér’s V、互信息等衡量关联强度差异（很多表格合成与 TabSyn 类工作也会报告“相关结构一致性”）。([assets.amazon.science][17])
* 相关矩阵差异：数值变量相关系数矩阵（或分桶后的相关）对比，定位结构性偏差。([assets.amazon.science][17])

C) 微观层（micro / plausibility & rule validity）

* 逻辑约束：如家庭成员年龄关系、家庭结构合法性、就业状态与年龄一致性等（通常是规则检查与违例率）。
* 下游可用性：把合成人口喂给交通/健康/政策模型，看关键输出是否接近观测（严格来说是“应用级验证”，但在缺 GT 时很常见）。([spatial-microsim-book.robinlovelace.net][30])

### 子问题2：在缺乏个体级 ground truth 时，如何做验证？

主流路线通常是“留出 + 多源锚点”：

1. Internal validation（留出约束/交叉表）

* 合成时只用一部分边际约束，留出另一部分（尤其是交叉表或不同尺度约束）作为检验集；如果留出集也能较好拟合，说明模型不只是“记住了约束”，而是真正在重建合理联合分布。([PMC][28])

2. External anchors（外部数据锚点）

* 例如用 LODES 验证工作地与通勤 OD 结构；用 voter file 验证成人居住点分布；用建筑足迹/住房容量验证落位容量合理性。([lehd.ces.census.gov][7])

3. Uncertainty-aware reporting（不确定性意识）

* ACS 自身有 MOE/CV，不同 tract 的可靠性不同；在验证合成结果时，要避免把高 MOE 的 target 当作硬真值，可用 CV 分层报告或对低可靠 tract 做单独标记。([Census.gov][31])

### 子问题3：多层级验证（宏观边际 + 中观关联 + 微观规则）的常见做法

一个常见的“论文可写、工程可跑”的模板是：

* 宏观：对每个约束变量（按 tract/TAZ）报告 RE 分布（均值/中位数/分位数）+ SRMSE；并按关键地理层级（tract、block group、city）分别报告。([PMC][28])
* 中观：选 5–10 个最重要的二元关系（例如 age×sex、income×tenure、employment×age），比较关联强度与关键 cell 误差。([spatial-microsim-book.robinlovelace.net][30])
* 微观：列出 5 条“硬规则”（例如家庭结构合法性）与 5 条“软规则”（例如空间可达性/容量），报告违例率或需要重采样的比例。([Taylor & Francis Online][2])

## 对我们项目的建议

你们至少应报告的“最小指标集”（我建议写进论文 main text，其余放 appendix）：

1. 关键边际（每层级）：对人口总量、年龄段、性别、收入组、就业状态、家庭规模、住房占用/空置等，报告 tract 级 RE（含分位数）+ SRMSE。([PMC][28])
2. 关键关联（至少 3 个）：age×sex、age×employment、income×tenure（或 income×poverty）。报告 Cramér’s V（或替代指标）在真实与合成之间的差。([spatial-microsim-book.robinlovelace.net][30])
3. 空间落位专项：容量一致性（住宅建筑/住房单元）+（如做了工作地）通勤 OD 与工作地分布一致性（LODES）。([Taylor & Francis Online][2])
4. ACS 不确定性处理：对 CV>30% 的 tract/变量组合打标，验证结论按可靠性分层呈现。([Sound Data Stories][32])

## 参考文献

* Lovelace & Dumont. Spatial Microsimulation with R（书与配套仓库/综述入口）。([spatial-microsim-book.robinlovelace.net][30])
* Wu et al. A synthetic population dataset…（示例性地用 SRMSE/RE/R² 做 internal validation）。([PMC][28])
* Müller & Axhausen（合成人口综述：流程与常见任务分解）。([STRC][1])
* LEHD/LODES（外部验证锚点：工作地/OD）。([lehd.ces.census.gov][7])
* ACS MOE/CV 方法与 2020 相关质量说明。([Census.gov][31])

---

# 调研报告：[任务 D1] Wayne County / Detroit 地址级微观数据可获取性

## 核心发现（3-5 句话）

Detroit/Wayne County 想拿到“地址级个体微观数据”，最现实的公开路径通常不是 Census/ACS，而是行政或半行政数据：选民名册（成人、地址级）、房产登记/交易（地块/地址级）、以及用于工作地验证的 LODES（block 级聚合）。([密歇根州政府][20])
HMDA 提供的是公开的记录级借贷数据，但位置一般到 census tract，因此更像“社会经济切片验证集”，不适合作为地址级落位真值。([Consumer Financial Protection Bureau][26])
LODES 最新版本（8.4）已增加到 2023 年，并提供到 block 的 OD/就业分布，是验证 work location 与通勤结构的强外部锚点。([Census.gov][8])
如果你们只能选一个“最通用验证集”，优先级通常是：工作/通勤验证选 LODES；居住地址落位验证选 voter file（但只覆盖登记选民）；住房容量/落位底座选 parcel/交易/建筑足迹。([lehd.ces.census.gov][7])

## 详细内容

### 数据源1：Michigan Voter Registration（QVF 导出/选民名册）

(1) 是否公开可得？
密歇根的选民登记相关数据可通过官方数据请求流程获取（表单化申请、费用与用途限制需遵循说明）。([密歇根州政府][20])

(2) 包含哪些字段？
官方数据请求材料显示可导出选民登记相关字段（通常包含姓名、登记信息与地址等；具体字段应以申请说明/数据字典为准）。([密歇根州政府][20])

(3) 地址精度到什么级别？
通常是登记住址（street address），理论上可地理编码到点；但需考虑 PO Box、地址变更滞后、以及保密登记等例外。([国家立法会议][25])

(4) 覆盖人群比例？
覆盖的是“登记选民”，不是全体人口（不含未登记成年人、未成年人、部分非公民等），因此更适合做“成人居住分布/空间落位”验证而非全人口验证。([国家立法会议][25])

### 数据源2：HMDA（Home Mortgage Disclosure Act）

(1) Detroit/Wayne County 的记录数？
HMDA 平台支持按地理与年份过滤得到记录数；由于这里无法直接在报告中给出实时过滤结果，建议你们用 FFIEC HMDA 平台或 CFPB 的数据工具按 county/tract 过滤导出并统计。([Consumer Financial Protection Bureau][26])

(2) 可用字段（收入、地址精度）？
HMDA 记录级数据包含申请/贷款相关字段与申请人收入等信息，但公开位置通常到 census tract（非精确地址）。([Consumer Financial Protection Bureau][26])

(3) 时间覆盖？
HMDA 有多年历史并持续发布年度数据（字段口径在近年有变化，使用时应选定一致年份段）。([Consumer Financial Protection Bureau][33])

### 数据源3：Wayne County Property Transactions（房产交易/地契）

(1) 数据获取渠道？
Wayne County Register of Deeds 提供地契/登记记录的查询入口；此外，City of Detroit 也在开放数据平台发布了房产销售相关数据集（更偏“交易事件表”）。([韦恩县政府][34])

(2) 是否包含买家信息？
地契/登记记录通常包含交易双方（grantor/grantee）等信息；是否结构化到可直接分析的字段取决于你用的是“扫描文书”还是已结构化的开放数据表。([韦恩县政府][34])

(3) 历史数据可追溯多久？
登记系统通常有较长历史跨度，但数字化深度与可批量获取能力要看具体系统与开放数据起始年（Detroit 的开放数据往往从某个年份开始覆盖）。([底特律开放数据门户][35])

### 数据源4：LODES/LEHD

(1) Michigan 的最新年份？
Census Bureau 公告显示 LODES 8.4 已增加 2023 年数据（OnTheMap 同步更新），并指出整体序列覆盖 2002–2023（多数州）。([Census.gov][8])

(2) Block 级别 OD 矩阵的下载方式？
LODES 的 OD 文件结构与命名在技术文档中给出示例（如 `[state]_od_main_[jobtype]_[year].csv.gz`），可通过 LODES 下载入口获取；技术文档也明确 OD 文件包含居住与工作 block geocode 以及就业计数。([lehd.ces.census.gov][36])

(3) 是否可用于验证 work location 分布？
可以，LODES 的核心用途之一就是提供细粒度（block）工作地与居住地空间分布及其关联（OD）。([lehd.ces.census.gov][7])

## 期望输出：可行性评估表（可得性 / 字段 / 覆盖度 / 使用限制）+推荐

| 数据源                                         | 可得性       | 字段价值           | 覆盖度偏差           | 使用限制/成本    | 适合验证什么                                                              |
| ------------------------------------------- | --------- | -------------- | --------------- | ---------- | ------------------------------------------------------------------- |
| Michigan Voter Registration（QVF 导出）         | 可申请获取     | 地址级个体（成人）强     | 仅登记选民           | 申请/费用/用途限制 | 居住空间落位、成人空间分布 ([密歇根州政府][20])                                        |
| HMDA                                        | 免费公开      | 收入/贷款字段丰富      | 仅借贷人群；位置到 tract | 口径随年变化     | 收入/住房金融切片的空间一致性（tract） ([Consumer Financial Protection Bureau][26]) |
| Wayne County deeds / Detroit property sales | 部分开放/部分查询 | 房产/交易事件、地址/地块  | 仅产权/交易相关人群      | 结构化程度不一    | 住房容量/落位底座、房产空间模式 ([韦恩县政府][34])                                      |
| LODES/LEHD                                  | 免费公开      | block OD、就业分布强 | 覆盖“就业/通勤”维度     | 需理解聚合与保密处理 | work location/OD 验证（非常推荐） ([Census.gov][8])                         |

推荐（如果只能选一个“最适合作为验证集”）：

* 若你们当前最关键是“工作地/通勤结构/就业空间分布”的可信度：**优先 LODES**。([lehd.ces.census.gov][7])
* 若你们当前最关键是“居住地址落位到点/建筑”的可信度：**优先 Michigan voter file（但要明确它只验证成人登记人口）**，并辅以房产/建筑足迹做容量合理性检查。([密歇根州政府][20])

## 对我们项目的建议

1. 把验证拆成两条主线：Residence（voter file/房产/足迹）与 Work（LODES），避免试图用单一数据源同时验证居住+就业+收入。([密歇根州政府][20])
2. HMDA 更适合作为“收入/住房金融切片”的一致性检查，而不是地址级验证集。([Consumer Financial Protection Bureau][26])

## 参考文献

* Michigan QVF Data Request（官方数据请求材料）。([密歇根州政府][20])
* FFIEC HMDA 平台与 HMDA 历史数据页面。([Consumer Financial Protection Bureau][26])
* Wayne County Register of Deeds 与 Detroit property sales 开放数据。([韦恩县政府][34])
* LODES 8.4（2023 数据发布与技术文档/数据说明）。([Census.gov][8])

---

# 调研报告：[任务 D2] Detroit ACS Summary Tables 的完整性检查

## 核心发现（3-5 句话）

截至 2026-01-28，ACS 5-year 最新发布节奏显示 2020–2024 ACS 5-year 估计值在 2025-12 或 2026-01 前后发布/更新（不同渠道给出了具体日期），你们做 tract 级验证时需要锁定一个明确的 release 与年份段并保持一致。([it.nc.gov][37])
ACS tract 级可用的边际分布非常丰富（年龄×性别、收入分组、就业状态、通勤方式等），且确实存在不少交叉表（例如 sex×age×employment 这类），但交叉表越细，MOE/CV 往往越大，需要做可靠性分层与类别合并。([Census.gov][38])
MOE 的标准差换算公式（SE = MOE/1.645）与基于 CV 的可靠性阈值是你们“标记高 MOE tract”的可操作基础；2020 相关调查冲击导致部分年份的 tract-level CV/ MOE 系统性变大，这在 Census 的用户说明与地方数据机构的分析中都有强调。([Census.gov][31])
对于 Detroit，地方数据机构（Data Driven Detroit）明确提醒要特别关注小地理单元（tract/block group）的 MOE，并给出了用“MOE 占估计值比例”做标准化比较的示例（包含具体 tract 对比）。([数据驱动底特律][39])

## 详细内容

### 子问题1：ACS 5-Year 在 tract 级别提供哪些边际分布？（可用于验证的 ACS 变量清单）

你们做合成人口验证，通常需要三类边际：人口学、社会经济、住房与通勤。下面给一个“论文/系统常用最小子集”（表码以 ACS Detailed Tables / Subject Tables 为主；Subject Tables 以 S 开头）。([Census.gov][38])

人口学（Demographics）

* 年龄×性别：常用 B01001（Detailed）或 S0101（Subject）。([人口普查数据][40])
* 种族/族裔：B02001、B03002 等（按需求选）。
* 家庭/户结构：如 household type、household size 等相关表（按你们 household 合成口径选）。

社会经济（Socioeconomic）

* 家庭收入分组：B19001；中位家庭收入：B19013。
* 贫困状态：B17001 或相关比率表。
* 教育程度：B15003。
* 就业/劳动力状态：B23025；以及更细的 sex×age×employment（用作交叉验证，但注意 MOE）。

通勤与工作（Travel/Work）

* 通勤方式：B08301；通勤时间：B08303 等。
* 工作地验证建议与 LODES 结合（ACS 给的是通勤与就业边际/流，LODES 给的是更细的就业空间锚点）。([lehd.ces.census.gov][7])

住房（Housing）

* 自有/租住：B25003；房屋价值、房龄、空置等（按你们 housing module 选）。

说明：你们最终应以 data.census.gov 或 API 的变量字典做一次“Detroit tract 级可取变量清单”固化到项目文档中。([Census.gov][41])

### 子问题2：是否有交叉表（如 年龄×收入）？还是只有单变量边际？

结论是：**有交叉表，但“可用性=表复杂度 × tract 样本量”强相关**。

* ACS 的 Detailed Tables 本身包含大量交叉维度（例如年龄×性别×就业状态一类），Subject Tables 也经常提供按年龄/性别/族裔分组的汇总。([Census.gov][38])
* 但 tract 级别一旦 cell 很细，MOE/CV 往往显著增大，所以更常见做法是：

  1. 只选少量关键交叉表用于“结构性验证”；
  2. 对类别做合并（例如把收入分成更少档）；
  3. 对高 CV 区域打标并降低结论权重。([Census.gov][24])

### 子问题3：数据质量标志（MOE, CV）在 Detroit tracts 中的分布如何？如何标记高 MOE 区域？

在不逐 tract 拉取数据并计算的情况下，这里给你们一个“可复现的标记流程 + Detroit 本地证据”：

1. 计算关系与阈值（你们报告里应写清楚）

* ACS 使用 90% 置信水平：MOE = 1.645 × SE，因此 SE = MOE/1.645。([Census.gov][31])
* CV（常用相对可靠性指标）：CV = (SE / Estimate) × 100 = (MOE/1.645)/Estimate × 100。([CCRPC][42])
* 一个常见的解释阈值（示例）：CV ≤15% 好；15–30% 尚可；30–50% 谨慎；>50% 非常谨慎。([Sound Data Stories][32])

2. Detroit 的“高 MOE 风险”地方证据

* Census Bureau 提醒：由于 2020 部分采访减少，2016–2020 5-year 的 tract-level CV 在相对意义上可能上升约 15–20%，并导致不少关键估计的 tract-level CV 中位数超过 0.30。([Census.gov][24])
* Data Driven Detroit 的分析建议用“MOE/Estimate 的比例”把不同 tract 的不确定性标准化比较，并给出具体 tract 间对比示例（例如对某些指标，部分 tract 的相对 MOE 明显更高），并提醒要特别关注小地理单元。([数据驱动底特律][39])

3. 你们可以在论文/报告里输出的“标记结果形式”（不依赖画图也能写清楚）

* 对每个 tract、每个关键验证变量，给出 CV 分层标签（Good/Fair/Caution/High caution）。
* 输出两张表：
  A) “高风险 tract 列表”：在 ≥K 个关键变量上 CV>30% 的 tract；
  B) “高风险变量列表”：在哪些变量上 Detroit tract 的高 CV 最普遍。
* 结论呈现时：对高风险 tract 的 mismatch 不做过度解读，或把它们从主要拟合统计中单列/剔除后再对比一次（敏感性分析）。([Sound Data Stories][32])

## 对我们项目的建议

1. 先锁定你们验证所用的 ACS 年份段（比如 2019–2023 或 2020–2024），并在全文保持一致；同时在方法部分写明“ACS 的 MOE/CV 处理策略”，避免审稿人质疑把高噪声目标当真值。([it.nc.gov][37])
2. 对交叉表验证采取“少而精”：选 3–5 张与合成人口最相关的交叉表（例如 age×sex、age×employment、income×tenure），其余以单变量边际为主，并对高 CV tract 做分层报告。([Sound Data Stories][32])
3. 如果你们后续做建筑级落位，建议在 Detroit 使用 building footprints 下推的容量/落位底座，同时把“ACS 不确定性”与“落位误差”分开讨论：前者是 target 噪声，后者是模型偏差。([Taylor & Francis Online][2])

## 参考文献

* ACS 2024 数据发布节奏与 2020–2024 5-year 发布信息（不同渠道日期略有差异，需你们锁定版本）。([it.nc.gov][37])
* Census：MOE 与 SE 的换算（90% 置信水平）。([Census.gov][31])
* CV 计算与可靠性阈值示例（实践指南/方法文档）。([CCRPC][42])
* Census 用户说明：2016–2020 5-year 的 MOE/CV 上升问题。([Census.gov][24])
* Data Driven Detroit：Detroit 小地理单元 MOE 分析与标准化比较示例。([数据驱动底特律][39])

---

如果你愿意，我下一步可以把 L1/L4 中“验证指标 + 你们项目最小指标集”进一步落成一份你们论文可直接复用的 LaTeX 小节（含公式与符号定义），并把 D1/D2 的数据源与变量表整理成你们 2.2 数据调研部分的统一版式（按你们现有章节结构对齐）。

[1]: https://www.strc.ch/2010/Mueller.pdf "https://www.strc.ch/2010/Mueller.pdf"
[2]: https://www.tandfonline.com/doi/full/10.1080/20964471.2020.1776200 "https://www.tandfonline.com/doi/full/10.1080/20964471.2020.1776200"
[3]: https://prapare.org/wp-content/uploads/2021/10/Full-Toolkit.pdf?utm_source=chatgpt.com "IMPLEMENTATION AND ACTION TOOLKIT"
[4]: https://www.jasss.org/16/4/12.html "https://www.jasss.org/16/4/12.html"
[5]: https://onlinelibrary.wiley.com/doi/abs/10.1002/1099-1220%28200009/10%296%3A5%3C349%3A%3AAID-IJPG196%3E3.0.CO%3B2-5 "https://onlinelibrary.wiley.com/doi/abs/10.1002/1099-1220%28200009/10%296%3A5%3C349%3A%3AAID-IJPG196%3E3.0.CO%3B2-5"
[6]: https://www.researchgate.net/publication/226959696_Modeling_Residential_Location_in_UrbanSim?utm_source=chatgpt.com "Modeling Residential Location in UrbanSim | Request PDF"
[7]: https://lehd.ces.census.gov/data/lehd-code-samples/sections/lodes.html "https://lehd.ces.census.gov/data/lehd-code-samples/sections/lodes.html"
[8]: https://www.census.gov/programs-surveys/ces/news-and-updates/updates/12182025.html "https://www.census.gov/programs-surveys/ces/news-and-updates/updates/12182025.html"
[9]: https://www.research-collection.ethz.ch/bitstreams/7e65c211-dc30-40f5-9505-224f11920dc5/download "https://www.research-collection.ethz.ch/bitstreams/7e65c211-dc30-40f5-9505-224f11920dc5/download"
[10]: https://ideas.repec.org/a/kap/transp/v39y2012i3p685-704.html "https://ideas.repec.org/a/kap/transp/v39y2012i3p685-704.html"
[11]: https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf "https://proceedings.mlr.press/v202/kotelnikov23a/kotelnikov23a.pdf"
[12]: https://arxiv.org/html/2310.09656v3 "https://arxiv.org/html/2310.09656v3"
[13]: https://arxiv.org/html/2407.02549v1 "https://arxiv.org/html/2407.02549v1"
[14]: https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/SIGMOD24-Gen.pdf "https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/SIGMOD24-Gen.pdf"
[15]: https://arxiv.org/abs/2210.04018 "https://arxiv.org/abs/2210.04018"
[16]: https://arxiv.org/abs/2107.03006 "https://arxiv.org/abs/2107.03006"
[17]: https://assets.amazon.science/a8/d6/054330464506b2b3bb2566db838f/mixed-type-tabular-data-synthesis-with-score-based-diffusion-in-latent-space.pdf "https://assets.amazon.science/a8/d6/054330464506b2b3bb2566db838f/mixed-type-tabular-data-synthesis-with-score-based-diffusion-in-latent-space.pdf"
[18]: https://openreview.net/pdf/2e38216fb31cc24533b7352c0edc5e96813075fd.pdf "https://openreview.net/pdf/2e38216fb31cc24533b7352c0edc5e96813075fd.pdf"
[19]: https://www.emergentmind.com/topics/mtabgen "https://www.emergentmind.com/topics/mtabgen"
[20]: https://www.michigan.gov/-/media/Project/Websites/sos/02lehman/FOIA_FORM.pdf?rev=84209476fe23480b9ef3529b65850b5e "https://www.michigan.gov/-/media/Project/Websites/sos/02lehman/FOIA_FORM.pdf?rev=84209476fe23480b9ef3529b65850b5e"
[21]: https://www.sciencedirect.com/science/article/abs/pii/S2212420920302016 "https://www.sciencedirect.com/science/article/abs/pii/S2212420920302016"
[22]: https://pubmed.ncbi.nlm.nih.gov/35288578/ "https://pubmed.ncbi.nlm.nih.gov/35288578/"
[23]: https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/453/2024/ "https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/453/2024/"
[24]: https://www.census.gov/programs-surveys/acs/technical-documentation/user-notes/2022-04.html "https://www.census.gov/programs-surveys/acs/technical-documentation/user-notes/2022-04.html"
[25]: https://www.ncsl.org/elections-and-campaigns/access-to-and-use-of-voter-registration-lists "https://www.ncsl.org/elections-and-campaigns/access-to-and-use-of-voter-registration-lists"
[26]: https://www.consumerfinance.gov/about-us/newsroom/2023-hmda-data-on-mortgage-lending-now-available/ "https://www.consumerfinance.gov/about-us/newsroom/2023-hmda-data-on-mortgage-lending-now-available/"
[27]: https://sites.fordschool.umich.edu/poverty2021/files/2021/12/PovertySolutions-Census-Undercount-in-Detroit-PolicyBrief-December2021.pdf "https://sites.fordschool.umich.edu/poverty2021/files/2021/12/PovertySolutions-Census-Undercount-in-Detroit-PolicyBrief-December2021.pdf"
[28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8776798/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC8776798/"
[29]: https://pcwww.liv.ac.uk/~william/microdata/workingpapers/hw_wp_2001_2.pdf "https://pcwww.liv.ac.uk/~william/microdata/workingpapers/hw_wp_2001_2.pdf"
[30]: https://spatial-microsim-book.robinlovelace.net/ "https://spatial-microsim-book.robinlovelace.net/"
[31]: https://www.census.gov/content/dam/Census/programs-surveys/acs/guidance/training-presentations/2016_MOE_Slides_01.pdf "https://www.census.gov/content/dam/Census/programs-surveys/acs/guidance/training-presentations/2016_MOE_Slides_01.pdf"
[32]: https://psrc.github.io/psrccensus/articles/calculate-reliability-moe-transformed-acs.html "https://psrc.github.io/psrccensus/articles/calculate-reliability-moe-transformed-acs.html"
[33]: https://www.consumerfinance.gov/data-research/hmda/ "https://www.consumerfinance.gov/data-research/hmda/"
[34]: https://www.waynecountymi.gov/Government/Elected-Officials/Register-of-Deeds "https://www.waynecountymi.gov/Government/Elected-Officials/Register-of-Deeds"
[35]: https://data.detroitmi.gov/datasets/property-sales "https://data.detroitmi.gov/datasets/property-sales"
[36]: https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf "https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf"
[37]: https://it.nc.gov/us-census-bureau-american-community-survey-program-updates/open "https://it.nc.gov/us-census-bureau-american-community-survey-program-updates/open"
[38]: https://www.census.gov/acs/www/data/data-tables-and-tools/subject-tables/ "https://www.census.gov/acs/www/data/data-tables-and-tools/subject-tables/"
[39]: https://datadrivendetroit.org/blog/2022/04/20/2020-american-community-survey-data-margin-of-error-analysis/ "https://datadrivendetroit.org/blog/2022/04/20/2020-american-community-survey-data-margin-of-error-analysis/"
[40]: https://data.census.gov/profile/Detroit_city%2C_Michigan?g=160XX00US2622000 "https://data.census.gov/profile/Detroit_city%2C_Michigan?g=160XX00US2622000"
[41]: https://www.census.gov/programs-surveys/acs/data.html "https://www.census.gov/programs-surveys/acs/data.html"
[42]: https://www.ccrpcvt.org/wp-content/uploads/2018/10/ACS_Guide_Final_20181003.pdf "https://www.ccrpcvt.org/wp-content/uploads/2018/10/ACS_Guide_Final_20181003.pdf"

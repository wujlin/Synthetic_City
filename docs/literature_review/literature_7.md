下面是按你们给的优先级（P0→P2）整理的一轮“近几年（大致 2022–2026，必要时补充经典工作）”文献与数据源检索结果。为了贴合你们的瓶颈（PUMS 只有 PUMA、但要 tract/BG/building 级并且要多变量联合分布），我会在每条里尽量标注：它到底在“联合分布”上做到了什么、依赖了哪些额外信息、以及哪里其实是“不可识别/需要强假设”的。

你们的核心痛点，在近年文献里通常被表述为：PUMS 在 PUMA（最小约 100k 人）公开，且单年样本约 1%/五年约 5%，而 tract/BG 只有边际表格，导致“想要小区域联合分布/微观记录”必须依赖约束重加权、额外数据、或强结构假设（copula/可迁移依赖结构/生成模型迁移）。这一点在不少 2024–2026 的 PopSyn/生成模型论文里是直接点名的研究缺口。 ([ScienceDirect][1])

---

主题 5：Administrative data / 替代数据源（P0）

5.1（数据源）IRS SOI：Individual income tax statistics – ZIP Code data（更新到 2022 税年，页面更新时间 2025-12-04）
方法：不是论文，是可直接用作约束/校准的公开行政数据：按州×ZIP×AGI 档提供选定收入与税务条目。
数据源与空间粒度：IRS 个人所得税申报汇总，地理到 ZIP；覆盖税年 1998、2001、以及 2004–2022。
精度：官方汇总统计（非微观），你们可把它当作“ZIP 级收入分布/AGI 档边际”额外约束源。
与我们的关联：如果你们 tract/BG 的收入分布在 ACS 里太噪或缺失细档，SOI 可作为“另一套边际/先验”去做分布融合（比如把 tract→ZIP 映射后做约束或贝叶斯先验）；但 ZIP 与 tract/BG 的边界不一致会引入区域叠置误差（areal interpolation）和可识别性问题。
局限性：地理不是 tract/BG；而且是税表口径（AGI）并受披露控制影响。 ([irs.gov][2])

5.2（数据源）HUD CHAS（2018–2022 版本于 2025-12-23 发布；披露抑制增强）
方法：不是论文，是 HUD 基于 ACS 5-year 估计加工的 Consolidated Planning/CHAS 表，用于住房负担与需求分析。
数据源与空间粒度：基于 ACS 2018–2022 5-year；HUD 明确说明自 2018 表起披露保护增强，部分原先可得的估计会被 suppress。
精度：本质上继承 ACS 小区域估计误差 + 披露抑制导致的缺失。
与我们的关联：CHAS 的强项是“住房/收入相关的分组表格”可作为额外边际约束（尤其是住房负担/收入分档/住房问题相关维度），可能帮助你们在 tract/BG 层面补强“收入×住房”这一块；但它仍然是汇总表，不是微观联合分布。
局限性：披露抑制会让某些小区域关键格子缺失；而且它能补的是特定住房主题维度，不等于完整 SES 联合。 ([Hud User][3])

5.3（数据源）Census LEHD/LODES（OnTheMap；工作地/居住地就业统计，含 earnings 三档）
方法：不是论文，是公开就业行政统计（由 LEHD 体系产出），常被用于小尺度就业/通勤/职住分配与 PopSyn 的“工作端约束”。
数据源与空间粒度：LODES 以 census blocks 枚举（版本信息在 LEHD data 页面给出），并提供 earnings 分组（CE01/CE02/CE03：≤$1250/月、$1251–$3333/月、>$3333/月）。
精度：汇总统计（不是个人微数据），且对某些就业类型有覆盖限制（这点常见于对 LODES 的使用注意事项，很多二次加工文档会提醒）。
与我们的关联：你们要 employment/earnings 的 BG 级分布，LODES 是少数能到很细地理并给“收入档”的公开源之一；它可直接成为 BG 级就业与粗收入档的硬约束，显著缓解“PUMS 无 sub-PUMA location”带来的就业端不可识别。
局限性：earnings 只有 3 档；且它是“工作”维度（job counts），不是 household income/education 的完整联合。 ([lehd.ces.census.gov][4])

5.4（数据源）HUD “Low to Moderate Income Population by Block Group”（BG 级 LMI 标识）
方法：不是论文，是 HUD GIS 服务：标出哪些 BG 中 ≥51% 家庭收入 < 80% AMI。
数据源与空间粒度：到 census block group；输出是一个阈值型/分类信号。
精度：更像“标签/代理变量”，不是分布。
与我们的关联：可作为 building/BG 分配阶段的“低中收入区域 prior”，帮助把 PUMA 内差异引入空间条件；但信息量很低（只有是否 LMI）。
局限性：不能替代收入分布，更不能替代多变量联合分布。 ([hudgis-hud.opendata.arcgis.com][5])

5.5（论文，且非常贴合你们问题）Nejad & Deka 2021，Transportation Research Part D：A statistical approach to small area synthetic population generation…
核心方法：用“大样本（ACS）+ IRS 数据”等大尺度信息来生成更小地理（census tract）的合成人口属性（包含 income），服务于后续模型（如 car ownership/疏散规划）。
数据源：论文摘要直接写到用 ACS 与 IRS 数据去生成 tract 级合成人口。
精度：在可见摘要片段里未给出具体数值指标（完整指标需看全文/补充材料）。
与我们的关联：这是“IRS + ACS 做 tract 级合成”的明确先例，和你们主题 5、主题 1 都高度相关；可以重点看它如何把 IRS 汇总约束引入 tract 级。
局限性：从摘要可看它是为特定下游任务服务；对“多变量联合分布恢复精度”的系统评估不一定是主线。 ([ScienceDirect][6])

5.6（论文/思路：非传统数据做小区域估计）Acolin, Decter‑Frain & Hall 2022，Demographic Research：Small-area estimates from consumer trace data
核心方法：用 consumer trace data（Data Axle）来做 tract 级 household count 等小区域估计，并用机器学习（Lasso）做校准以减轻偏差。
数据源：consumer data + 人口统计基准（文中明确讨论与 survey-based estimates 的偏差与校准）。
精度：在你们关心的“是否能替代普查/调查小区域统计”上，它给出评估框架与对比；但在我当前抓到的页面片段里，MAPE 的具体数值还没露出（需要你们进一步翻到结果段落/表格）。
与我们的关联：这类“consumer/trace 数据”可以作为你们 PUMA 内空间异质性的外生信号源，尤其适合 building-level 分配阶段（dasymetric/parcel allocation）；但它通常更强在“总量/单变量”，对“多变量联合分布”帮助取决于字段丰富度与校准策略。
局限性：consumer 数据覆盖偏差与选择偏差是主问题；需要校准、且可迁移性不保证。 ([Demographic Research][7])

---

主题 1：Sub‑PUMA 社会经济属性空间降尺度（P0）

1.1（很关键，且直接给出一个可量化验证例子）Graetz, Ummel & Aldana Cohen 2022，Sociological Methodology：Small‑Area Analyses Using Public ACS Data: A Tree‑Based Spatial Microsimulation Technique
核心方法：提出 tree‑based spatial microsimulation（用决策树/树模型来更“信息化地选约束并做小区域估计/微观推断”），用公开 ACS microdata 推 tract（small‑area）层面的目标变量估计。
数据源与空间粒度：公开 ACS（PUMS microdata + tract 层面的可用约束/表），目标是 tract 级小区域估计。
精度：文中在纽约市五县做 out‑of‑sample 验证（以 tract‑level mean years of educational attainment 为例），报告 explained R²=0.832；并指出传统 stepwise selection 在多个拟合统计上更差。
与我们的关联：你们要的是“PUMA microdata → tract/BG”，这篇提供了一个公开数据可落地、且把“约束选择/避免 shrinkage 到全局均值”作为核心改进点的路径；对你们来说，价值在于：把“PUMA 内空间差异”更多地从 tract 可观测边际/上下文特征中提取出来，从而提升 sub‑PUMA 的可辨识信息。
局限性：它展示的更多是“目标变量的小区域估计”能力，而不是完整微观联合分布的逐格恢复；联合分布精度仍取决于可用约束集合与模型假设。 ([Nick Graetz][8])

1.2（“全国 BG 级合成人口数据集”的工程化路线）Rineer et al. 2025，Scientific Data（PMC）：A National Synthetic Populations Dataset for the United States
核心方法：在 block group 层面用 IPF 估计 household‑level joint counts，然后从 PUMA microdata 中按这些 joint counts 抽取 household/person 记录生成 100% 合成数据；当本地 PUMA 候选池太小，会扩展到州内相似 PUMA、放宽分箱、或移除部分匹配属性以满足最小候选池阈值。
数据源与空间粒度：输入是 PUMS（PUMA）+ ACS block group 级边际/类别计数（文中例子里提到的匹配变量包括户主年龄、户收、户规模、族裔、种族等）；输出到 census block group。
精度：文中强调在 BG 与州级做 household count 的校验；并给出一个“需要候选 microdata 数/所需 household 数达到约 0.15”以降低稀有记录可重识别风险的工程阈值。
与我们的关联：这几乎就是你们主题 1 的“主航道”之一：用 tract/BG 边际约束 + PUMS 依赖结构来做小尺度合成，并且它明确讨论了当 microdata 支撑不足时的候选池扩展/约束放松策略（这正是 PUMA→BG 的现实瓶颈）。
局限性：这类方法对“你没约束到的联合结构”无法保证；当必须放宽属性匹配或跨 PUMA 借样本时，小区域的高阶联合分布可能被平滑/偏置。 ([PMC][9])

1.3（以“building resolution”作为明确目标）Khachman, Morency & Ciari 2024，Transportation：Integrated multiresolution framework for spatialized population synthesis
核心方法：提出 integrated multiresolution framework（IMF），把“生成 + spatialization”更紧耦合，目标是直接在 building resolution 生成合成人口，并用多分辨率控制的优化方法扩展传统框架。
数据源与空间粒度：案例在 Montreal；强调“minimal data requirements”下实现 building resolution 合成，并对比 conventional framework。
精度：摘要层面结论是：IMF 在牺牲少量 sociodemographic accuracy 的情况下，显著提升 spatial precision、overall quality 与 building‑resolution fit。
与我们的关联：你们最终要 building 级，这篇的价值是：它不是先 tract 再撒到建筑，而是把多分辨率控制纳入优化框架，属于“从方法论上正面打 building resolution”路线。
局限性：订阅可见全文（摘要可读），需要进一步核对：它在多变量联合分布上到底约束了什么、building 级 ground truth 用什么定义与评估。 ([ideas.repec.org][10])

1.4（parcel 级/暴露评估驱动的 building/parcel 分辨率需求）Black‑Ingersoll et al. 2026（PMC）：A Novel Method for Generating Spatially Resolved Synthetic …
核心方法：面向“高度空间异质的暴露/风险评估”，提出更高空间分辨率的合成人口生成方法（parcel/更细尺度），并指出既有应用多停留在 tract。
数据源与空间粒度：从可见片段可确认其问题设定：现有方法空间分辨率不足，过去常用 tract 级。
精度：需进一步看全文的验证设计（你们可重点看它怎么把人口放到 parcel/建筑，以及是否引入 property/land use 等外部数据）。
与我们的关联：如果你们的 building 级定位需要“可发表的方法学叙事”，这篇提供了一个很清晰的动机框架（tract 不够）。
局限性：需要核对它在 SES 联合分布上的处理是否只是“先合成再分配”，还是在分配阶段引入了 SES‑aware 的约束。 ([PMC][11])

1.5（copula 路线：把“依赖结构”当作可迁移对象）Bastin et al. 2023（论文 PDF）：Copula‑based synthetic population generation
核心方法：用 copula（依赖结构）+ 给定边际分布来生成合成人口；并在数值实验里做 state/county/tract 多尺度生成。
数据源与空间粒度：文中明确写到：用 ACS 变量样本 + “实际边际分布（marginals）”来自 Decennial Census Housing Data 与 IRS，用于多尺度（含 census tract）实验。
精度：需要进一步翻到数值实验结果表（我当前抓到的是实验设置段与方法段，未抓到误差表的具体数值）。
与我们的关联：你们关心“在只有 PUMA microdata 的依赖结构时，能否把 tract 边际喂进去恢复 tract 联合”，copula 路线就是典型答案之一：假设依赖结构（或其参数化形式）能从大尺度迁移到小尺度，然后用小尺度边际“重塑”联合。
局限性：本质上是强假设（依赖结构可迁移/可共享）；在信息论意义上，边际并不能唯一确定联合（需要额外先验），copula 只是选择了一类先验/结构。 ([ResearchGate][12])

1.6（开源/开放数据的空间微模拟框架）Tuccillo et al. 2022/2023（OSTI 报告）：UrbanPop: A spatial microsimulation framework…
核心方法：提出 UrbanPop 空间微模拟（SMSM）框架，强调完全基于开放数据、面向高空间/时间/人口学分辨率分析。
数据源与空间粒度：open data；用于高分辨率人口动态分析。
精度：需进一步看其误差度量与对“联合分布”的评估（报告结尾明确提到未来会沿用 Graetz 2022 的思路研究用树模型辅助约束选择）。
与我们的关联：如果你们要做可复现 pipeline（尤其是开源/开放数据叙事），UrbanPop 属于同一生态位；并且它明确把“约束选择”当作关键后续方向。
局限性：通常微模拟框架强依赖你给的约束表与外部空间数据质量；对“未被约束的联合结构”同样缺少保证。 ([osti.gov][13])

---

主题 2：移动/访问数据推断个体 SES（P1）

2.1（偏“负面/谨慎”，但非常值得你们读）Uğurel et al. 2025（arXiv）：On Predicting Sociodemographics from Mobility Signals
核心方法：明确指出“从 mobility 推 sociodemographics 很难”，原因包括关系弱且不一致、跨场景泛化有限；提出基于 directed mobility graphs 的高阶 mobility 描述符，并用 multitask learning 同时预测多个属性（age/gender/income/household structure），同时引入不确定性诊断指标。
数据源与空间粒度：面向交通规划常见设定：HTS（有真值属性）+ 被动移动数据（只有时空轨迹）之间的推断问题（论文摘要与引言直接讨论这一生态）。
精度：我当前抓到的段落主要是方法与实验设计引入（提到 AUROC、NLL、ECE 等作为评估，并声称多任务与新特征提升表现），但没抓到具体数值表；你们可直接在论文的 Experiments/Results 表里提取各属性的 AUROC/Accuracy/R²（如果 income 是回归则看 R²）。
与我们的关联：如果你们在考虑“用 Dewey visit/POI 访问”来补足 sub‑PUMA 空间异质性，这篇的关键价值在于：它把难点说得非常直白，并且把“泛化/代表性偏差/置信度校准”当作第一等公民指标（这往往是很多 ‘income prediction from GPS’ 工作的软肋）。
局限性：它本身就承认 mobility→SES 关系弱、且受采样偏差影响（智能手机数据会系统性欠采样某些群体），因此即使局部有效，也要非常谨慎外推。 ([arXiv][14])

2.2（把移动数据“融合进”人口合成，而不是直接预测 SES）Vo, Kim & Bansal 2023（SSRN；后续发表于 TRB/期刊版本）：A novel data fusion method to leverage passively‑collected mobility data in generating spatially‑heterogeneous synthetic population
核心方法：用 cluster‑based data fusion 把 HTS（有 sociodemographics 但样本稀）与 PCM（cellular traces，空间覆盖强但属性少）融合，得到同时有 sociodemographics + home–work locations 且“高空间异质性”的合成人口；强调通过聚类对齐空间分辨率、并把融合重写为 cluster‑specific 低维优化子问题。
数据源与空间粒度：HTS + LTE/5G cellular signaling（首尔）用于验证。
精度：摘要里给了一个很强的“异质性提升”信号：生成的人口中约 97% 在 HTS 中未被观测到，同时宣称能在对照 census 的验证中维持分布一致性。
与我们的关联：你们主题 2 想知道“移动数据路线是否可行”，这篇给出一种更稳的 framing：不必声称‘从 GPS 精准预测收入’，而是用移动数据把空间异质性注入合成过程（尤其是 home–work/活动空间结构），再用 census/ACS 边际把 sociodemographics 拉回去。
局限性：它解决的是“空间属性/活动位置异质性”更直接；对 income/education/occupation 的个体级推断仍取决于 HTS 的覆盖与融合假设。 ([papers.ssrn.com][15])

2.3（经典基线，供你们对后续工作做谱系追踪）Blumenstock, Cadamuro & On 2015，Science：Using mobile phone data to map poverty
核心方法：用手机元数据（含通信/使用行为特征，通常也含一定 mobility/社交网络信号）预测贫困/财富，并做空间映射。
数据源：移动运营商数据 + 调查/基准真值（该类工作通常用 survey asset/wealth index 做 ground truth）。
精度：建议你们直接把该文的主表指标（R²/误差）摘出来作为“行业记忆点基线”，再与 2021–2025 一批工作对比“泛化/偏差/可复制性”。
与我们的关联：你们在做“visit 推断属性”时，它是必须引用的祖师爷之一；但它能否迁移到美国、能否从收入推到教育/职业、以及能否稳定落到个体级，是后续争论焦点。
局限性：代表性偏差、跨区域/跨时间迁移、以及隐私与可验证性问题，是后续文献最常攻击点。

2.4（mobility 与收入分组/隔离的关系，偏群体/区域层面）Moro et al. 2021，Nature Communications：Mobility patterns are key to …
核心方法：研究 mobility patterns 与 income segregation/不平等空间格局之间的关系（更偏区域层面规律，而非个体收入回归）。
数据源：移动/轨迹数据 + 收入分组/区域统计（通常通过 home location 匹配 census income 分组）。
精度：通常不会给“个体 income prediction R²”，而是给解释性/机制性指标。
与我们的关联：如果你们要论证“mobility 信号能提供 SES 的可辨识信息”，这类工作是强证据；但它更支持“区域/分组可推断”，不等于“个体可高精度预测”。
局限性：生态推断风险（ecological fallacy）：群体层面关系不自动推出个体层面可预测。 ([estebanmoro.org][16])

---

主题 3：多源数据融合构造更细训练数据（P1）

3.1（数据融合做得很“工程+统计”）Ummel et al. 2024，Scientific Data：fusionACS
核心方法：把多份调查（能源、交通、住房、消费等）通过统计/机器学习融合到 ACS 上，生成“以 ACS 为骨架、但补齐其它调查变量”的 synthetic microdataset，并强调不确定性（multiple implicates）框架。
数据源与空间粒度：ACS（骨架）+ 多调查源（donors）。空间粒度通常受 ACS microdata 发布限制（公开 microdata 仍在 PUMA），但它能显著补齐“变量维度”。
精度：通常通过分布一致性、变量关系保持、以及多重插补一致性诊断来评估（你们可重点看它如何验证融合质量）。
与我们的关联：你们主题 3 要的是“多个不完整源拼成更强训练集”，fusionACS 是非常直接的可借鉴模板：哪怕空间分辨率未提升，它也能告诉你如何在严格的统计框架下融合、并量化不确定性。
局限性：提升的是变量维度而不一定提升空间分辨率；空间细化仍要靠额外地理约束或其它可定位数据源。 ([Nature][17])

3.2（把“passive mobility”当作可融合源）Vo, Kim & Bansal 2023/2025（同 2.2）
核心方法与价值点：明确把“空间异质性不足”归因于 HTS 采样稀疏，并用 PCM 数据补空间覆盖，再在融合分布上保持分布一致性。
与你们主题 3 的关联：这是“mobility + survey”的明确融合路线，而且是为 PopSyn 服务（不是纯预测 income）。 ([papers.ssrn.com][15])

3.3（理论基础：data combination 的可识别性与界）Ridder & Moffitt 2005（讲义/综述 PDF）：The Econometrics of Data Combination
核心方法：系统讨论 statistical matching / data combination 的识别条件与界（很多融合方法隐含 conditional independence 或结构性假设）。
与我们的关联：你们如果要把 “PUMS + ACS tables + admin + mobility” 串起来，最怕的是“看起来能做，其实不可识别”；这份材料非常适合作为你们内部方法论红线：哪些融合在理论上就只能做 partial identification/区间估计。
局限性：较经典（不是 2022+），但属于绕不开的理论地基。 ([econ2.jhu.edu][18])

3.4（更近年的偏理论推进）Jiang & Janson 2025（PDF）：Semiparametric Inference for Partially Identifiable Data Fusion Estimands
核心方法：把 data fusion 放进“部分可识别（partial identification）+ 半参数推断”的框架，讨论当融合估计量不可点识别时怎么做推断与界。
与我们的关联：如果你们打算在论文里认真讨论“我们到底从数据中识别了什么、没识别什么”，这类工作能给非常现代的统计语言与工具。 ([lucasjanson.fas.harvard.edu][19])

---

主题 4：深度生成模型用于空间人口合成（P2，但你们会很关心“是否有人解决 sub‑PUMA 条件化”）

4.1（2026，直接点名 PUMA→tract 的生成模型迁移）Qian et al. 2026，Applied Soft Computing：A deep generative framework for joint households and individuals population synthesis
核心方法：提出 parameter‑efficient transfer learning，把在 PUMA‑level microdata 训练的生成模型“适配”到 census tract level，同时要求符合 tract 层面的目标边际分布（ACS census tables），并强调保持 microdata 的变量依赖结构与记录真实性。
数据源与空间粒度：PUMS microdata（PUMA）+ tract/BG 的 ACS tables（边际）；目标输出 tract 级 household+individual 合成。
精度：摘要片段强调“保持 realism + 符合 tract marginals”，但具体误差指标（TVD/RMSE/联合一致性）需要读全文实验部分（ScienceDirect 订阅）。
与我们的关联：这是你们主题 4 问题的正面命中：深度生成模型如何做 sub‑PUMA 条件化（至少到 tract），并用迁移学习把“PUMA 依赖结构”搬过去。
局限性：仍然绕不开“边际不能唯一确定联合”的理论事实；模型输出的高阶联合质量，取决于迁移假设与训练/正则策略，而不仅仅是 marginal matching。 ([ScienceDirect][1])

4.2（扩散模型路线，偏“把 PopSyn 当 tabular generation + constraints”）Tang et al. 2025（arXiv PDF）：Diffusion‑based population synthesis
核心方法：用 diffusion model 做人口合成（tabular generation），通常会结合条件生成/约束满足机制。
数据源与空间粒度：多为 microdata + 聚合约束的设定（需要你们看它是否支持分层地理条件）。
精度：需核对其对比基线（IPF/CO/CTGAN/TabDDPM 等）与约束满足误差。
与我们的关联：如果你们要做“生成模型 SOTA 盘点”，扩散模型在 2024–2026 的 tabular 生成里属于显著趋势；但是否真正解决 sub‑PUMA，关键看它怎么引入 tract/BG 级条件特征与验证。 ([arXiv][20])

4.3（VAE 在 PopSyn 的系统评估趋势）Aemmer & Mackenzie 2022：A variational autoencoder for population synthesis of sparse multivariate data
核心方法：用 VAE 处理稀疏/高维多变量的 PopSyn。
数据源与空间粒度：典型是 survey/microdata 训练，输出合成个体/家庭；空间条件化是否支持需要看实现。
精度：建议关注它在稀疏高维下如何避免不合理组合（structural zeros）与如何评估分布距离。
与我们的关联：你们要的是“多变量联合分布 + 小区域”，VAE 的优势在高维生成，但小区域约束仍要外接（marginal constraints / hierarchical conditioning）。 ([ScienceDirect][21])

4.4（GAN 辅助 PopSyn 的早期实证）Kotnana et al. 2022（IEEE 论文 PDF）：Using GANs to Assist Synthetic Population Creation for Simulations
核心方法：用 GAN 辅助合成人口生成，并对合成与真实 census 分布做对比图示（例如 age 分布对比）。
数据源与空间粒度：面向仿真用 synthetic population；空间与约束细节需看全文。
精度：该 PDF 片段显示其对不同算法/不同地理区做分布对比，并承诺扩展评估。
与我们的关联：可作为“生成模型进 PopSyn”的引用点，但你们真正关心的 sub‑PUMA 条件化/联合分布严格匹配，可能需要更近年的扩散/迁移学习路线。
局限性：GAN 在 tabular 上常见问题（mode collapse、难以严格满足边际约束）仍在；需要约束机制或后处理。 ([NSF PAR][22])

4.5（“building‑resolution fit”导向的 PopSyn 框架）Khachman et al. 2024（同 1.3）
与主题 4 的关系：它不一定用深度模型，但它把“building resolution”作为质量度量与优化目标之一，对你们的 building‑level 合成评估指标设计很有参考价值。 ([ideas.repec.org][10])

---

主题 6：理论极限 / 不可能性（P2，但对你们“避免死胡同”非常关键）

6.1（近年 ecological inference 的“界/部分识别”推进）Elzayn, Goldin, Ho 等 2025（arXiv/NBER）：Monotone Ecological Inference
核心方法：把 ecological inference 放在 partial identification 框架下，利用单调性等结构信息来收紧界（bounds），而不是声称点识别。
与我们的关联：你们问“只有 PUMA 微数据时，sub‑PUMA 联合分布恢复有没有理论下界？”——这类工作给出的核心结论倾向是：如果没有额外结构假设，很多个体层参数只能在区间内识别；能做的是给 bounds，并讨论什么假设能收紧。
局限性：它主要处理的是 ecological inference 典型设定（很多是二元/分组/单调性），你们的多变量高维联合会更难，但思想是一样的：强调“可识别集”而非唯一解。 ([arXiv][23])

6.2（从“边际推联合”本质上是病态问题，需要先验）Frogner et al. 2019（PMLR PDF）：Fast and Flexible Inference of Joint Distributions from their Marginals
核心方法：讨论从 marginals 推 joint distribution 的算法，但关键点写得很直白：一般情况下这问题是 ill‑posed，需要 prior information 才能选出唯一解。
与我们的关联：这几乎就是你们问题的数学内核：PUMA microdata 给了一个层级的 joint 信息，但 tract/BG 只有边际；想恢复 tract/BG joint，必须引入“先验/结构”（独立性、copula 形式、可迁移依赖、空间平滑、外部 covariates 等）。
局限性：这是 ML/推断视角，和 PopSyn 的具体约束实现不同，但对你们写“理论极限”章节非常好用。 ([Proceedings of Machine Learning Research][24])

6.3（copula 作为“选择一个 joint”的结构先验）Bastin et al. 2023（同 1.5）
与主题 6 的关系：它的路线本质是在“边际不足以定联合”的情况下，用 copula 结构去约束解空间；你们可以把它放进“强假设换可解性”的框架讨论。 ([ResearchGate][12])

---

给你们的“结论型摘要”：学界哪些路更像“有效解法”，哪些更像“有理论天花板/必须强假设”

第一，PUMA→tract/BG 的“空间降尺度”在工程上是可做的，而且已经有人把全国 BG 级 synthetic population 做成公开数据集（Rineer 2025 这类），通常套路就是：用 tract/BG 级边际约束（ACS tables/其它汇总）+ PUMS 作为依赖结构来源，IPF/抽样/候选池扩展来生成小区域微观记录。它能保证的是“你约束的边际与某些低阶结构”，而不是全联合的点真值。 ([PMC][9])

第二，若你们把目标升级到 building-level，文献里越来越多是“两段式”或“多分辨率一体化”：先在 tract/BG 合成，再用 parcels/buildings/land use 做 dasymetric/容量约束分配；或者像 Khachman 2024 那样把 building resolution 直接纳入多分辨率优化框架来提升 building-fit。 ([ideas.repec.org][10])

第三，移动/访问数据推 SES：近年更成熟的说法是“个体级很难稳定泛化，区域/分组层面更稳”，并且代表性偏差与跨域泛化是硬伤；更可行的路线往往是“用移动数据补空间异质性/活动位置结构”，再把 SES 通过 census/ACS 边际拉回去（Vo et al. 这一类融合式 PopSyn）。 ([arXiv][14])

第四，理论上，“仅靠边际恢复联合”一般不可点识别：你们如果要避免在死胡同里卷，建议把工作拆成两层目标：a) 明确哪些联合结构是可识别/可被强约束的（靠 LODES earnings 三档、SOI ZIP 收入档、CHAS 住房负担表等补充约束），b) 对剩余不可识别部分，用结构先验（copula/层级贝叶斯/生成模型迁移）并用 bounds/敏感性分析表达不确定性，而不是把一个解当真值。 ([Proceedings of Machine Learning Research][24])

---

如果你们希望我把上述每条进一步“补齐你们模板里最缺的两项：精度表格（R²/TVD/RMSE）+ ground truth/验证数据来源”，我建议下一步优先把这 6 篇/资源作为“精读+抽表”对象（按你们 P0/P1 排序）：

1. Rineer 2025（BG 级全国合成数据集，工程细节与约束放松策略） ([PMC][9])
2. Graetz 2022（公开数据下 small-area 的可量化验证范式，且有 R² 报告） ([Nick Graetz][8])
3. Qian 2026（生成模型 + transfer learning 做 PUMA→tract 条件化的最新路线之一） ([ScienceDirect][1])
4. Bastin 2023（copula + IRS/Decennial marginals，多尺度实验，适合写“强假设换可解性”） ([ResearchGate][12])
5. Vo 2023/2025（mobility + HTS 融合注入空间异质性，给出 97% 未观测异质性指标） ([papers.ssrn.com][15])
6. LODES + IRS SOI + CHAS（把你们最缺的 sub‑PUMA 约束源补上：就业与 earnings 档、ZIP 收入档、住房负担相关表） ([lehd.ces.census.gov][4])

如果你们愿意，我也可以基于你们实际要合成的变量清单（income/education/employment 的具体分箱、是否需要 race/age/household type 等）把“哪些约束来自 ACS tables、哪些可由 LODES/SOI/CHAS/parcel 代理补强、哪些只能靠结构先验”画成一张可执行的约束-数据矩阵，直接服务你们后续模型设计与实验对照。

[1]: https://www.sciencedirect.com/science/article/abs/pii/S1568494625016886 "https://www.sciencedirect.com/science/article/abs/pii/S1568494625016886"
[2]: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi "https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi"
[3]: https://www.huduser.gov/portal/datasets/cp/CHAS/data_doc_chas.html "https://www.huduser.gov/portal/datasets/cp/CHAS/data_doc_chas.html"
[4]: https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf "https://lehd.ces.census.gov/doc/help/onthemap/LODESTechDoc.pdf"
[5]: https://hudgis-hud.opendata.arcgis.com/datasets/HUD%3A%3Alow-to-moderate-income-population-by-block-group/about "https://hudgis-hud.opendata.arcgis.com/datasets/HUD%3A%3Alow-to-moderate-income-population-by-block-group/about"
[6]: https://www.sciencedirect.com/science/article/abs/pii/S0966692320309790 "https://www.sciencedirect.com/science/article/abs/pii/S0966692320309790"
[7]: https://www.demographic-research.org/volumes/vol47/27/47-27.pdf "Small-area estimates from consumer trace data"
[8]: https://ncgraetz.com/media/Graetz_2022_Soc_Method.pdf "https://ncgraetz.com/media/Graetz_2022_Soc_Method.pdf"
[9]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11762717/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC11762717/"
[10]: https://ideas.repec.org/a/kap/transp/v51y2024i3d10.1007_s11116-022-10358-w.html "https://ideas.repec.org/a/kap/transp/v51y2024i3d10.1007_s11116-022-10358-w.html"
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12765813/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC12765813/"
[12]: https://www.researchgate.net/profile/Fabian-Bastin/publication/368665035_Copula-based_synthetic_population_generation/links/640745ed0d98a97717e7f337/Copula-based-synthetic-population-generation.pdf "Copula-based synthetic population generation"
[13]: https://www.osti.gov/servlets/purl/1906627 "https://www.osti.gov/servlets/purl/1906627"
[14]: https://arxiv.org/pdf/2511.03924 "https://arxiv.org/pdf/2511.03924"
[15]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4612180 "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4612180"
[16]: https://estebanmoro.org/pdf/News_or_social_media__Socio_economic_divide_of_mobile_service_consumption.pdf "https://estebanmoro.org/pdf/News_or_social_media__Socio_economic_divide_of_mobile_service_consumption.pdf"
[17]: https://www.nature.com/articles/s41597-023-02788-7 "https://www.nature.com/articles/s41597-023-02788-7"
[18]: https://www.econ2.jhu.edu/people/moffitt/comsampdefmin.pdf "https://www.econ2.jhu.edu/people/moffitt/comsampdefmin.pdf"
[19]: https://lucasjanson.fas.harvard.edu/papers/Semiparametric_Inference_For_Partially_Identifiable_Data_Fusion_Estimands-Jiang_Janson-2025.pdf "https://lucasjanson.fas.harvard.edu/papers/Semiparametric_Inference_For_Partially_Identifiable_Data_Fusion_Estimands-Jiang_Janson-2025.pdf"
[20]: https://www.arxiv.org/abs/2508.09164 "https://www.arxiv.org/abs/2508.09164"
[21]: https://www.sciencedirect.com/science/article/abs/pii/S0198971522000965 "https://www.sciencedirect.com/science/article/abs/pii/S0198971522000965"
[22]: https://par.nsf.gov/servlets/purl/10409331 "https://par.nsf.gov/servlets/purl/10409331"
[23]: https://arxiv.org/pdf/2504.14752 "https://arxiv.org/pdf/2504.14752"
[24]: https://proceedings.mlr.press/v97/frogner19a/frogner19a.pdf "https://proceedings.mlr.press/v97/frogner19a/frogner19a.pdf"

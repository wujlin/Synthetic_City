## （三）研究基础

### 1. 研究基础与可行性分析

#### 1.1 研究工作积累

**申请人在合成人口算法研究领域已形成系统性积累，为本项目奠定了坚实基础。** 针对个人微观数据有限对基于智能体模型（ABM）等计算模拟方法应用受限的问题，申请人致力于高精度地理信息合成人口算法研究。申请人开发了两代合成人口算法并开源了总量超10GB的合成人口数据集：基于2010年人口普查数据构建了纽约都市区2200万人口数据库；利用2020年数据建立了覆盖全美3亿人口的标准化数据库。两个算法分别发表在*Computational Urban Science*与*Nature*旗下期刊*Scientific Data*，成果曾荣获2023年计算社会科学学会美洲会议（CSSSA）最佳论文，并入选2022年*Computational Urban Science*期刊下载量前三文章。

上述工作验证了深度生成模型在合成人口领域的有效性，并建立了从数据预处理、模型训练到验证评估的完整技术流程。本项目将在此基础上，进一步探索多源数据融合与建筑物尺度人口生成的方法创新。

---

#### 1.2 理论基础

**本项目的理论基础建立在生态推断（Ecological Inference）与深度生成模型两大支柱之上。** 生态推断问题（King, 1997）揭示了从聚合数据推断个体行为的根本不可识别性：边际分布相同时，微观联合分布可以有无穷多种形态。Fishman与McAuliffe（2021）将这一问题形式化为可学习的推断框架，表明要在聚合观测下推断微观量，必须引入额外信息源收缩解空间。

在技术层面，扩散模型在表格数据合成领域已趋成熟。TabDDPM（Kotelnikov et al., 2023）系统性地将扩散模型落地到表格数据；TabSyn（Zhang et al., 2024）进一步提出"VAE表示学习+潜空间扩散"的两阶段方案。在人口合成领域，Kang等人（2023）展示了DDPM用于人口合成的可行性，Tang等人（2025）讨论了生成过程中可行性与多样性的平衡。**这些工作为本项目提供了成熟的技术组件。**

本项目的核心洞见在于：**移动定位数据可作为收缩生态推断解空间的关键信息源**。Aiken等人（2022）在*Nature*上展示移动数据可与调查数据结合识别人口特征；Moro等人（2021）发现移动行为模式与收入隔离强相关。这些证据表明，移动行为数据携带人口属性信号，可作为桥接"人口属性"与"居住位置"的弱监督信号。

---

#### 1.3 技术路径

**本项目提出"多源对齐+联合扩散"的生成框架，通过三个阶段实现建筑物尺度人口画像的生成。**

**阶段一：多源数据对齐学习。** 在共享隐空间中对齐"人口属性-设备行为-建筑特征"三类实体。关键设计是：**不需要个体级配对标签，仅需利用区域级的统计一致性约束**。具体而言：（1）利用设备的居住区域将设备与同区域内建筑通过对比学习在隐空间中对齐；（2）对每个区域，将人口属性分布与设备分布做分布匹配，使两者在统计上相容。这种区域级约束足以建立跨数据源的语义对齐，而无需昂贵的个体级标注。

**阶段二：软配对构造。** 基于阶段一训练好的编码器，在隐空间中计算人口样本与建筑的相似度，构造概率配对关系。配对信号来自设备的行为轨迹——设备日常活动范围内的建筑更可能是其居住地，设备的POI访问模式隐含其社会经济特征。核心洞见是：**行为轨迹是人口属性的行为代理**，通过行为-空间关联，模型间接学到"什么样的人倾向于住在什么样的地方"。

**阶段三：联合扩散训练与生成。** 以阶段二构造的软配对数据为训练集，训练扩散模型学习 P(人口属性, 建筑特征 | 区域) 的联合分布。模型以PUMA等粗粒度区域作为条件变量组织训练。扩散模型的优势在于：通过"逐步去噪"的迭代过程，高维联合分布的学习被分解为多次低难度的局部修正。**验证层级（如街道/tract级边际分布）完全由模型自主泛化，而非外部引导**——这是评估模型是否真正学到联合结构的关键。

---

#### 1.4 可行性分析与风险应对 此阶段的目标是在共享隐空间中对齐"人口属性-设备行为-建筑特征"三类实体。关键洞见是：**不需要个体级（instance-level）的人-设备配对标签，仅需利用区域单元内可获得的弱监督信号实现对齐。** 这里的"区域单元"在美国验证场景可对应Census Block Group（CBG）/tract，在国内应用推广场景可对应街道/社区等统计单元。具体而言：（1）利用设备的居住单元（home CBG）将设备与同单元内建筑作为正样本，通过对比学习使二者隐空间接近；（2）对每个区域单元，基于普查/ACS边际构造该单元的人口属性分布（可通过采样/重加权得到一批满足边际的“人口样本”），经人口编码器映射为latent分布后，与同单元的device latent分布做分布匹配（distribution-level，如MMD），使两者在统计上相容。需要强调的是，这里的"统计对齐"指**区域内分布对齐**，而非对person与device做一一配对。

**阶段二：软配对构造。** 基于阶段一训练好的编码器，为每个区域内的人口样本构造其与建筑的概率配对。配对概率来自两类信号：一是隐空间相似度（反映属性/行为表征的相近性），二是设备的日常活动范围（activity space）形成的空间先验——**设备高频停留与访问所覆盖的活动范围内的建筑，更可能是其居住地**。这一过程不依赖收入-房价等手工规则，而是利用"行为即人口特征"的经验事实：POI访问模式可作为社会经济属性的行为代理。通过行为-空间关联，模型间接学到"什么样的人倾向于住在什么样的地方"。

**阶段三：联合扩散训练。** 以阶段二构造的软配对数据为训练集，训练扩散模型生成"（人口属性，建筑特征）"的联合分布，并**以PUMA作为可观测的条件变量**组织训练与采样（PUMA在微观样本/公开数据中可用，而tract/CBG等更细粒度标签在训练数据中往往不可得）。扩散模型的优势在于：通过"逐步去噪"的迭代过程，高维联合分布的学习被分解为多次低难度的局部修正；宏观统计与规则约束可在每个生成步骤中持续注入，实现"约束内生满足"而非事后修补。

**阶段四：约束引导采样。** 采样时注入区域级边际引导（仅限训练时见过、可观测的条件层级，如PUMA）和硬规则约束（如"年龄<16则收入=0"、建筑容量限制），但**不**注入更细粒度（如tract/CBG/街道）的边际引导。**验证层级必须是模型泛化的结果，而非引导的结果**：否则即便TVD下降，也只能说明“引导器能把边际拉准”，并不能证明模型学到了可迁移的联合结构与关联机制。

---

#### 1.4 可行性分析与风险应对

##### 1.4.1 数据可获取性

**本项目所需的核心数据均可通过公开渠道或合作协议获取。** 人口普查与统计年鉴数据可通过国家统计局公开获取；建筑信息可通过OpenStreetMap、高德地图开放平台等渠道获取，已有研究（Huang et al., 2021; Pajares et al., 2021）验证了利用建筑轮廓数据进行人口下推的可行性。移动定位/行为数据方面：在**方法验证阶段**可使用美国城市样例（如Detroit）中的商业移动数据（如Veraset等）开展可复现验证；在**应用推广阶段**可通过与腾讯位置大数据、电信运营商等机构的合作协议获取国内数据，申请人已与相关数据提供方建立初步合作意向。

##### 1.4.2 技术可行性

**本项目的核心技术组件均有成熟实现基础。** 扩散模型在表格数据生成上已有TabDDPM、TabSyn等成熟方案；对比学习与隐空间对齐技术在多模态学习中有广泛应用；基于移动数据的人口推断已有Aiken等人（2022）等顶刊工作支撑。人口合成领域也已出现Kang等人（2023）、Tang等人（2025）等直接应用扩散模型的先例。本项目的创新在于将这些成熟技术组件整合为服务于"建筑物尺度人口画像"这一特定目标的完整框架。

##### 1.4.3 风险应对措施

**针对潜在风险，本项目设计了以下应对措施：**

（1）**移动数据覆盖不均风险**：若部分区域移动数据稀疏，可降级为规则配对作为fallback，但需在结果中标注该区域的配对信号来源，并在验证时单独评估。

（2）**对齐学习效果不佳风险**：若隐空间对齐效果不理想，可引入更强的监督信号（如小规模人工标注的人-建筑配对样本）进行微调，或调整对齐损失的权重配比。

（3）**验证数据缺失风险**：若目标城市缺乏细粒度普查数据进行验证，可采用迁移验证策略——在有验证数据的城市上确认方法有效性后，迁移至目标城市，并通过一致性检验评估迁移效果。

（4）**计算资源风险**：扩散模型训练需要较大计算资源，但本项目可利用申请人所在单位的GPU集群资源，且TabDDPM等方案已验证在合理规模数据上的计算可行性。

---

### 2. 工作条件

（待补充：实验条件、设备、团队配置等）

### 3. 正在承担的相关科研项目

（待补充）

### 4. 完成国家自然科学基金项目情况

（待补充）

---

## 参考文献（APA7，已核实条目）

- Fishman, N., & McAuliffe, C. (2021). Deep ecological inference. *International Conference on Learning Representations (ICLR 2021)* (blind submission). OpenReview. https://openreview.net/forum?id=mxfRhLgLg_
- Hendrick, M., Rinaldo, A., & Manoli, G. (2025). A stochastic theory of urban metabolism. *Proceedings of the National Academy of Sciences of the United States of America, 122*(33), e2501224122. https://doi.org/10.1073/pnas.2501224122
- Kang, J., Kim, Y., Imran, M. M., Jung, G.-s., & Kim, Y. B. (2023). *Generating population synthesis using a diffusion model*. In *Proceedings of the Winter Simulation Conference*. https://informs-sim.org/wsc23papers/247.pdf
- King, G. (1997). *A solution to the ecological inference problem: Reconstructing individual behavior from aggregate data*. Cambridge University Press.
- Kotelnikov, A., Baranchuk, D., Rubachev, I., & Babenko, A. (2023). *TabDDPM: Modelling tabular data with diffusion models* (arXiv:2209.15421). arXiv. https://arxiv.org/abs/2209.15421
- La, D. M., Vu, H. L., Kamruzzaman, L., & Miller, E. (2025). Population synthesis: A problem-based review. *Transport Reviews, 45*(3), 366–389. https://doi.org/10.1080/01441647.2025.2469069
- Lebakula, V., Sims, K., Reith, A., Rose, A., McKee, J., Coleman, P., Kaufman, J., Urban, M., Jochem, C., Whitlock, C., Ogden, M., Pyle, J., Roddy, D., Epting, J., & Bright, E. (2025). LandScan Global 30 arcsecond annual global gridded population datasets from 2000 to 2022. *Scientific Data, 12*(1), 495. https://doi.org/10.1038/s41597-025-04817-z
- McCartan, C., & Kuriwaki, S. (2025). Identification and semiparametric estimation of conditional means from aggregate data (arXiv:2509.20194). arXiv. https://doi.org/10.48550/arXiv.2509.20194
- Tang, M., Lu, P., & Feng, Q. (2025). *Generating feasible and diverse synthetic populations using diffusion models* (arXiv:2508.09164). arXiv. https://doi.org/10.48550/arXiv.2508.09164
- Uğurel, E., Chen, C., Lee, B. H. Y., & Rodrigues, F. (2025). On predicting sociodemographics from mobility signals (arXiv:2511.03924). arXiv. https://doi.org/10.48550/arXiv.2511.03924

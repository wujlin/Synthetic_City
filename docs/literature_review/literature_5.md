我按你给的优先级，把检索重点放在「需求1：聚合约束引导的扩散采样」和「需求2：无配对数据下学习联合分布」上；其余需求（3–5）给出可直接迁移到你们场景的“技术抓手 + 代表文献”，并明确指出哪些方向目前在“表格/人口合成”里文献稀疏，反而可能构成贡献点。

---

需求1：聚合约束引导的扩散模型采样（⭐⭐⭐最高优先级）

1. 直接“命中”你们问题的关键文献：Distribution Guidance（批量/集合级别的分布约束）
   你们要的本质是：在采样过程中对一个 batch（一个 tract 的 N 个个体）施加“集合级别”的分布约束（年龄边际、收入边际等）。在扩散领域，一个非常接近的、可直接借鉴的做法叫 Distribution Guidance：它不是对每个样本逐个指定标签（sample guidance），而是“联合去噪整个 batch”，并推动 batch 的属性分布去逼近一个给定的目标分布。([CVF开放获取][1])

这篇工作的关键点非常贴合你们的“tract 内边际约束”叙事：

* 它明确提出：逐样本硬指定属性会伤质量，所以改为 batch-level 约束；“jointly denoise a whole batch of samples”，并让生成 batch 的属性分布与参考分布接近。([CVF开放获取][1])
* 它用一个可微的“分布预测器”把 batch 映射成一个分布估计，然后对“两个分布的距离”做反向传播，把梯度回传到去噪过程（相当于 classifier guidance 的集合级扩展）。([CVF开放获取][1])

对你们的启发是：你们 tract 的 ACS 边际完全可以扮演它的 “reference attribute distribution”；而“年龄分布/收入分布/就业状态分布”等就是它的 batch-level guidance 目标。

2. 从“理论/通用框架”层面支撑你们：把聚合约束当作 reward/energy 来做 guidance
   如果你们不想被某个“特定分类器/特定领域”锁死，近两年有两条很实用的路线：

* Universal Guidance（ICLR 2024）：强调“指导信号不一定是分类器”，可以把很多目标写成一个可计算的打分/奖励/能量函数，然后在采样时把它注入反向扩散更新（训练外/推理时注入为主）。([ICLR 会议录][2])
* FreeDoM（ICCV 2023）：同样是“训练外的能量引导”，把条件写成能量函数，用梯度去 steer 采样轨迹，达到条件控制。([CVF开放获取][3])

你们的聚合约束（比如 tract 内年龄直方图与 ACS 的距离）天然就是一个“能量/损失”：
E(X_batch) = D( agg(X_batch), target_marginal )
然后把 −∇E 当成 guidance。

3. 用“逆问题/后验采样”视角理解你们的聚合约束：DPS
   另一条很稳的解释是：把“聚合统计”当作观测 y，把一整个 tract 的 N 个样本拼成未知量 X，聚合算子 M(X)=y（直方图/计数/边际），在扩散采样中做后验采样或似然引导。Diffusion Posterior Sampling（DPS）明确讨论了：用扩散模型作为先验、用“似然项梯度”在采样时融合观测一致性，且不要求严格投影到完全一致（更适合带噪/现实场景）。([arXiv][4])

对你们来说，“ACS 边际”就像 noisy observation；你们不必追求每个 tract 都精确 match（硬约束），而是以合理的软约束强度逼近即可。

4. 你问的关键技术点：如何把“聚合统计距离”对单样本求梯度？
   这里给你一套“可落地且和文献一致”的做法（核心是：让聚合统计可微）——它本质上就是 Distribution Guidance 的 tabular 版本。([CVF开放获取][1])

(1) 类别变量（sex、就业状态、年龄分箱等）
如果你们的扩散输出对某个离散字段是 logits（或 soft one-hot），令第 i 个样本在 K 类上的概率为 p_i ∈ Δ^K，那么 batch 的“期望类频率”就是：
\hat{h} = (1/N) * Σ_i p_i
然后定义分布距离 D(\hat{h}, h_target)（TVD、KL、χ²、Wasserstein 都行），梯度自然回传到每个 p_i，再回传到 logits/去噪网络。

这个做法最大的优点：对离散变量不需要硬 argmax，也就不会“梯度断掉”。

(2) 连续变量（income）+ 直方图边际
如果你们做分箱边际（ACS 常见），要让直方图可微，常用两种“soft binning”：

* 高斯核 soft histogram：
  w_{ik} = exp(−(x_i−c_k)^2 / (2σ^2))，再对 k 归一化
  \hat{h}*k = (1/N) Σ_i w*{ik}
* Sigmoid 近似的 soft binning（对每个 bin 的上下界做平滑指示）

随后同样对 D(\hat{h}, h_target) 求梯度：
∂D/∂x_i = Σ_k (∂D/∂\hat{h}_k) * (∂\hat{h}_k/∂x_i)

(3) 采样时注入的数学形式（最常用的一类）
你们的设计文档里已经把“采样时聚合引导”写成 CFG 的聚合变体（先算当前分布，再算朝 target 的“梯度/方向”）。严格一点的写法通常是把约束写成 log-likelihood 或 energy：
guidance ∝ ∇*{x_t} log p(y | x_t)  或  −∇*{x_t} E( \hat{x}_0(x_t) )
Distribution Guidance 就是把 batch 分布损失对中间表示反传，把导数灌回反向扩散更新。([CVF开放获取][1])

(4) 实操提醒（很关键）

* batch size N 变小会让“分布估计”方差变大，指导会抖；Distribution Guidance 也专门 ablate 了 batch size。([CVF开放获取][1])
* guidance_scale 不宜常数拉满：更稳的做法是“后期强、前期弱”（越接近 x0 越强调满足约束），否则很容易牺牲多样性换一致性。

5. 这条线在“表格/人口合成”里是否文献稀缺？
   我检索到的“真正直接把 batch-level 边际约束作为 guidance”最典型的是上面这篇 Distribution Guidance（但它在图像公平生成语境中）。([CVF开放获取][1])
   在“表格扩散/人口合成”里，更多工作是“条件生成（sample-level condition）”或“后处理修正边际”，而不是“采样中实时对 tract 级边际做可微引导”。这意味着：你们如果把这套机制做成严谨的 tabular + tract-level aggregate guidance，很可能就是一个清晰的贡献点（尤其是配上理论视角：reward/energy guidance、后验采样/逆问题解释）。([ICLR 会议录][2])

---

需求2：无配对数据下学习联合分布 P(person, building | tract)（⭐⭐⭐最高优先级）

先给一个“结论句”，方便你们定架构：
仅给定 P(person|tract) 和 P(building|tract)，联合分布一般不可识别；任何方法本质上都在“选择一个 coupling（耦合/配对机制）”，必须引入额外假设/代价函数/弱监督信号（否则就是默认独立）。这一点在生态推断/数据融合/统计匹配里是经典结论。([gking.harvard.edu][5])

1. 你们当前 Scheme C 的“最小假设随机配对”= 默认独立耦合（需要警惕）
   你们文档里提出训练阶段用 tract 内“均匀随机采样 k 栋建筑”作为软标签，并强调不引入收入-房价等人为规则。从统计角度，这相当于用训练数据把 (person, building) 人为打散成近似独立样本，模型最容易学到的是：
   P(person, building | tract) ≈ P(person|tract) P(building|tract)

然后你们再在采样阶段用 ACS 边际做聚合引导，能强力修正 P(person|tract) 的边际，但它并不会自动“凭空创造” person 与 building 之间的相关结构（例如 income–rent/price 相关），除非你们的 guidance 或条件里明确引入了跨域耦合信号（比如对 (income, price_tier) 的联合约束或能量项）。

这点和你们文档中想验证 “Income-Price correlation” 的目标其实是张力很大的：你们要模型学到 income–price 相关，但训练数据如果是随机配对，这个相关在训练分布里接近 0，模型没有信号来源。

所以：需求2的解决方案，基本决定了 Scheme C 是否能真的“学到联合分布”，还是只能做到“边际正确 + 空间变量像噪声”。

2. 代表性“可直接搬到你们问题”的现代方案：OT-guided conditional diffusion（NeurIPS 2023）
   我这里最推荐你们优先精读的是：Optimal Transport-Guided Conditional Score-Based Diffusion Model（NeurIPS 2023）。它研究的就是“没有配对数据，如何训练条件/联合生成模型”，并且把 OT 当成“软配对器”。([ar5iv][6])

它的关键做法（对应你们的 person/building）：

* 在每个训练 batch 上估计一个最优传输耦合（他们用 L2 正则的 OT 来求 coupling），让两个未配对集合之间形成一个“概率匹配矩阵”；([ar5iv][6])
* 引入“compatibility function（兼容性函数）”和 “resampling-by-compatibility”，用这个 coupling 产生用于训练的伪配对，从而在没有真配对的情况下也能学到条件生成关系。([ar5iv][6])

对你们来说，几乎可以 1:1 映射成：

* 在 tract 内，对 persons 集合与 buildings 集合做 entropic OT / Sinkhorn（更常用的版本是熵正则 OT，数值稳定，且可控“接近独立 vs 接近硬匹配”）；([math.columbia.edu][7])
* cost/compatibility 由你们可用的弱先验构造（例如 household size vs capacity proxy，income vs price tier，car ownership vs parking proxy，等等）；
* 把 OT coupling 产生的 top-k 建筑候选当作你们文档里的“软标签”，但它不再是均匀随机，而是“数据+先验驱动的概率配对”，这与文档开头“用多源数据构造概率配对作为弱监督”的总叙事完全一致。

3. “只用边际”构造联合分布：Entropic OT / 最大熵耦合 / IPF 的关系
   如果你们暂时不想做 OT-guided diffusion，而是先要一个“联合分布估计器”来做 building assignment 或生成训练伪配对，那么熵正则 OT 是非常标准的选择：它在所有满足给定边际的耦合 Π(μ,ν) 里，最小化期望 cost + ε * KL(π || μ⊗ν)。([math.columbia.edu][7])

直觉上：

* ε → ∞ 时，π 会趋近于独立耦合 μ⊗ν（对应你们的均匀随机/独立假设）
* ε → 0 时，π 会逼近更“尖锐”的最优匹配（更像 hard assignment）

这给了你们一个非常清晰的“可解释旋钮”：在没有真配对数据时，你们能以“假设强度”来做敏感性分析，而不是拍脑袋选一种分配规则。

4. 更进一步：Schrödinger Bridge / 多边际耦合（多源数据融合）
   如果你们未来不止 person/building 两个边际，还想把 workplace/LODES、school、POI activity 等也纳入统一框架，那么“多边际耦合”会变得重要。Schrödinger bridge 本质上和熵正则 OT 同源，可以看成带随机过程的最大熵耦合；数据驱动 Schrödinger bridge给出了只用两端样本来拟合桥的算法路径。([math.nyu.edu][8])
   TreeDSB（NeurIPS 2023）把这种思路扩展到树结构的多边际情形，可用于“多分布融合/重心”等。([NeurIPS 会议录][9])

对你们来说，它的价值在于：你们不必强行把所有信息塞进一个“硬配对数据集”，而是把它们当作多个边际约束/观测，通过桥/耦合去统一。

5. 给你们一个非常具体的架构建议（直接影响可行性）
   如果你们坚持“训练阶段完全随机配对 + 采样阶段只用 ACS(person 边际) 做引导”，那模型要学到 “income–price correlation” 这类 person-building 依赖关系会非常困难（因为训练分布里被你们人为打散了）。

要把需求2做实，我建议你们在 Scheme C 的训练阶段至少引入以下三种之一（按推荐顺序）：

* 方案2A（最推荐，最贴合你们叙事）：OT coupling 产生“概率配对软标签”，替换均匀随机 soft labels；训练联合扩散时按 coupling resample。对应 OT-guided diffusion 的路线。([ar5iv][6])
* 方案2B（折中）：训练仍可随机，但采样 guidance 增加一个“跨域能量项”，例如让生成样本的 (income, price_tier) 相关、或让高收入更倾向高价房（哪怕是非常弱的 soft constraint）。这相当于把“联合结构”的学习推迟到推理时，用 energy guidance 去塑形。([ICLR 会议录][2])
* 方案2C（如果你们有任何联合的聚合信息）：哪怕只有很粗的 cross-tab（如 tract 级 income×rent bins），也足够把它变成“集合级联合分布约束”，用需求1那套 batch guidance 来学依赖结构（这是最“统计闭环”的版本）。([CVF开放获取][1])

---

需求3：表格扩散的约束注入机制（硬约束 + 软约束）

这一块我建议你们把“硬约束”和“软约束”分开处理：硬约束追求 0 violation；软约束追求逼近目标统计且保持多样性。

1. 硬约束（规则/可行性）怎么做：CTDF（2025）给了一个很工程化、但效果明确的答案
   Constrained Tabular Diffusion for Finance（CTDF）提出在每一步反向扩散都插入“feasibility operation（可行性算子）”，可同时处理：

* 数值型约束用欧式投影（Euclidean projection）；
* 类别型约束通过把不合法类别概率置零再重归一；
* 更复杂的 functional/symbolic constraints 也能纳入，从而实现“zero violations”。([OpenReview][10])

对你们的规则（如 age<18 ⇒ income=0，或 structural zeros）而言，这比“采样完后再修”更干净：你可以把 feasibility operator 作为一个模块插到 sampler 里，而不是把规则塞到数据后处理。

2. 软约束（边际/分布目标）怎么做：回到需求1的 batch guidance
   这块就用 Distribution Guidance / energy guidance：把软目标写成可微的统计距离，在采样时持续施加。([CVF开放获取][1])

3. “训练时约束 vs 采样时约束”的取舍：PDM / Mirror Diffusion 代表了更偏理论的“约束采样”路线

* Projected Diffusion Models 把反向采样写成“无约束更新 + 投影/约束操作”的迭代过程，目标是保证输出在约束集合内。([arXiv][11])
* Mirror Diffusion Models 则从约束优化/镜像映射角度处理一些凸约束集合。([NeurIPS Papers][12])

如果你们后续真要做“严格满足 tract 级总人数、严格满足某些硬边际”的版本，这类工作可以作为你们方法论叙事的理论支撑（哪怕你们最后用的是更工程的 feasibility operator）。([arXiv][11])

---

需求4：建筑级/地址级人口空间化的最新方法论（SOTA & 空白判断）

这里需要先把两类文献分开，否则容易“看起来很多，实际上不对口”：

A) 很多文献做的是“建筑级人口估计/制图”（输出是每栋楼的人口数或密度），不生成个体微观属性
例如：

* ISPRS 2024：利用建筑体量/3D 建筑信息（由高程模型、建筑轮廓、土地利用等推导）来估计住宅建筑人口，属于 building-level population estimation（计数）。([国际摄影测量与遥感学会档案][13])
* Remote Sensing 2022：融合多源地理特征，用随机森林等模型进行 building-level population spatialization（同样偏计数/分配权重）。([MDPI][14])
* 以及更早但常被引用的 building-level/细尺度人口估计方法（例如 2021 的工作讨论 building-level estimation 与特征相关性）。([spj.science.org][15])

B) 相对少一些的文献做的是“空间化的合成人口”（输出个体/家庭），但空间分配常作为独立模块
例如：

* CEUS 2022：提出集成框架生成空间细粒度、异质性的合成人口，并显式构建 built environment 的空间实体（ontology-based data fusion），属于“合成 + 空间实体融合”的路线。([科学网][16])

C) 最新方向苗头：把“坐标”直接纳入生成模型

* 2025 arXiv：Population synthesis with geographic coordinates：先用 normalizing flows 把空间坐标映射到更规则的潜空间，再与 VAE 结合生成带空间相关的合成人口特征，强调学习空间与非空间特征的联合分布、利用空间自相关。([arXiv][17])

对你们“建筑物尺度、个体属性 + 建筑锚定”的目标，我的判断是：
“用现代生成模型直接学 (person attributes, building/address) 的联合分布，并在采样时满足 tract 级边际约束”这一组合，在公开文献里仍然相对稀疏；现有工作要么偏计数制图（不生成微观个体），要么微观合成与空间分配分离（靠后处理/规则/优化）。([国际摄影测量与遥感学会档案][13])

这和你们文档里想强调“分离式 Scheme B 不符合叙事、需要联合生成与内生约束”的动机是一致的。

---

需求5：Ecological Inference / 生态推断（补充，但对你们“无配对”问题是底层理论背景）

1. 经典定义与不可识别性
   King 对生态推断的经典表述是：用聚合数据推断个体层行为/关联，这正是你们“从 ACS 边际与建筑边际推断 person-building 关系”的抽象。([gking.harvard.edu][5])
   同时，生态推断长期争议点也在于：没有额外假设时会出现生态谬误与不可识别，Freedman 等对 King 方法也有系统性的批评与反例讨论（提醒大家别过度相信 diagnostics）。([stat.berkeley.edu][18])

2. 现代进展：更弱条件下的识别与半参数估计（2025）
   例如 McCartan 2025 的工作，专注于“只观察聚合均值/边际时如何识别/估计 conditional means”，强调很多既有 EI 方法隐含了过强的聚合过程假设，并给出更弱条件下的识别与估计路径。([arXiv][19])

对你们的实际意义是：需求2一定要在论文里写清楚“我们额外引入了什么识别假设/弱监督信号”（OT cost、compatibility、能量项、或任何 cross-aggregate 信息），并报告对关键超参（如 OT 的 ε、cost 权重、guidance 强度）的敏感性，否则很容易被 reviewer 用 EI 的不可识别性来质疑。

---

对你们项目（以及 Partner 的最优工作顺序）的建议

第一优先（直接决定架构可行性）：

* 先把需求1做成一个“可运行的 tract-batch guidance sampler”：
  用可微直方图/soft binning + 分布距离（TVD/KL/Wasserstein）构造能量项，按 Distribution Guidance 的思路对 batch 施加引导（你们文档里的 aggregate_guided_sampling 就是这个雏形）。
* 同时把需求2从“随机配对”升级为“OT 概率配对”：
  用 entropic OT 在 tract 内产生 coupling，再按 coupling resample 成训练软标签；或者直接参考 OT-guided diffusion 的训练流程。([ar5iv][6])

第二优先（让系统可控、可解释、可保证可行性）：

* 把硬规则交给 feasibility operator / 投影模块（CTDF/PDM 思路），把软边际交给 batch guidance（Distribution Guidance/Universal Guidance 思路），两者分治。([OpenReview][10])

第三优先（叙事与实验对齐）：

* 你们文档里计划用 “Income-Price correlation、空间自相关”等指标验证“学到了合理配对模式”。
  建议把这些指标也纳入“需求2 的 coupling/compatibility 设计”：如果 coupling cost 里完全不包含 income–price 相关的任何信号，那实验里想得到稳定正相关会非常难（这不是实现问题，是信息论/识别问题）。

---

我这次整理出的“必读文献清单”（按你们优先级排序）

需求1（聚合约束 guidance）：

* Parihar et al., 2024, “Balancing Act: Distribution-Guided Debiasing in Diffusion Models”（batch-level distribution guidance，非常贴合 tract 边际约束）([CVF开放获取][1])
* Bansal et al., 2024, “Universal Guidance for Diffusion Models”([ICLR 会议录][2])
* Chung et al., 2023, “Diffusion Posterior Sampling for General Noisy Inverse Problems”（把边际/聚合看作观测的一类后验采样解释）([arXiv][4])

需求2（无配对联合分布 / coupling）：

* Gu et al., NeurIPS 2023, “Optimal Transport-Guided Conditional Score-Based Diffusion Model”（OT 软配对 + resampling-by-compatibility，几乎直接对应 person/building）([ar5iv][6])
* Nutz, 2022, “Introduction to Entropic Optimal Transport”（理解 entropic OT、耦合、ε 的意义）([math.columbia.edu][7])
* Tabak et al. / Pavon et al., 2021, “The Data-Driven Schrödinger Bridge”（把耦合/桥当成多源数据融合基础模块）([math.nyu.edu][8])
* Noble et al., NeurIPS 2023, “Tree-Based Diffusion Schrödinger Bridge …”（多边际融合）([NeurIPS 会议录][9])

需求3（tabular 约束注入）：

* “Constrained Tabular Diffusion for Finance (CTDF)”（feasibility operator，强调 0 violation）([OpenReview][10])
* “Projected Diffusion Models”([arXiv][11])
* “Mirror Diffusion Models”([NeurIPS Papers][12])

需求4（建筑级空间化背景）：

* Vergara et al., ISPRS 2024 building-level population estimation（计数/估计类）([国际摄影测量与遥感学会档案][13])
* Wang et al., Remote Sensing 2022 building-scale population spatialization（计数/制图类）([MDPI][14])
* Zhou et al., 2022 “Creating spatially-detailed heterogeneous synthetic populations …”（合成 + 空间实体融合）([科学网][16])
* 2025 arXiv “Population synthesis with geographic coordinates”（把坐标纳入生成）([arXiv][17])

需求5（生态推断理论背景）：

* King, “A Solution to the Ecological Inference Problem”([gking.harvard.edu][5])
* Freedman critique（提醒不可识别与诊断风险）([stat.berkeley.edu][18])
* McCartan 2025（更弱条件下的识别/估计）([arXiv][19])

---

如果你希望我下一步继续“更工程化地落地”，我建议我们就按两条线并行推进：
(1) 把你们文档里的 aggregate_guided_sampling 严格化成“可微直方图 + 正确的链式梯度 + 分步 guidance schedule”；
(2) 在同一 tract 内做一个最小可行的 OT coupling（先只用 2–3 个兼容性特征），把它替换掉均匀随机 soft labels，看看 income–price correlation 是否开始出现。

[1]: https://openaccess.thecvf.com/content/CVPR2024/papers/Parihar_Balancing_Act_Distribution-Guided_Debiasing_in_Diffusion_Models_CVPR_2024_paper.pdf "https://openaccess.thecvf.com/content/CVPR2024/papers/Parihar_Balancing_Act_Distribution-Guided_Debiasing_in_Diffusion_Models_CVPR_2024_paper.pdf"
[2]: https://proceedings.iclr.cc/paper_files/paper/2024/file/dfbb3d1807b21dadee735eb75069ada4-Paper-Conference.pdf "https://proceedings.iclr.cc/paper_files/paper/2024/file/dfbb3d1807b21dadee735eb75069ada4-Paper-Conference.pdf"
[3]: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf "https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf"
[4]: https://arxiv.org/abs/2209.14687 "https://arxiv.org/abs/2209.14687"
[5]: https://gking.harvard.edu/files/gking/files/part1.pdf "https://gking.harvard.edu/files/gking/files/part1.pdf"
[6]: https://ar5iv.org/pdf/2311.01226 "https://ar5iv.org/pdf/2311.01226"
[7]: https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf "https://www.math.columbia.edu/~mnutz/docs/EOT_lecture_notes.pdf"
[8]: https://math.nyu.edu/faculty/tabak/publications/SB.pdf "https://math.nyu.edu/faculty/tabak/publications/SB.pdf"
[9]: https://proceedings.neurips.cc/paper_files/paper/2023/file/ad08767706825033b99122332293033d-Paper-Conference.pdf "https://proceedings.neurips.cc/paper_files/paper/2023/file/ad08767706825033b99122332293033d-Paper-Conference.pdf"
[10]: https://openreview.net/pdf/a6e75c8d9bb48bccea51c921903cfe46bc4f424d.pdf "https://openreview.net/pdf/a6e75c8d9bb48bccea51c921903cfe46bc4f424d.pdf"
[11]: https://arxiv.org/html/2402.03559v2 "https://arxiv.org/html/2402.03559v2"
[12]: https://papers.neurips.cc/paper_files/paper/2023/file/85f5c7372625d1e0df0e3996f85062d6-Paper-Conference.pdf "https://papers.neurips.cc/paper_files/paper/2023/file/85f5c7372625d1e0df0e3996f85062d6-Paper-Conference.pdf"
[13]: https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/453/2024/isprs-archives-XLVIII-4-W8-2023-453-2024.pdf "https://isprs-archives.copernicus.org/articles/XLVIII-4-W8-2023/453/2024/isprs-archives-XLVIII-4-W8-2023-453-2024.pdf"
[14]: https://www.mdpi.com/2072-4292/14/8/1811 "https://www.mdpi.com/2072-4292/14/8/1811"
[15]: https://spj.science.org/doi/10.34133/2021/9803796 "https://spj.science.org/doi/10.34133/2021/9803796"
[16]: https://www.sciencedirect.com/science/article/abs/pii/S0198971521001241 "https://www.sciencedirect.com/science/article/abs/pii/S0198971521001241"
[17]: https://arxiv.org/html/2510.09669v1 "https://arxiv.org/html/2510.09669v1"
[18]: https://www.stat.berkeley.edu/~census/515.pdf "https://www.stat.berkeley.edu/~census/515.pdf"
[19]: https://arxiv.org/abs/2509.20194 "https://arxiv.org/abs/2509.20194"

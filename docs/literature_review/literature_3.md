# 文献梳理需求清单（主线清晰版）

## 0. 主线（建议你在综述里用这一条线串起来）
1) **传统方法的“结构性失效”证据**：在高维/稀疏/层级一致性约束下，边际拟合并不等于联合结构重建；候选库方法受覆盖度约束；深度生成若把约束后置，往往要靠后处理修补，且会破坏结构。（A）  
2) **扩散模型为什么更合适**：score/梯度场本质上在学习联合分布的“几何结构”，并且可以把约束通过guidance嵌入采样过程，再配合投影/修正实现软硬结合。（B）  
3) **表格扩散与合成人口先例**：TabDDPM/STaSy/D3PM证明表格扩散可行；WSC与arXiv的合成人口扩散工作说明“能补sampling zeros/但仍有局限”，并提出可用的权衡指标。（C）  
4) **验证框架要能落到工程**：对齐PopulationSim/ActivitySim的控制量与验证流程，再补上“时间层”的动态验证（如手机信令/移动数据），形成三层验证。（D）  
5) **分层/级联生成与空间下推**：从“家庭→个体”的属性层级，到“区域→建筑”的空间层级，用分层扩散/级联扩散的思想对齐传统分层拟合逻辑。（E/F）

---

## A. 传统方法失效的实证证据（用来“打地基”）

### A1. IPF/IPU 在高维场景下关联丢失（边际拟合 ≠ 联合重建）
- **要支撑的claim**：只匹配边际/少量控制变量，并不能保证未控制变量之间的相关结构被恢复；高维联合表稀疏会加剧该问题。  
- **你要找/摘录的证据类型**
  - 高维下：联合分布稀疏、零单元（zero-cells）多、需要稀疏结构/降维/替代方法等论述与实验对比。  
  - “受控—未控”相关性不强时，IPF对未控变量误差会放大（可作为“关联丢失/漂移”的理论+经验依据）。
- **起步文献（可先从摘要/方法/实验章节抓关键信息）**
  - IPF方法与局限综述（含零单元/单层合成等问题）：Choupani (2016)《Population Synthesis Using IPF: A Review…》:contentReference[oaicite:0]{index=0}  
  - IPF在空间微观模拟中的性能评估与讨论：Lovelace & Ballas (2015)（JASSS）:contentReference[oaicite:1]{index=1}  
  - 用copula替代IPF重建联合分布、并强调高维稀疏问题：Jeong et al. (2016)（PLOS ONE）:contentReference[oaicite:2]{index=2}  
  - 讨论IPF/IPF类方法局限与替代路径的综述性框架：Barthélemy & Toint (2013):contentReference[oaicite:3]{index=3}  

**建议检索式**：`("iterative proportional fitting" OR IPF) high-dimensional sparsity zero-cells correlation uncontrolled variables synthetic population`

---

### A2. CO（组合优化）在缺乏微观样本时的“候选库覆盖度限制”（sampling zeros补不齐）
- **要支撑的claim**：CO类方法往往是“从样本库里挑选/重加权”，**候选池里没有的组合**（sampling zeros）很难生成；样本缺失/稀疏会直接变成覆盖度瓶颈。  
- **你要找/摘录的证据类型**
  - CO/IPU依赖微观样本（seed/HTS）的机制性描述；  
  - 与“sampling zeros/structural zeros”的对照阐释：传统方法补不出sampling zeros，深度生成能补但会引入structural zeros（不可行组合）。
- **起步文献**
  - 将方法家族分为SR/CO，并明确CO与样本依赖关系：Barthélemy & Toint (2013):contentReference[oaicite:4]{index=4}  
  - 讨论无样本（sample-free）合成为何“少见且困难”的CO路线：Huynh (2016)（JASSS）:contentReference[oaicite:5]{index=5}  
  - 明确“sampling zeros/structural zeros”并给出量化指标（feasibility/diversity权衡）：Tang et al. (2025, arXiv):contentReference[oaicite:6]{index=6}  
  - WSC 2023 合成人口扩散工作把“能生成sampling zeros”作为动机之一，并报告相对基线的误差指标（示例：SRMSE数值对比出现在摘要/正文段落中）：Kang et al. / Corlu et al. (WSC 2023):contentReference[oaicite:7]{index=7}  

**建议检索式**：`combinatorial optimization population synthesis candidate pool "sampling zeros"`, `IPU CO seed sample limitation`

---

### A3. VAE/GAN 用于合成人口后，约束需后处理修复（约束后置 → 结构破坏/不可行样本）
- **要支撑的claim**：深度生成（VAE/GAN）能补稀有组合，但容易生成不可行（structural zeros）；若把硬约束放在生成之后做过滤/重采样/修补，往往会改变分布结构或效率低。  
- **你要找/摘录的证据类型**
  - 明确提出 sampling zeros vs structural zeros，并给出定量对比（可直接做“问题定义+指标”引用）。  
  - 典型后处理手段：规则过滤、拒绝采样、再加权/再训练等，并指出仅靠后处理不足。
- **起步文献**
  - 人口合成语境下的“可行性-多样性”度量与DGM问题刻画（含定量结果）：Kim & Bansal (2022, arXiv):contentReference[oaicite:8]{index=8}  
  - “可定制/可约束”的表格合成：指出仅拒绝采样不够，需要把约束融入训练或采样：Vero et al. (CuTS, 2023/2024):contentReference[oaicite:9]{index=9}  
  - 把一般DGMs转成“保证满足线性约束”的Constrained DGM（把约束内生化，而不是生成后修）：Stoian et al. (ICLR 2024):contentReference[oaicite:10]{index=10}  

**建议检索式**：`population synthesis VAE GAN structural zeros post-processing`, `constrained tabular generative models rejection sampling not sufficient`

---

## B. 扩散模型机制的理论支撑（用来“立论：为什么扩散适合联合结构+约束”）

### B1. Score-based 生成模型的理论基础（梯度场编码联合结构）
- Song & Ermon (2019) 提出通过估计数据分布梯度（score）并用Langevin动态采样的框架:contentReference[oaicite:11]{index=11}  
- Song et al. (2021) 用SDE统一score-based与扩散模型，并给出反向SDE/ODE采样视角:contentReference[oaicite:12]{index=12}  

### B2. Classifier-free guidance（把“条件/约束信号”嵌入采样过程）
- Ho & Salimans (2022) 系统阐述 classifier-free guidance：用条件/无条件score线性组合实现可控生成:contentReference[oaicite:13]{index=13}  

### B3. Constrained / guided sampling（软约束引导 + 硬约束投影/一致性）
- Dhariwal & Nichol (2021) 介绍classifier guidance等“引导采样”思想:contentReference[oaicite:14]{index=14}  
- Chung et al. (2022) 提出Diffusion Posterior Sampling（DPS）等把观测一致性/后验采样融合进扩散采样路径的做法:contentReference[oaicite:15]{index=15}  

> 你在写作里可以用一句话把这段“翻译”成：score近似联合分布的梯度场；guidance相当于把约束/条件变成采样过程中的额外漂移项；硬约束再用投影/一致性步骤兜底。

---

## C. 扩散用于表格/合成人口的直接先例（用来“证明可行+定位差距”）

### C1. 表格扩散：TabDDPM、STaSy、D3PM
- TabDDPM（Kotelnikov et al., 2022/2023）：混合连续/离散特征的通用表格扩散框架与基准评测:contentReference[oaicite:16]{index=16}  
- STaSy（Kim et al., 2022/2023）：score-based表格合成及训练策略:contentReference[oaicite:17]{index=17}  
- D3PM（Austin et al., 2021）：离散状态空间的扩散建模（对纯离散属性/类别变量很关键）:contentReference[oaicite:18]{index=18}  

### C2. 合成人口扩散：WSC 2023 的发现与局限
- WSC 2023《Generating Population Synthesis Using a Diffusion Model》作为“合成人口+扩散”的早期工程先例:contentReference[oaicite:19]{index=19}  

### C3. feasibility–diversity 指标：定位“权衡机制”
- Tang et al. (2025, arXiv) 以sampling zeros / structural zeros刻画可行性与多样性权衡，并用于扩散式人口合成评估:contentReference[oaicite:20]{index=20}  
-（可选补强）Kim & Bansal (2022) 同样把可行性/多样性量化并给出基线对比:contentReference[oaicite:21]{index=21}  

---

## D. 验证方法的已有实践（把“怎么证伪/证真”写成可复用流程）

### D1. PopulationSim / ActivitySim 的验证流程（支撑“三层验证框架”）
- PopulationSim 官方验证页：控制量差异、地理层级、验证统计等:contentReference[oaicite:22]{index=22}  
- PopulationSim 对多地理层级与户/人双层控制的配置说明（可当“框架设计依据”引文）:contentReference[oaicite:23]{index=23}  
- ActivitySim 技术报告中对“PopulationSim生成+后处理+在ABM里使用”的描述（用于衔接下游验证）:contentReference[oaicite:24]{index=24}  

### D2. 基于手机信令/移动数据的人口动态验证（支撑“时间层验证”）
- 经典动态人口制图：Deville et al. (2014) 用手机数据进行动态人口分布映射:contentReference[oaicite:25]{index=25}  
- “代表性/偏差”检验类工作：Mu et al. (2024) 将移动大数据与最新普查对比评估代表性:contentReference[oaicite:26]{index=26}  
- 官方统计视角综述与案例（含与普查/出行调查验证的引用线索）：US Census Bureau 报告:contentReference[oaicite:27]{index=27}  

---

## E. 分层/级联生成方法：理论与实践（把“家庭→个体→空间”写成方法路线）

### E1. Cascaded Diffusion（由粗到细）
- Ho et al.（Cascaded Diffusion Models）提出多级扩散管线：低分辨率到高分辨率逐级细化:contentReference[oaicite:28]{index=28}  
- Imagen（Saharia et al. 2022）同样是“多阶段/多模块”扩散体系的代表:contentReference[oaicite:29]{index=29}  

### E2. Latent Diffusion / 分层扩散（先骨架后细节）
- Rombach et al.（LDM）把扩散搬到潜空间，形成“先压缩表征→再生成细节”的典型分解逻辑:contentReference[oaicite:30]{index=30}  

### E3. 传统分层拟合：PopulationSim/IPU 的层级一致性逻辑（与传统方法对话）
- IPU用于同时匹配户与人的属性分布：Ye et al. (2009):contentReference[oaicite:31]{index=31}  
- 多层拟合算法（HIPF等）与“人-户层级”合成讨论：Müller & Axhausen 等（multi-level fitting）:contentReference[oaicite:32]{index=32}  
- PopulationSim 对“户/人+多地理层级控制”的工程化实现说明:contentReference[oaicite:33]{index=33}  

### E4. 家庭-个体一致性问题（支撑“属性层次”必要性）
- 你可以把这一块的“问题定义”落到：**joint household–person generation**、**consistency**、**illogical households** 等关键词；例如 Aemmer et al. (2022) 讨论联合生成与小群体/稀疏样本场景:contentReference[oaicite:34]{index=34}  
-（可选）“一阶段/一致性强约束”的替代路线：如基于Gibbs采样/层级一致性的方案讨论:contentReference[oaicite:35]{index=35}  

---

## F. 空间层次与建筑物尺度生成（把“区域→建筑”的下推与扩散分层对齐）

### F1. 建筑物尺度人口分配：已有方法
- Dasymetric mapping + 建筑数据：Pirowski et al. (2024) 基于建筑数据的人口分布高分辨率分析:contentReference[oaicite:36]{index=36}  
- 30m/栅格级dasymetric人口产品与流程：Swanwick et al. (2022)（Scientific Data）:contentReference[oaicite:37]{index=37}  
- Building/LiDAR/体量（volumetric）路线：  
  - 建筑级人口估计框架示例：Chen et al. (2021):contentReference[oaicite:38]{index=38}  
  - Areametric/Volumetric（含灾害/应急场景动机）：Lwin & Murayama (2009):contentReference[oaicite:39]{index=39}  

### F2. 多尺度空间下推/分解（支撑“区域→建筑”的分层逻辑）
- Top-down人口空间下推的综述与典型路线（含dasymetric/ML等）：Wardrop et al. (2018, PNAS):contentReference[oaicite:40]{index=40}  
- WorldPop 方法页（RF-dasymetric 等）作为工程侧权威入口:contentReference[oaicite:41]{index=41}  
-（可选）多尺度动态人口估计/下推（强调多尺度框架）：2025 的一篇多尺度动态人口估计工作:contentReference[oaicite:42]{index=42}  

---

## 附：建议你在“读文献时统一提取”的字段（保证梳理出来主线不散）
- 场景：数据维度（属性数/类别数/层级数）、稀疏度（zero-cells比例/罕见组合）、是否有人-户层级、是否多地理层级  
- 约束：边际约束（哪些）、一致性约束（哪些）、硬约束处理方式（投影/拒绝采样/规则过滤/整数化等）  
- 指标：边际误差（RMSE/APE等）、联合结构（相关/互信息/高阶交互）、sampling zeros恢复率、structural zeros率（feasibility）、多样性指标（diversity）  
- 下游：ActivitySim/ABM敏感性（出行/活动/拥堵等输出对人口合成误差的传播）

（如果你愿意，我也可以在这份清单基础上，直接帮你把每个小点扩成“文献卡片模板 + 预计写作段落（每段该引用哪几篇）”。）

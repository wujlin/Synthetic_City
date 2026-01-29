# 合成人口（SynthPop）代码架构：基于扩散的多层次人口生成方法（Detroit 落地 v0）

> 核心叙事：人口画像不是“把若干边际拼起来”，而是**在宏观统计约束与多源城市证据条件下重建并采样高维联合分布**。工程上对应的是一条可闭环的管线：**数据 → 条件构建 → 受控生成（软约束引导 + 硬约束渐进校正）→ 三层验证 → 误差归因与迭代**。

本文件用于给 PI review 的“架构方案稿”：强调**为什么这样拆模块**，而不是堆技术名词。Detroit 数据结构与落盘口径详见 `docs/detroit_code_data_structure.md` 与 `docs/WORKSTATION_GUIDE.md`。

---

## 0. 设计原则（KISS）

- **先把闭环跑通，再追求更细的模型**：v0 先支持 Detroit 的静态/日夜两时段版本；小时级动态、通勤 OD 等放到后续迭代。
- **训练学结构、采样管约束**：训练期以学习联合结构与条件响应为主；约束满足以“采样过程内生满足”为核心（软约束引导 + 硬约束渐进校正），避免事后筛选破坏分布。
- **多层次 ≠ 多工程复杂度**：多层次的本质是“宏观—建筑—家庭/个体—时间”的组织方式；代码用清晰的数据接口把层次切开，而不是堆一堆相互耦合的脚本。

---

## 1. 方法路线 → 工程模块的对应关系

| 方法要点（NSFC/workflow.md 与 NSFC/生成扩散.md） | 工程落点（模块/产物） | 目的 |
|---|---|---|
| 高维联合分布重建 | `src/synthpop/model/`（扩散生成模型 API） | 学到“关联结构”，而不是只拟合边际 |
| 多层次级联（区域→建筑→家庭/个体→时间） | `src/synthpop/pipeline/`（stage 编排） + `src/synthpop/features/`（条件构建） | 把“高维难题”拆成可控的分步生成 |
| 条件与锚定（宏观条件 + 空间分配特征） | `src/synthpop/features/` + `src/synthpop/spatial/building_allocation.py` | 让生成条件与空间落位逻辑都可审查、可复现 |
| 软约束引导（宏观统计拉回） | `src/synthpop/constraints/soft_guidance.py` | 让生成总体贴近目标统计口径 |
| 硬约束渐进校正（结构性零/逻辑可行性） | `src/synthpop/constraints/hard_rules.py` + `src/synthpop/constraints/projection.py` | 保证可行域内采样，减少拒绝采样与事后修补 |
| 三层验证（统计/空间/时间） | `src/synthpop/validation/` + `outputs/runs/<run_id>/metrics/` | 把“真实性”写成可检验指标，并能反向定位偏差来源 |

---

## 2. 技术选型（v0）：TabDDPM 主线

PI review 指出的最大技术风险在于“混合类型表格扩散不是 trivial”。为把风险变成可控的工程路径，v0 明确采用 **TabDDPM 风格**作为主线：

- **扩散形式**：Gaussian DDPM（在连续向量空间做加噪/去噪）
- **混合类型处理（v0）**：连续变量标准化后直接进入扩散；离散变量先 one-hot（或低维嵌入）映射到连续空间，与连续变量拼接后进入同一去噪网络联合建模；采样后对离散片段做 `argmax` 解码
- **条件化（v0 PoC）**：先在 PUMS 子集上验证“混合变量扩散 + 宏观条件化（按 PUMA）”可训练、可采样；后续把宏观单元升级到 tract/BG，并将建筑特征用于显式分配（或作为可选条件做对照实验）

为什么 v0 选 TabDDPM 而不是 STaSy/D3PM：
- **可控性与成本**：TabDDPM 路线可以先用统一的连续扩散把 PoC 跑通；STaSy 的离散 score 估计与 D3PM 的纯离散状态空间会显著增加实现复杂度
- **可迁移性**：v0 把“编码/解码/条件化接口”固定下来后，后续可把离散部分替换为 Multinomial/D3PM 等离散扩散，而无需推翻整个 pipeline

落地入口（v0）：
- 模型实现：`src/synthpop/model/diffusion_tabular.py`（TabDDPM-style）
- 最小技术验证脚本：`tools/poc_tabddpm_pums.py`（PUMS 子集 PoC；不写复杂约束）

---

## 3. 最小可用管线（v0）定义：我们先做成什么样

### v0 默认决策（PI 建议口径）

- 研究区：Detroit **city boundary**（Wayne County 作为 robustness check）
- 层级：v0 **person-only**（暂不显式生成 household 结构；v1 再补户—人一致性级联）
- 时间维度：v0 **静态**；日/夜两时段作为 v0.5
- 控制量层级：**tract 主控 + BG 诊断/细化**
- 属性最小集：年龄/性别/收入/就业状态/（个人所属）家庭规模
- 空间锚定：**Building + POI** 足够；parcel 作为 optional

**v0 目标输出**（见 `docs/detroit_code_data_structure.md` 的 outputs 约定）：
- `synthetic/persons.parquet`
- `synthetic/persons_geo.parquet`（可选：先输出 tract/BG 落点，便于两阶段空间分配）
- `synthetic/assignments_building.parquet`（人/户 → building）
- `metrics/stats_metrics.json`（边际 + 关键二阶关联 + 规则违规率）
- `metrics/spatial_metrics.json`（用地/POI 一致性等弱监督指标）
- `metrics/temporal_metrics.json`（若启用日/夜两时段）

**v0 的默认“闭环粒度”**（可由 PI 定稿）：
- 研究区：Detroit city boundary（可扩展到 Wayne County）
- 控制量层级：以 tract 为主；BG 用作细化与诊断（避免 BG 稀疏导致过拟合）
- 时间维度：静态 + 日/夜两时段（二选一：先静态也可）
- 种子样本：ACS PUMS household/person（公开可得）

---

## 4. 管线分解（Pipeline Stages）

> 这不是“脚本步骤清单”，而是把每一步的**输入/输出**写清楚，确保后续迭代不会互相污染。

### Stage A：数据标准化（raw → processed）

**输入**：`data_root/detroit/raw/*`（TIGER/ACS/PUMS/Buildings/POI 等）  
**输出**：`data_root/detroit/processed/*`（统一 schema 的 parquet / geoparquet）

核心产物：
- `processed/geo_units/geo_units.parquet`：city/tract/BG 统一管理（含 geometry）
- `processed/buildings/buildings.parquet`：building footprints + 派生特征 + (bg/tract crosswalk)
- `processed/pums/{pums_households,pums_persons}.parquet`：清洗后的种子样本
- `processed/marginals/marginals_long.parquet`：可直接用于约束/评估的控制量长表
- `processed/rules/rules.yaml`：结构性零与层级一致性规则

### Stage B：条件构建（processed → features）

**输入**：`processed/*`  
**输出**：`processed/features/*`（可复用的条件向量/索引）

两类条件向量（建议）：
- **区域条件**（tract/BG/PUMA）：宏观边际摘要 + 环境摘要（作为扩散模型的条件输入）
- **建筑特征**（building）：footprint/height/capacity/price tier/POI 邻域等（作为空间分配模块的输入；v0 不进入训练条件）

> 原则：条件/特征是“生成/分配时喂给模型/分配器的信号”，必须可复现并写入磁盘；不要临时在 notebook 里算完就丢。

### Stage C：训练（学习联合结构）

**输入**：种子样本（PUMS）+ 宏观条件向量（v0 PoC：PUMA one-hot）  
**输出**：`outputs/runs/<run_id>/checkpoints/`（模型与配置快照）

训练期做两件事：
- v0：学到 person-level 的联合结构（混合类型属性耦合）
- v0 PoC：验证宏观条件化训练可行（先按 PUMA；后续升级到 tract/BG；建筑特征用于显式空间分配并做消融）

### Stage D：采样（受控生成：软约束引导 + 硬约束渐进校正）

**输入**：`marginals_long.parquet`（宏观目标）+ `buildings.parquet`（空间锚定）+ `rules.yaml`（可行域）  
**输出**：`synthetic/*.parquet`

空间锚定的关键口径（Scheme B）：**扩散模型负责学“宏观地理单元内的属性联合结构”，建筑落位由显式分配模块完成**。  
动机：PUMS 不提供可训练的人-建筑配对；训练期人为配对会引入虚假关联，且难以向评审解释“学到了什么”。相反，把空间分配显式化可审查、可参数化，也便于做消融与敏感性分析。

v0 两阶段（先跑通闭环，再逐步细化到 tract/BG）：

1) **属性生成**：在 tract/BG/PUMA 等宏观条件下采样  
   \[
   x \sim P(\text{attrs}\mid \text{macro\_cond})
   \]
2) **建筑分配**：在同一空间单元内，把已生成的人口分配到建筑  
   \[
   \text{bldg\_id} \leftarrow f(\text{attrs}, \{\text{buildings in group}\}; \text{method})
   \]
   - **经济条件（可选但强）**：Wayne County parcel assessment → `price_per_sqft` → 分位数 `price_tier(Q1..Q5)`，用于 `income_price_match` 分配策略

采样时的“两个力”：
- **软约束引导**：把生成总体的统计偏差写成可优化的目标，并在采样过程中持续把生成分布拉回目标统计附近（实现上留接口，v0 可先用“分批采样→评估→调强度”的简化版本）。
- **硬约束渐进校正**：对违反规则的少量字段做最小改动（投影/局部重采样），在迭代过程中反复执行，避免“生成后统一过滤”。

### Stage E：三层验证（统计/空间/时间）

**输入**：`synthetic/*.parquet` + 对应对照数据（普查/POI/夜光/轨迹等）  
**输出**：`metrics/*.json` + `metrics/figures/*`

验证输出必须支持“误差归因”：
- 统计层偏差 → 回溯软约束/边际口径
- 空间层偏差 → 回溯建筑特征/POI 锚定
- 时间层偏差 → 回溯时段条件/动态模块

---

## 5. 代码目录结构（目标形态，v0 起步）

> 现有入口在 `src/synthpop/cli.py`；新增模块应保持“可读、可替换、低耦合”。

```text
src/synthpop/
  cli.py
  paths.py
  config.py                   # 配置读取与校验（轻量）

  pipeline/
    detroit_v0.py              # Detroit v0: orchestration（只负责串联，不写算法）

  data/
    geo.py                     # TIGER: 读取/转换/裁剪（后续实现）
    census.py                  # ACS/PUMS: 下载后的解析与口径统一（后续实现）
    buildings_gba.py           # GBA LoD1: 裁剪/投影/面积/高度特征（后续实现）
    poi_safegraph.py           # SafeGraph: 读取/抽取 Detroit 子集（后续实现）

  features/
    condition_vectors.py       # 区域/建筑/时段条件向量构建

  model/
    diffusion_tabular.py       # TabDDPM-style：混合类型表格扩散（Gaussian DDPM on encoded vectors）

  constraints/
    soft_guidance.py           # 软约束引导接口（先定义抽象）
    hard_rules.py              # 规则检查（结构性零/层级一致性）
    projection.py              # 渐进校正：投影/局部重采样

  validation/
    stats.py                   # 边际 + 二阶关联 + 违规率
    spatial.py                 # POI/用地/夜光一致性等
    temporal.py                # 日/周周期（若启用动态）

  spatial/
    assign_buildings.py        # BG/tract → building 抽样（可作为生成过程中的空间条件）
```

> 说明：v0 先把“接口与数据契约”固定，算法实现按优先级逐步补齐；不要一上来把全部下载、清洗、训练、评估都写成一个巨大脚本。

---

## 6. CLI 入口（建议）

保持 `argparse` 风格（与现有 `src/synthpop/cli.py` 一致），避免引入额外框架。

建议新增（按优先级）：
- `python -m src.synthpop detroit prepare`：raw→processed（可拆子命令）
- `python -m src.synthpop detroit features`：processed→features
- `python -m src.synthpop detroit train`：训练并写 run 目录
- `python -m src.synthpop detroit sample`：受控采样（软/硬约束）
- `python -m src.synthpop detroit validate`：三层验证与诊断

---

## 7. 待 PI 定稿的 6 个问题（避免后期返工）

1) Detroit 研究区：city boundary 还是 Wayne County？（默认：city boundary）  
2) v0 是否必须做 household–person 两层级联？（默认：person-only；v1 再补 household）  
3) 时间维度：先静态还是直接日/夜两时段？（默认：静态；v0.5 加日/夜）  
4) 控制量层级：tract 主控 + BG 诊断是否接受？（默认：接受）  
5) 属性集合 v0 最小集：年龄/性别/收入/就业状态/家庭规模 是否足够？（默认：足够；后续再加种族/教育/车辆）  
6) 空间锚定：building + POI 是否足够？（默认：足够；parcel optional）

# Synthetic City 架构说明

> 这份文档回答两个问题：  
> 1. 这个仓库在研究上到底解决什么问题  
> 2. 代码目录是如何围绕这些问题组织起来的

---

## 1. 核心主线

这个项目不是“从统计表直接生成点坐标”的单步系统，而是一条**分层推断链**。  
我们把合成人口与空间化拆成三个不能混在一起的任务：

1. **Phase 1：恢复人口联合结构**  
   回答“一个 PUMA 里住着什么样的人”
2. **Phase 2：分解到 tract / CBG**  
   回答“这些不同类型的人在 PUMA 内部应如何分布到小区域”
3. **Phase 3：显式空间化**  
   回答“人和 worker 在 tract / CBG 内最终落到哪里”

这三个阶段分别对应三种不同的数据角色、不同的约束来源、以及不同的验证方式。  
如果把它们揉成一步，最后既很难解释，也很难诊断误差。

---

## 2. 研究问题到代码模块的映射

### 2.1 Phase 1：人口联合结构恢复

这一层不直接处理点位，而是学习：

\[
P(\text{attrs} \mid \text{PUMA})
\]

也就是在宏观单元内恢复人口属性之间的联合结构，而不是只拼边际。

**主要模块**

- `src/synthpop/model/`
- `src/synthpop/alignment/`
- `src/synthpop/features/condition_vectors.py`

**典型脚本**

- `tools/train_us_puma_5var_diffusion.py`
- `tools/train_us_puma_3var_diffusion.py`
- `tools/poc_tabddpm_pums.py`
- `tools/poc_tabddpm_acs_supervised_b01001.py`

**说明**

- 这一层偏“人口画像”
- 不负责 tract/CBG 分配
- 不负责 home/work 显式落位

---

### 2.2 Phase 2：PUMA -> tract / CBG 的小区域分解

这一层回答的是：

\[
P(\text{small-area} \mid \text{attrs}, \text{PUMA})
\]

它本质上是一个**受约束的空间分解问题**，不是 person-level location prediction。

**主要模块**

- `src/synthpop/spatial/puma_to_small_area.py`
- `src/synthpop/constraints/soft_guidance.py`
- `src/synthpop/constraints/hard_rules.py`
- `src/synthpop/constraints/projection.py`
- `src/synthpop/data/acs_crosstab.py`

**典型脚本**

- `tools/exp_phase2_puma_to_small_area.py`
- `tools/exp5_tract_postalign.py`

**说明**

- `ACS / Census` 在这一层是主锚点
- `mobility` 在当前最终主线里不再承担核心 demographic allocation 约束
- 它更多用于 residual heterogeneity 探索或外部验证

---

### 2.3 Phase 2.5：household synthesis

这一层把 person-level synthetic population 组织成 household，使 `home` 具备正确语义。

**主要模块**

- `src/synthpop/spatial/tract_householding.py`

**典型脚本**

- `tools/exp_phase25_synthesize_households.py`

**说明**

- 这一步不是装饰
- 它解决的是 home assignment 的语义问题：`home` 应该首先是 household-level residence，而不是 individual scatter

---

### 2.4 Phase 3：显式 home / work 空间化

这一层把 tract / CBG 内的人口质量真正落到显式位置上。

#### Home

`home` 的当前主线是：

- household-aware
- road-constrained
- tract/CBG 内显式候选点分配

**主要模块**

- `src/synthpop/spatial/road_location_allocation.py`

#### Work

`work` 的当前主线不是“在 home tract 里撒白天点”，而是：

1. `worker -> destination tract`
2. `destination tract -> explicit work point`

**主要模块**

- `src/synthpop/spatial/work_destination_allocation.py`
- `src/synthpop/data/lodes.py`

**典型脚本**

- `tools/exp_phase3b_assign_work_destinations.py`
- `tools/exp_phase3_validate_mobility_anchor.py`

---

## 3. 数据源的角色分工

这个仓库里最重要的原则之一，是**不同数据源不承担相同角色**。

| 数据源 | 在当前主线中的角色 |
|---|---|
| `PUMS` | 提供人口微观属性与联合结构种子 |
| `ACS / Census` | 提供 tract / CBG 的 population anchor 与 household constraints |
| `LODES` | 提供 work destination mass 与 OD skeleton |
| `TIGER roads / MTFCC` | 提供显式 home/work 候选支持集 |
| `mobility anchor` | 提供 home/work 的外部验证与局部 pattern 检查 |
| `POI / visit products` | 可做 refinement 或诊断，但不是当前主线骨架 |

关键边界：

- `ACS` 负责 small-area totals 和结构锚点
- `LODES` 负责 work destination 骨架
- `roads` 负责“可以落在哪里”
- `mobility` 主要负责“落出来的 pattern 像不像”

---

## 4. 当前仓库的代码拓扑

### 4.1 `src/`

这里放可复用模块，而不是一次性脚本。

```text
src/synthpop/
  alignment/     条件对齐、distribution matching、contrastive utilities
  constraints/   软约束、硬规则、投影修正
  data/          ACS / Census / LODES / mobility / POI 等数据读取
  detroit/       Detroit 路径、常量、stage helpers
  features/      条件向量与特征构造
  model/         diffusion 与 joint generation 模块
  pipeline/      可复用编排入口
  spatial/       small-area expansion、householding、home/work allocation
  validation/    统计、空间、temporal、mobility-anchor validation
```

这部分应尽量保持：

- 单一职责
- 可测试
- 不依赖某一次 run 的目录结构

### 4.2 `tools/`

这里放真正驱动实验的入口脚本。  
仓库当前是**research-first** 结构，所以大多数可执行入口都在这里。

大致可分为：

- `poc_*`：早期方法 PoC
- `exp_phase2_* / exp_phase25_* / exp_phase3*`：阶段性实验
- `build_* / prepare_*`：数据准备和中间表构建
- `viz_* / make_fig*`：验证图与论文图生成
- `run_*.sh`：批量实验调度

### 4.3 `docs/`

这里不是 API 文档，而是**研究文档和运行文档**。

主要包括：

- 方法草稿
- 发现总结
- Detroit 数据与代码蓝图
- 各次 pilot 的结论记录
- figure / framework 设计说明

### 4.4 本地写作工作区

论文草稿、proposal、以及临时写作材料通常保存在开发者本地工作区中。  
这些内容可以与仓库同级共存，但**不属于公开代码仓库的核心契约**，默认不作为版本化主内容维护。

---

## 5. 当前最重要的结果状态

如果只用一句话概括当前项目状态：

> `home` 已经是强结果；`work tract + commute shape` 基本成立；`OD` 仍是开放问题。

更具体一点：

- `home`
  - tract-level share 有稳定 external validation
  - within-tract BG rank 也有支撑
  - 已经具备正文图和正文叙事
- `work`
  - destination tract pattern 与 commute shape 已明显优于早期版本
  - 但还不能像 `home` 一样说“完全站住”
- `OD`
  - 现在最需要的是更强的 pairing structure 或更干净的 reference
  - 目前尚未形成可主张的强结果

---

## 6. 推荐阅读顺序

如果要快速理解整个仓库，推荐按这个顺序读：

1. `README.md`  
   先看项目概况和当前主线
2. `docs/synthpop_architecture.md`  
   看研究问题与代码模块如何对应
3. `docs/detroit_code_data_structure.md`  
   看 Detroit 数据、目录和 run 结构
4. `docs/phase3_small_area_to_road_constrained_locations_method.md`  
   看显式 spatialization 逻辑
5. `docs/phase3_work_destination_detroit_2026-03-29.md`  
   看 work 结果当前站到什么程度
6. `docs/phase3_mobility_anchor_validation_detroit_2026-03-29.md`  
   看 validation 口径

如果关心更早的 distribution-level diffusion 路线，再看：

- `docs/methods.md`
- `docs/findings.md`

---

## 7. 仓库维护原则

这个仓库不追求“像一个产品那样整洁”，但要追求两件事：

1. **代码与研究问题保持对应**
2. **大数据与 run 产物不污染 git**

因此：

- 原始数据、licensed data、大 run 目录都应放外部数据盘
- `outputs/` 用于研究产物，不作为长期版本控制对象
- `figures/` 只保留当前在用或可复现的重要图
- `tools/` 可以很多，但要命名清楚、职责清楚

这比“把所有东西都塞进一个 package”更符合当前项目阶段。

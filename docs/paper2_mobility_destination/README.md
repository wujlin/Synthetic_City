# Paper 2: Coarse-to-Fine Work Destination

## 目录作用

这个目录单独服务于第二篇文章的工作线。  
主题不是继续提升 demographic consistency，而是补上当前 synthetic population pipeline 中缺失的 **home -> work destination structure**。

当前这条线的核心判断是：

- 第一篇文章已经基本回答了 `谁住在哪里`
- 但还没有真正回答 `这些人去哪里工作`
- `mobility` 的角色不是替代 `ACS / Census / LODES`
- 它更适合在 **覆盖足够稳定的空间尺度** 上提供行为约束

## 当前问题定义

新论文的核心对象不是显式点位本身，而是：

\[
P(g_w \mid g_h, x)
\]

其中：

- `g_h`：home tract / CBG
- `g_w`：work destination tract / CBG
- `x`：worker type / household context / origin context

## 当前方法立场

这篇文章不把 `mobility` 当作：

- 个体级 SES 真值
- tract 级 demographic hard constraint
- 对 `ACS / LODES` 的简单替代品
- tract-OD 尺度上的 dense supervision

而是把它当作：

- origin / destination activity pattern 的行为线索
- coarse destination state 上的偏好约束
- `gravity / LODES` 解释不了的 residual heterogeneity

## 当前主方法方向

当前不再把 tract-OD mobility residual 视为最终主方法，而把它视为一个 `proof-of-signal` 诊断。

真正准备推进的子刊级主线是：

\[
P(g_w \mid g_h, x)
=
P(z_w \mid g_h, x)
\times
P(g_w \mid g_h, x, z_w)
\]

其中 `z_w` 是一个 coarse destination state，例如：

- `work county`
- `job center`
- `destination regime`

这意味着：

- `LODES / WAC / gravity` 提供结构完整的 destination skeleton
- `mobility` 只在 coverage 足够稳定的 coarse state 上进入模型
- `tract within center` 的细化继续由结构数据完成

## 当前优先怀疑的主状态

当前优先怀疑最合适的上层状态是 `job center`，因为它：

- 比 `county` 更细
- 比 `tract` 更稳
- 更接近真实通勤决策中的就业聚集地

## 当前首轮目标

首轮不是直接追求一个复杂模型，而是先把实验线按信息增量压实，并用诊断结果决定主模型该落在哪一层：

1. 复现实验缺口：当前 work 失败来自 destination layer 缺失
2. 建立透明骨架：`distance / gravity / LODES`
3. 证明 mobility 确实携带 OD 增量信息
4. 判断 `tract / center / county / top-K candidate set` 中，哪一层最适合进入主模型

## 目录内容

- `paper_outline.md`
  - 论文主线与章节骨架
- `experiment_roadmap.md`
  - 实验设计、baseline ladder、coverage diagnostics、阶段目标
- `detroit_runbook_v1.md`
  - Detroit 首轮实验的可执行入口与运行顺序
- `lab_log.md`
  - 连续日志，记录判断、变更、风险与下一步

## 备注

这个目录是第二篇文章的长期工作区。  
后续所有与 `coarse-to-fine destination model` 有关的设计、结果与判断，都优先沉淀到这里，而不是散落在临时对话中。

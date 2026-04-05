# Detroit Work Destination Pilot (2026-03-29)

## Question

`phase3` 里原来的 `work` 是否真的是工作地，还是只是 `home tract` 内的白天 support point？

## Answer

原来的 `work` 不是完整的工作地对象。问题不在道路点本身，而在于它缺少 `worker -> destination tract` 这一步。  
一旦把 Detroit 的 `LODES tract OD` 接进来，`work` 的定义就从“居住 tract 内的白天点”变成了“先抽 destination tract，再在 destination tract 内落显式点”，并且外部 mobility 验证随之显著改善。

## What Changed

新链路变成：

1. `home tract -> work destination tract`
2. `work destination tract -> road-constrained work point`

实现上：

- 用 `LODES OD` 构造 Detroit study-internal `tract -> tract` job flow
- 给每个 synthetic worker 分配 `work_tract_geoid`
- `phase3` 的 `work` 候选池改为按 `work_tract_geoid` 取，而不是按 `tract_geoid`

## Key Evidence

### 1. Destination tract 已经真正被引入

在 Detroit `phase3b` 里：

- `work_eligible = 2,696,819`
- `work_destination_assigned = 2,696,819`
- `work_destination_unassigned = 0`
- `same_tract_share_among_assigned = 0.0324`

这说明 synthetic workers 不再被硬锁在 home tract。

### 2. 显式空间层仍然闭合

在新的 Detroit `phase3` 里：

- `home_unassigned = 0`
- `work_unassigned = 0`
- `home_zero_fallback_success = true`
- `work_zero_fallback_success = true`
- `overall_zero_fallback_success = true`

同时验证内部一致性保持为：

- `work candidate_ref_hit_rate = 1.0`
- `work group_match_rate = 1.0`
- `work stage_match_rate = 1.0`

### 3. Mobility work validation 从“对象错位”变成“部分可比”

旧版 Detroit（没有 destination tract）：

- `work tract Spearman = 0.195`
- `work tract cosine = 0.570`
- `work OD Spearman = -0.778`
- `synthetic commute median = 1.21 km`

新版 Detroit v0（只接入 destination tract，不加距离/分段）：

- `work tract Spearman = 0.578`
- `work tract cosine = 0.812`
- `work tract TVD = 0.328`
- `work OD Spearman = 0.0248`
- `work OD cosine = 0.204`
- `synthetic commute median = 18.40 km`

新版 Detroit v1（`LODES OD + distance deterrence + earnings-segment prior`）：

- `work tract Spearman = 0.558`
- `work tract cosine = 0.839`
- `work tract TVD = 0.302`
- `work OD Spearman = 0.0277`
- `work OD cosine = 0.229`
- `work OD TVD = 0.910`
- `synthetic commute median = 8.25 km`
- `mobility commute median = 9.33 km`

这里最关键的变化不是某一个指标涨了多少，而是：

`work tract` 已经明显改善，而且通勤距离终于回到了 Detroit mobility 的合理尺度；但 `OD` 仍然很弱。

## Interpretation

这组结果说明两件事：

1. 原来的坏结果主要来自**对象定义错误**，不是道路点生成失败。  
   只要补上 `destination tract`，`work tract` 对外部 mobility 的一致性就会明显提升。

2. 但 `destination tract` 只解决了第一层。  
   `distance deterrence + earnings-segment prior` 已经把通勤尺度从过长状态拉回来了，但 `OD` 仍然弱，说明现在的 `worker -> destination tract` 还只是一个轻量条件分配器，没有把更细的 worker type、industry、job accessibility 或 mobility day-anchor 的结构信息接进去。

## Typed-OD Follow-Up

进一步把 `worker type` 更正式地接进 Detroit `OD` 之后，一个边界变得很清楚：

- 直接用 `SA/SE` 的 **raw count** 作为 type-conditioned reweight 会过强。  
  这会把大流量边重复放大，导致 synthetic commute 过度本地化，`same_tract_share` 从约 `0.11` 推高到约 `0.20`，而 `work tract` 与 `OD` 验证都明显变差。

- 把同样的 `SA/SE` 改成相对 `S000` 的 **segment share** 后，行为恢复正常。  
  这版 typed-OD 的通勤分布与上一版 best run 基本对齐，`work tract` 和 `work OD` 指标也回到几乎同一水平。

这说明：

1. `worker type` 的正确进入方式，不是重复乘总量，而是在总量骨架内部改变相对偏好。  
2. 仅靠 `age + earnings` 两个边际分段，尚不足以显著超过当前 best run。  
3. 下一步真正值得接入的，不再是更多同类 reweight，而是更强的 destination structure，例如 industry、job accessibility，或修复后再接入 `WAC CA/CE` residual。

## WAC Residual Follow-Up

把 Detroit `WAC CA/CE` 真正接回 `tract_od` 之后，一个边界也变清楚了：

- 修复 `tract_wac` 管线本身是必要的。  
  之前的问题不是 Detroit 没有 `WAC`，而是 `tract_geoid` 在聚合时被错误转成了数值型，导致 study tract 过滤全部失配。

- 但单独加入 `CA/CE` destination residual，并没有带来净增益。  
  在 `type-OD share + destination residual` 版本里：
  - `work tract Spearman = 0.5566`
  - `work tract cosine = 0.8307`
  - `work OD Spearman = 0.0236`
  - `work OD cosine = 0.2238`
  - `synthetic commute median = 8.50 km`

相对当前 best run，这版更多是在补 destination 描述，而不是在提升 Detroit 的外部 mobility 一致性。  
这说明 `CA/CE` residual 是一个合理但偏弱的 destination-side prior，它本身不足以突破当前 `OD` 瓶颈。

## Accessibility Follow-Up

真正更有信息量的是 `job accessibility`。

在保持当前 best run 主体不变的前提下，只额外加入：

- `destination earnings prior`
- `destination accessibility prior`

其中 accessibility 定义为 tract-level `gravity access to jobs`，即用 destination tract 的 job mass 和 tract centroid 间距离构造的引力可达性。

这版结果是：

- `same_tract_share_among_assigned = 0.1002`
- `work tract Spearman = 0.5706`
- `work tract cosine = 0.8367`
- `work tract TVD = 0.3046`
- `work OD Spearman = 0.0282`
- `work OD cosine = 0.2296`
- `work OD TVD = 0.9131`
- `synthetic commute median = 8.72 km`
- `commute cosine = 0.9766`
- `commute TVD = 0.1315`

这里最重要的不是所有指标都一起变好，而是它揭示了 destination structure 的作用方式：

1. `accessibility` 比 `WAC residual` 更能推高 tract-level ranking，一定程度上改善了 `work tract` 的排序一致性。  
2. `accessibility` 对通勤距离分布的帮助更明显，`cosine` 和 `TVD` 都优于当前 best run。  
3. 但它仍然没有实质性解决 `OD`，说明 Detroit 的剩余误差已经不是单一 destination attractiveness 能解释的。

所以现在最稳的判断是：

- `WAC residual` 值得修，但不是主突破口
- `job accessibility` 比 `WAC residual` 更有效
- `OD` 的下一步，应该是把 accessibility 保留为结构轴，再继续补更细的 `N_{o,k,d}` 条件化，而不是回头继续调道路点

## Balanced N_{o,k,d} Follow-Up

进一步把 `worker -> destination tract` 从独立抽样改成更正式的 `N_{o,k,d}` 之后，Detroit 的边界又清楚了一层。

这里的 `N_{o,k,d}` 版本不是简单的 row-wise sampling，而是对每个 origin tract 单独求一个带硬约束的分配矩阵：

- 行约束：origin 内每个 worker type 的 synthetic 人数
- 列约束：origin 对各 destination tract 的 `OD` 总量（按 synthetic worker 总数缩放）
- 先验：`distance + od age/earn share + destination earnings share + accessibility`

这个版本最直接的变化是：

- `same_tract_share_among_assigned = 0.0323`

也就是说，它确实把分配重新拉回到了更接近 `LODES OD` 骨架的状态，而不是让独立抽样再次把流量本地化。

但最终外部 mobility 验证说明：

- `work tract Spearman = 0.5776`
- `work tract cosine = 0.8121`
- `work tract TVD = 0.3284`
- `work OD Spearman = 0.0358`
- `work OD cosine = 0.2069`
- `work OD TVD = 0.9543`
- `synthetic commute median = 18.39 km`
- `commute cosine = 0.9021`
- `commute TVD = 0.2505`

这组结果揭示的不是“更正式的 `N_{o,k,d}` 没用”，而是：

1. 仅靠把 row/column constraints 做硬，并不会自动得到更好的外部 `OD`。  
   它会更忠于 `LODES` 骨架，但也更容易把通勤尺度重新拉长。

2. 当前 Detroit 里，`OD` 的问题已经不只是“是否保住 margins”。  
   真正的误差来自：  
   `LODES aggregate skeleton`、`synthetic worker composition`、以及 `mobility day-anchor reality` 之间并不完全同构。

3. 因此，`balanced N_{o,k,d}` 更像一个重要的诊断实验，而不是当前最优解。  
   它证明了：  
   `OD` 的剩余问题不是独立抽样造成的数值噪声，而是 destination choice structure 本身还不够对。

所以现在最稳的结论变成了：

- `accessibility` 是值得保留的结构轴
- `balanced N_{o,k,d}` 证明 margins 不是唯一瓶颈
- 下一步如果还要继续提高 `OD`，就不能只在分配器层做更硬的约束，而要补更有辨识力的 destination structure 或 worker-side latent structure

## Engineering Note

这条 Detroit `phase3` 线已经不再被逐人循环卡住。  
在完全相同的 `persons_with_worktract` 输入上：

- `n_jobs=1` 时，`candidate_build_seconds = 67.69s`
- `n_jobs=48` 时，`candidate_build_seconds = 22.04s`

候选点生成这一步单独约 `3.07x` 加速；整段 `phase3 core runtime` 从 `132.03s` 降到 `86.91s`，约 `1.52x`。  
更重要的是，并行前后 `home_candidate_id / work_candidate_id / source_stage` 的 person-level 结果完全一致，说明这次提速没有改变方法语义，只是把 tract 级独立 clip/interpolate/legalize 吃满了 WSA 的 CPU 并行度。

## Take-Home Message

Detroit 的 `work` 线现在已经跨过了最关键的一步：  
它不再是 `home tract` 内的伪 daytime point，而是一个真正带 `destination tract` 的两阶段对象。

但这条线还没有终结。  
当前结果表明：

- `home` 已经是可交付的空间锚点
- `work tract` 已经进入可解释、可比较的阶段
- `work OD` 仍然是下一阶段真正需要建模的核心问题

## Job-Center + County Friction Follow-Up

在 accessibility 版本的基础上，我又补了两个更有辨识力的 destination 结构轴：

- `job-center accessibility`
- `same-county friction`

这里的 `job-center accessibility` 不是一般的 job mass gravity，而是先从 Detroit study tract 中筛出高就业中心，再计算 tract 到这些中心的引力可达性。  
`same-county friction` 则用来表达一个更弱但常见的 destination 规律：在相同距离下，跨 county 通勤往往比同 county 通勤更少见。

这版结果是：

- `same_tract_share_among_assigned = 0.0787`
- `work tract Spearman = 0.5801`
- `work tract cosine = 0.8246`
- `work tract TVD = 0.3215`
- `work OD Spearman = 0.0243`
- `work OD cosine = 0.2212`
- `work OD TVD = 0.9226`
- `synthetic commute median = 9.87 km`
- `commute cosine = 0.9858`
- `commute TVD = 0.0951`

这组结果最重要的地方不在于“是否全都更好”，而在于它把 Detroit 里剩余误差的结构进一步揭开了：

1. `job-center + county` 确实在改变 destination choice。  
   `same_tract_share` 从 accessibility 版的约 `0.100` 进一步降到约 `0.079`，而 `work tract Spearman` 也继续升到当前已测试版本中的最高值。

2. 但它改善的是 tract-level ordering 和 commute shape，不是 `OD` 本身。  
   `commute cosine/TVD` 明显优于当前 best run，说明 synthetic workers 去得更像 mobility 里的真实通勤尺度；  
   但 `work OD Spearman/cosine/TVD` 并没有同步提高，反而略弱。

3. 这说明 Detroit 当前的 `OD` 瓶颈不是“destination 结构太少”这么简单。  
   更强的 destination structure 已经能把 worker 推向更像就业中心、也更像真实通勤半径的地方；  
   但具体到 `origin -> destination` 配对时，仍然缺一个更细的 conditional mechanism。

所以现在最稳的判断变成了：

- `job-center accessibility` 是有信息量的，而且比普通 accessibility 更强
- `same-county friction` 至少在组合版里不是灾难性的，但也没有把 `OD` 真正拉起来
- Detroit 的下一步，不能只靠增加 destination attractiveness 项，而要继续拆清 `N_{o,k,d}` 里到底是哪一部分条件结构没被表达出来

## Type-Specific Utility Follow-Up

在 `job-center + county friction` 的基础上，我又把第一层更正式的 `N_{o,k,d}` 条件化接进来了：

- 不再让所有 worker 共享同一套 utility coefficients
- 而是让 `distance / destination access / job-center access / same-county friction`
  的系数随 worker earnings segment 变化

这一版没有上黑箱学习，而是先用可解释的 segment-specific multipliers 做 Detroit pilot：

- `CE01` 更偏近距离、同 county、弱中心
- `CE03` 更偏远距离、弱同 county、强中心
- `CE02` 作为中性 baseline

这版结果是：

- `same_tract_share_among_assigned = 0.0781`
- `work tract Spearman = 0.5810`
- `work tract cosine = 0.8211`
- `work tract TVD = 0.3237`
- `work OD Spearman = 0.0232`
- `work OD cosine = 0.2214`
- `work OD TVD = 0.9233`
- `synthetic commute median = 9.97 km`
- `commute cosine = 0.9872`
- `commute TVD = 0.0874`

它和上一版 `job-center + county` 的对比非常说明问题：

1. type-specific utility 确实在发挥作用。  
   `same_tract_share` 继续下降，`commute cosine/TVD` 继续改善，说明 worker type 已经开始影响 destination choice。

2. 但它改善的仍然主要是 commute shape，而不是 `OD`。  
   `work tract Spearman` 只小幅升高，`work OD` 几乎没有改善，`TVD` 还略差。

3. 这说明我们已经把问题进一步逼近到一个更窄的位置：  
   仅仅让不同收入段对同一个 destination profile 有不同响应，还不足以恢复 Detroit 的 `origin -> destination` 配对结构。

所以这版实验的真正价值不是“又得到一个更好的数”，而是进一步缩小了误差来源：

- `worker type` 的确重要
- 但当前缺的不是“有没有 type signal”
- 而是更细的 `origin context × worker type × destination regime` 交互

换句话说，下一步如果还要继续做 `N_{o,k,d}`，最可能有效的不是继续手工加一层 multipliers，而是把中间的 destination regime 显式建出来，例如 county / subcenter / job-center 的层级选择。

## Hierarchical County Follow-Up

于是我把下一步真正做成了一个最小的 hierarchy：

1. 先选 `destination county`
2. 再在 county 内选 tract

第一版没有上复杂的多层 latent regime，而是先实现一个可解释的 `hierarchical_county`：

- county stage 用 county-aggregated `OD mass + distance + access + center + same-county`
- tract stage 在已选 county 内再做 tract 选择
- worker-side 仍保留上一版的 earnings-conditioned utility coefficients

这版结果是：

- `same_tract_share_among_assigned = 0.0764`
- `work tract Spearman = 0.5836`
- `work tract cosine = 0.8232`
- `work tract TVD = 0.3222`
- `work OD Spearman = 0.0238`
- `work OD cosine = 0.2192`
- `work OD TVD = 0.9239`
- `synthetic commute median = 9.97 km`
- `commute cosine = 0.9868`
- `commute TVD = 0.0894`

这组结果把判断又推进了一小步：

1. hierarchy 确实不是空转。  
   `same_tract_share` 继续下降，`work tract Spearman` 继续升到当前所有 tested variants 里的最高值。

2. 但 hierarchy 仍然没有把 `OD` 拉起来。  
   `work OD` 依旧停在很低的水平，而且并不优于平面 `center+county` 版。

3. 这意味着 Detroit 里剩余的 `OD` 误差，不只是“缺一个中间层 county choice”。  
   county 这个层级是有信息量的，但它还不够细。  
   如果继续走 hierarchical 路线，下一步更像样的对象应该是：
   `county -> subcenter / job-center -> tract`

所以这版 hierarchy 的真正价值在于：

- 它验证了“中间层 regime choice”这条路是有方向感的
- 但也同时说明，`county` 不是足够细的 regime
- 下一步若继续做 hierarchical OD，最合理的是更细的 job-center / subcenter hierarchy，而不是仅停在 county

## Hierarchical County-Center Follow-Up

于是我又把 hierarchy 往前推进了一层：

1. 先选 `destination county`
2. 再在 county 内选 `job center / subcenter`
3. 最后在 center 内选 tract

这一版使用的是显式 `work_center_geoid`，它来自 Detroit tract-level `WAC + centroid` 的 center membership 预处理；  
worker-side 仍保留上一版的 earnings-conditioned utility coefficients。  
也就是说，这次实验不是再加一个平面 prior，而是在已有 county hierarchy 上，测试更细的 `county -> center -> tract` 是否能真正改善 `OD`。

先说内部状态：

- `assignment_mode = hierarchical_county_center`
- `work_destination_assigned = 2,696,819`
- `work_destination_unassigned = 0`
- `same_tract_share_among_assigned = 0.0761`

显式落点层也仍然闭合：

- `home_unassigned = 0`
- `work_unassigned = 0`
- `overall_zero_fallback_success = true`
- `home_within_group_polygon_rate = 0.99708`
- `work_within_group_polygon_rate = 0.99448`

但真正关键的是 Detroit mobility validation：

- `work tract Spearman = 0.5846`
- `work tract cosine = 0.8226`
- `work tract TVD = 0.3226`
- `work OD Spearman = 0.0234`
- `work OD cosine = 0.2189`
- `work OD TVD = 0.9239`
- `synthetic commute median = 9.98 km`
- `commute cosine = 0.9865`
- `commute TVD = 0.0904`

和上一版 `hierarchical_county` 对比，这个结果很说明问题：

1. `county -> center -> tract` 只带来了极小的 tract-level gain。  
   `work tract Spearman` 从 `0.5836` 微升到 `0.5846`，`same_tract_share` 也略降。  
   这说明更细的 center regime 不是完全无信息量。

2. 但它没有把 `OD` 真正推起来。  
   `work OD Spearman`、`cosine` 和 `TVD` 都没有改善，反而略差；  
   `commute` 相关指标也没有继续变好。

3. 这意味着 hierarchy 这条线到这里已经基本判明：  
   `county` 太粗，`center` 更细，但仅靠显式 destination hierarchy 仍然不足以恢复 Detroit 的 `origin -> destination` 配对结构。

所以这版实验最重要的结论不是“又多了一层 hierarchy”，而是：

- `job-center / subcenter hierarchy` 不是完全无效
- 但它带来的增益已经小到不足以称为 `OD` 的有效解决
- Detroit 当前剩余的 `OD` 误差，已经不像是 destination regime 还能单独解决的问题

换句话说，`hierarchical OD` 这条线到这里已经完成了它最重要的研究任务：

- 证明了 road point 不是主瓶颈
- 证明了 destination structure 和 hierarchy 都有信息量
- 同时也证明了：**仅靠 destination-side hierarchy，仍然不足以解决 `N_{o,k,d}`**

如果后面还要继续推进 `OD`，更值得怀疑的就不再是“destination hierarchy 还不够细”，而是：

- worker-side latent job type / industry
- home-to-work activity chain 的未观测约束
- 或更强的 `origin context × worker type × destination regime` 联合机制

所以截至这一轮，最稳的判断已经可以明确写出来：

**Detroit 的 `OD` 还没有被有效解决；而 `hierarchical_county_center` 的结果表明，继续单独深挖 hierarchy 的边际收益已经很低。**

## Home-Center Context Follow-Up

基于前面的判断，我又做了一次更直接的 `origin context × worker type × destination regime` 判别实验。

这次没有再改 hierarchy 层级，而是在现有 `hierarchical_county_center` 的基础上，给 utility 显式加入一个新的 regime 信号：

- `same_home_center = 1{work_center_geoid == home_center_geoid}`

直观上，这个项测试的是：

- 如果一个 worker 的 home tract 已经被归到某个 home-side center，
- 那么 destination tract 所属的 work center 是否更倾向于落在同一个 center regime 里。

同时，这个信号继续允许 earnings-conditioned coefficient：

- `same_home_center_weight = 0.25`
- `same_home_center_earn_multiplier_map = {CE01: 1.2, CE02: 1.0, CE03: 0.9}`

因此，这一版不是再给 destination 叠一个平面 mass prior，而是在问一个更具体的问题：

**显式的 origin-side center context，是否能真正改变 Detroit 的 destination choice。**

先看内部状态：

- `use_same_home_center_prior = true`
- `work_destination_assigned = 2,696,819`
- `work_destination_unassigned = 0`
- `same_tract_share_among_assigned = 0.0761`

显式落点层仍然闭合：

- `home_unassigned = 0`
- `work_unassigned = 0`
- `overall_zero_fallback_success = true`

但最关键的是，这一版的外部 Detroit mobility validation 与上一版 `hierarchical_county_center` **完全相同**：

- `work tract Spearman = 0.5846`
- `work tract cosine = 0.8226`
- `work tract TVD = 0.3226`
- `work OD Spearman = 0.0234`
- `work OD cosine = 0.2189`
- `work OD TVD = 0.9239`
- `synthetic commute median = 9.98 km`
- `commute cosine = 0.9865`
- `commute TVD = 0.0904`

这组结果的意义很强，因为它不是“小幅无增益”，而是：

1. `same_home_center` 这类单独的 origin-context regime bonus 没有改变实际的 destination choice。  
   指标与 `hierarchical_county_center` 完全一致，说明这类额外项在当前 Detroit 设定下没有进入有效判别链。

2. 问题因此进一步收缩。  
   这次实验已经排除了一个看起来很合理的解释：  
   `OD` 持续偏弱，并不是因为我们缺了一个简单的 origin-side center context。

3. 这也意味着，下一步若继续推进 `OD`，就不该再沿着“再加一个 regime bonus”这条路打转。  
   更值得怀疑的，已经是更强的联合选择结构本身，而不是单个额外 prior。

所以这轮 follow-up 给出的最稳判断是：

**Detroit 的 `OD` 误差，不是靠再加入一个显式 origin-context bonus 就能修好的。**

这也进一步支持前面的总判断：

- road point 不是主瓶颈
- destination hierarchy 已经接近边际收益上限
- 单个 origin-context regime bonus 也不是主解

如果后面还要继续推进 `OD`，更值得尝试的方向只剩两类：

- 更强的 `origin context × worker type × destination regime` 联合机制
- worker-side latent job type / industry

## Joint Regime Follow-Up

基于上面的判断，我又把一个更强的联合机制真正落成了可运行版本，而不只是继续给扁平 utility 加单个 bonus。

这一版的核心变化是把 `assignment_mode` 改成 `hierarchical_regime`：

- 先在更细的 regime 集合里做选择：
  - `same_tract`
  - `same_center`
  - `same_county`
  - `cross_county`
- 再在选中的 regime 内选 tract
- 同时保留 type-specific coefficient，让 `distance / access / center / county / same-tract / same-home-center` 对不同 worker type 的效用不同

这比前面的 `hierarchical_county_center + same_home_center bonus` 更强，因为它不再只是给某个 tract 加分，而是显式改变：

**来自同一个 origin、属于不同 worker type 的人，会先偏向哪类 destination regime。**

内部状态显示，这个机制确实在改变 destination choice：

- `work_destination_assigned = 2,696,819`
- `work_destination_unassigned = 0`
- `same_tract_share_among_assigned = 0.1581`

显式落点层也仍然闭合：

- `home_assigned = 5,736,484`
- `home_unassigned = 0`
- `work_assigned = 2,696,819`
- `work_unassigned = 0`

但 Detroit mobility validation 的结论是混合的：

- `work tract Spearman = 0.5777`
- `work tract cosine = 0.8308`
- `work tract TVD = 0.3093`
- `work OD Spearman = 0.0203`
- `work OD cosine = 0.2163`
- `work OD TVD = 0.9121`
- `synthetic commute median = 8.46 km`
- `commute cosine = 0.9914`
- `commute TVD = 0.0662`

和前面的 `hierarchical_county_center` 对比，这个结果说明了两件事：

1. 更强的 regime gate 不是空转。  
   它明显改善了 tract-level cosine/TVD，也让 commute 分布更像 mobility。

2. 但它仍然没有把 `OD` 推起来。  
   `work OD Spearman` 反而更低，`OD cosine` 也没有改善。

所以这次实验把问题收得更紧了：

**更强的联合 regime 选择能改善通勤尺度和部分 tract mass，但它仍然不足以恢复 Detroit 的 `origin -> destination` 配对结构。**

这意味着，当前剩余的 `OD` 误差不再像是“少一个 regime 层级”或“少一个 regime bonus”的问题，而更像是：

- regime 内部的 conditional choice 仍然太弱
- 或 worker-side 的 job semantics 仍然没有进入模型

## Latent Job-Family Pilot

在联合机制之外，我也沿着更高风险、但更接近真实劳动市场语义的方向做了一次 worker-side latent job type 试探。

这次没有直接做 full latent industry，而是先构造了一个更粗的 latent job-family：

- `JF_SERVICE`
- `JF_INDUSTRIAL`
- `JF_PROFESSIONAL`

它不是来自观测到的 worker-side industry label，而是用 worker 的：

- `EARN_16p_bin`
- `AGEP_bin`
- `SCHL_allpop`

做一个 coarse heuristic assignment，然后利用 destination tract 的 `WAC CNS` 聚合，构造 tract-level 的：

- `work_share_JF_SERVICE`
- `work_share_JF_INDUSTRIAL`
- `work_share_JF_PROFESSIONAL`

再把这些 share 作为 destination prior 接回 `N_{o,k,d}`。

这次 pilot 的 latent family 在 Detroit worker 中的分布是：

- `JF_PROFESSIONAL = 1,484,760`
- `JF_INDUSTRIAL = 756,591`
- `JF_SERVICE = 455,468`

显式落点层仍然闭合：

- `home_assigned = 5,736,484`
- `home_unassigned = 0`
- `work_assigned = 2,696,819`
- `work_unassigned = 0`

但最关键的外部 mobility validation 明确显示：这条线没有带来净增益，反而弱于当前 best。

- `work tract Spearman = 0.5569`
- `work tract cosine = 0.7747`
- `work tract TVD = 0.3564`
- `work OD Spearman = 0.0181`
- `work OD cosine = 0.2088`
- `work OD TVD = 0.9261`
- `synthetic commute median = 10.14 km`
- `commute cosine = 0.9843`
- `commute TVD = 0.0958`

这组结果的意义很直接：

1. worker-side latent semantics 的方向本身不是错的。  
   它至少把问题指向了更接近真实 labor market semantics 的层面。

2. 但这版 coarse latent family 还不够可识别。  
   仅靠 `age + education + earnings` 的 heuristic family，无法稳定恢复 Detroit 的 destination choice。

3. 因而，`latent industry/job type` 虽然更接近真问题，但它当前的识别风险也确实更高。  
   在没有更强 worker-side约束的情况下，它还不足以成为有效解。

所以这轮 pilot 给出的最稳判断是：

**worker-side latent job semantics 可能是对的方向，但这版 coarse heuristic family 还没有把它做成有效的 `OD` 解决方案。**

到这里，Detroit 的 `OD` 路线已经有一个比较清楚的收束：

- `destination hierarchy` 有信息量，但边际收益很低
- 单个 origin-context bonus 不是主解
- 更强的 regime gate 能改善 commute 和部分 tract mass，但仍不能解决 `OD`
- coarse latent job-family 更接近真问题，但当前识别还不够强

所以当前最诚实的结论应该是：

**Detroit 的 `OD` 仍未被有效解决；但问题结构已经被缩小到“更强的 regime 内 conditional choice”与“更可识别的 worker-side job semantics”这两个层面。**

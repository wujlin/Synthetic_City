# 联合生成架构设计：属性-空间-条件的统一扩散框架

> 版本：v0.1 (设计文档)
> 日期：2026-01-28
> 状态：架构设计阶段

## 1. 设计动机

### 1.1 Scheme B 的根本局限

当前 Scheme B（分离式）的问题不是实现细节，而是**架构层面不符合研究叙事**：

| 叙事要求 | Scheme B 实际 | 问题 |
|----------|---------------|------|
| "属性-位置联合生成" | 先生成属性，再规则分配位置 | 分离，非联合 |
| "约束内生满足" | 后处理规则 + 15% fallback | 外挂，非内生 |
| "扩散模型核心作用" | 只生成属性边际 | 未发挥联合分布学习能力 |
| "多源数据融合" | 只用 PUMS | 无条件注入机制 |

### 1.2 核心挑战

**没有真实的 (person, building) 配对数据**——无法直接监督"谁住在哪"。

但我们有：
- 人的属性分布（PUMS/ACS）
- 建筑的空间属性（位置、价格、容量、周边POI）
- 聚合层面的合理性先验（高收入→高价房、年轻人→商业区附近）

**核心思路**：用多源数据构造**概率配对**作为弱监督，让扩散模型学习**合理的联合分布**。

---

## 2. 联合生成架构（Scheme C）

### 2.1 核心表示

```
生成目标: x = (x_person, x_building)
  - x_person ∈ R^{d_p}: 人口属性向量 (AGEP, PINCP, SEX, ...)
  - x_building ∈ R^{d_b}: 建筑潜变量 (从建筑特征编码得到)

条件向量: c = (c_geo, c_spatial, c_temporal)
  - c_geo: 宏观地理条件 (PUMA/tract one-hot 或 embedding)
  - c_spatial: 空间条件 (POI密度、夜间灯光强度、土地利用类型)
  - c_temporal: 时间条件 (预留：工作日/周末、日间/夜间)
```

### 2.2 模型组件

```
┌─────────────────────────────────────────────────────────────┐
│                    Joint Diffusion Model                     │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Building   │    │    Joint     │    │   Building   │   │
│  │   Encoder    │───▶│   Denoiser   │───▶│   Decoder    │   │
│  │   E_θ(b)     │    │   ε_θ(x,t,c) │    │   D_θ(z)     │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         ▲                   ▲                   │            │
│         │                   │                   ▼            │
│    Building            Conditions          Nearest          │
│    Features            c = [c_geo,         Building         │
│    (price,             c_spatial,          Lookup           │
│     cap,               c_temporal]                          │
│     poi, ...)                                               │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2.1 Building Encoder

将建筑的异构特征映射到低维连续潜空间：

```python
class BuildingEncoder(nn.Module):
    """
    输入: 建筑特征向量 (price_tier, cap_proxy, dist_cbd, poi_density, ...)
    输出: 潜变量 z_bldg ∈ R^d_latent
    """
    def forward(self, building_features):
        # 连续特征: 标准化
        # 离散特征: embedding
        # 拼接后 MLP 编码
        return z_bldg  # shape: (n_buildings, d_latent)
```

**设计要点**：
- `d_latent` 建议 16-32 维（足够表达建筑差异，但不过度稀疏）
- Encoder 可预训练（自监督重构）或端到端联合训练

#### 2.2.2 Joint Denoiser

条件扩散模型，同时对人口属性和建筑潜变量去噪：

```python
class JointDenoiser(nn.Module):
    """
    输入: 
      - x_noisy = [x_person, z_bldg] (加噪后的联合向量)
      - t: 时间步
      - c: 条件向量
    输出:
      - ε_pred: 预测的噪声
    """
    def forward(self, x_noisy, t, c):
        # 拼接 [x_noisy, t_emb, c]
        # Transformer / MLP 架构
        return epsilon_pred
```

#### 2.2.3 Building Decoder / Lookup

采样后，将生成的 `z_bldg` 映射回真实建筑：

```python
def decode_building(z_bldg_generated, building_index):
    """
    在同一 PUMA/tract 内，找与 z_bldg_generated 最近的真实建筑
    """
    # 计算生成的潜变量与所有建筑潜变量的距离
    distances = cdist(z_bldg_generated, building_index.z_bldg)
    # 返回最近邻建筑的 bldg_id
    return building_index.bldg_id[distances.argmin(axis=1)]
```

**无 fallback**：最近邻查找是确定性的，没有"找不到就随机"的逻辑。

---

## 3. 训练策略：最小假设 + 聚合引导

### 3.1 核心问题

没有真实的 `(person, building)` 配对数据，如何训练联合模型？

**关键洞见**：与其用人为规则构造配对（这会把规则嵌入训练数据），不如：
1. 训练时使用**最小假设**
2. 采样时用**聚合约束引导**

### 3.2 最小假设配对（训练阶段）

**核心原则**：只使用几乎不可避免的假设——"人住在某个地理单元内的某栋建筑"。

```python
def minimal_assumption_pairing(person, buildings_in_tract, n_soft_labels=5, seed=None):
    """
    最小假设配对：tract内均匀随机采样
    
    不假设：
    - 收入-房价关系（让模型自己学）
    - 年龄-区位偏好（让模型自己学）
    - 任何社会学先验（避免人为设定）
    
    唯一假设：
    - 人住在其所属tract内的建筑中（地理约束，不可避免）
    """
    rng = np.random.default_rng(seed)
    
    # 软标签：为每个person采样多栋候选建筑
    # 这增加训练多样性，避免过拟合单一配对
    n_buildings = len(buildings_in_tract)
    sampled_indices = rng.choice(n_buildings, size=n_soft_labels, replace=True)
    
    return [buildings_in_tract[i] for i in sampled_indices]
```

**为什么这是最小假设？**
- 地理约束（人住在某tract）是**数据本身的结构**，不是我们添加的规则
- 均匀随机意味着**不偏向任何配对模式**
- 如果最终模型学到"高收入→高价房"，那是**模型发现的规律**，不是我们设计的

### 3.3 聚合一致性引导（采样阶段）

**核心思路**：约束在**聚合层面**定义，而非微观配对层面。

我们有的真值是**聚合统计**（ACS边际），不是微观配对。所以约束应该在采样时通过聚合统计施加：

```python
def aggregate_guided_sampling(model, tract, acs_marginals, n_samples, guidance_scale=2.0):
    """
    聚合引导采样：让生成的tract级分布接近ACS真值
    
    实现方式：Classifier-Free Guidance 的聚合变体
    """
    # 初始化噪声
    x_T = torch.randn(n_samples, d_person + d_bldg)
    c = encode_condition(tract=tract)
    
    for t in reversed(range(T)):
        # 标准去噪预测
        eps_pred = model(x_t, t, c)
        
        # 计算当前生成样本的聚合统计
        x_0_pred = predict_x0(x_t, eps_pred, t)
        current_age_dist = compute_age_distribution(x_0_pred[:, :d_person])
        current_income_dist = compute_income_distribution(x_0_pred[:, :d_person])
        
        # 聚合引导梯度：推动分布接近ACS
        grad_age = gradient_towards_target(current_age_dist, acs_marginals['age'])
        grad_income = gradient_towards_target(current_income_dist, acs_marginals['income'])
        
        # 组合引导
        guidance = guidance_scale * (grad_age + grad_income)
        
        # 去噪步 + 引导
        x_t = denoise_step(x_t, eps_pred, t) + guidance
    
    return x_0

def gradient_towards_target(current_dist, target_dist):
    """
    计算推动current_dist接近target_dist的梯度
    可以用KL散度、TVD或Wasserstein距离的梯度
    """
    # 简化实现：直接用分布差异作为梯度方向
    return target_dist - current_dist  # 或更精细的梯度计算
```

### 3.4 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  输入:                                                        │
│    - PUMS: {(person_attrs, tract)}                           │
│    - Buildings: {(bldg_features, tract)}                     │
│    - ACS: tract级边际分布（用于采样时引导，不用于训练）         │
│                                                              │
│  步骤:                                                        │
│                                                              │
│  1. Building Encoder 预训练                                   │
│     - 对比学习：同tract建筑靠近，不同tract建筑远离            │
│     - 或自监督重构：z_bldg → bldg_features                   │
│     - 输出：每栋建筑的 z_bldg ∈ R^16                          │
│                                                              │
│  2. 最小假设配对（软标签）                                     │
│     - 对每个 PUMS person，在其 tract 内随机采样 k 栋建筑      │
│     - 输出: {(person_attrs, z_bldg_1, tract),                │
│              (person_attrs, z_bldg_2, tract), ...}           │
│                                                              │
│  3. 联合扩散训练                                              │
│     - x = [person_attrs, z_bldg]                             │
│     - c = tract_embedding                                    │
│     - L = E[||ε - ε_θ(x_t, t, c)||²]  # 标准DDPM损失         │
│     - 无聚合损失（聚合约束在采样时施加）                       │
│                                                              │
│  4. 采样时聚合引导                                            │
│     - 注入ACS边际约束                                         │
│     - 注入空间证据（POI、夜光）作为条件                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.5 为什么这是更好的设计？

| 维度 | 人为配对（旧方案） | 最小假设+聚合引导（新方案） |
|------|-------------------|---------------------------|
| **假设** | 嵌入收入-房价等规则 | 仅假设地理约束 |
| **规则来源** | 人为设计 | 模型学习 |
| **约束层面** | 微观配对 | 聚合统计 |
| **可解释性** | 规则显式但人为 | 结果可验证 |
| **与叙事对齐** | ❌ 约束外挂 | ✅ 约束内生 |

### 3.6 验证模型是否学到了合理配对

训练后，我们可以验证模型是否**自动学到**了合理的配对模式：

```python
def validate_learned_patterns(generated_samples):
    """
    检查模型是否学到了合理的人-建筑配对模式
    这些模式应该是模型发现的，不是我们设计的
    """
    # 检查1：收入-房价相关性
    corr = pearsonr(generated_samples['income'], generated_samples['bldg_price'])
    print(f"Income-Price correlation: {corr}")  # 期望正相关
    
    # 检查2：空间聚集性
    moran_i = compute_morans_i(generated_samples, variable='age')
    print(f"Age spatial autocorrelation: {moran_i}")  # 期望显著
    
    # 检查3：与ACS边际的一致性
    tvd = compute_tvd(generated_samples['age_dist'], acs['age_dist'])
    print(f"Age TVD vs ACS: {tvd}")  # 期望小
```

如果这些指标合理，说明模型成功学到了人-建筑的合理配对模式，**且这些模式是数据驱动的，不是规则驱动的**。

---

## 4. 采样与条件注入

### 4.1 无条件采样

```python
def sample_joint(model, n_samples, puma):
    """基础采样：生成 (person_attrs, z_bldg)"""
    # 初始化噪声
    x_T = torch.randn(n_samples, d_person + d_bldg)
    
    # 条件向量
    c = encode_condition(puma=puma)
    
    # DDPM 反向采样
    for t in reversed(range(T)):
        x_t = denoise_step(model, x_t, t, c)
    
    # 分离属性和建筑潜变量
    person_attrs = x_0[:, :d_person]
    z_bldg = x_0[:, d_person:]
    
    # 解码建筑（最近邻查找）
    bldg_ids = decode_building(z_bldg, building_index[puma])
    
    return person_attrs, bldg_ids
```

### 4.2 条件注入（Classifier-Free Guidance）

多源条件可以在采样时动态注入：

```python
def sample_with_guidance(model, n_samples, conditions, guidance_scale=1.5):
    """
    conditions 可包含:
      - puma: 地理约束
      - poi_density: 期望的POI密度范围
      - price_tier: 期望的房价档次
      - age_target: 期望的年龄分布（来自ACS边际）
    """
    for t in reversed(range(T)):
        # 有条件预测
        eps_cond = model(x_t, t, c=conditions)
        # 无条件预测
        eps_uncond = model(x_t, t, c=None)
        # Guidance
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        x_t = denoise_step_with_eps(x_t, t, eps)
    
    return x_0
```

**这实现了"约束内生满足"**——ACS边际、POI密度等约束在采样过程中持续注入，而非后处理。

---

## 5. 验证指标体系

### 5.1 统计层（与 ACS 对齐）

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| Marginal TVD | 分箱后 0.5*L1 | < 0.05 |
| Association preservation | Pearson/Cramér's V | diff < 0.05 |
| Structural zero violation | 不可行组合占比 | 0% |

### 5.2 空间层（联合生成特有）

| 指标 | 计算方式 | 目标 |
|------|----------|------|
| Income-Price correlation | corr(person_income, bldg_price) | > 0.3 (正相关) |
| Spatial autocorrelation | Moran's I on building-level age | 合理的空间聚集 |
| POI-density consistency | 商业区年轻人比例 | 符合文献预期 |

### 5.3 消融对比

| 实验 | 目的 |
|------|------|
| Scheme B (分离式) vs Scheme C (联合) | 证明联合生成的优势 |
| 不同兼容性先验的消融 | 识别哪些先验最有效 |
| Guidance scale 消融 | 找到约束强度与多样性的平衡 |

---

## 6. 实现路线图

### Phase 1: 基础设施（1周）

- [ ] `BuildingEncoder` 实现与预训练
- [ ] 建筑特征标准化与潜空间索引构建
- [ ] 概率配对构造函数

### Phase 2: 联合扩散（1-2周）

- [ ] `JointDenoiser` 架构（基于现有 TabDDPM 扩展）
- [ ] 训练流程适配（输入维度扩展、条件拼接）
- [ ] 采样与建筑解码

### Phase 3: 条件注入（1周）

- [ ] Classifier-free guidance 实现
- [ ] 多源条件编码器（POI、夜光、ACS边际）
- [ ] 边际引导采样

### Phase 4: 验证与消融（1周）

- [ ] 空间层验证指标
- [ ] Scheme B vs C 对比实验
- [ ] 消融分析

---

## 7. 与基金叙事的对齐

| 叙事关键词 | 架构对应 |
|------------|----------|
| "属性-位置联合生成" | `x = [person_attrs, z_bldg]` 联合建模 |
| "普查边际为宏观约束" | ACS marginal guidance 在采样时注入 |
| "建筑物/POI为空间锚定" | Building encoder + POI条件 |
| "约束内生满足" | Classifier-free guidance，无后处理 |
| "扩散模型核心作用" | 学习联合分布的生成动力学 |
| "多源数据融合" | 条件向量 `c = [c_geo, c_spatial, c_temporal]` |

---

## 8. 风险与缓解

| 风险 | 缓解策略 |
|------|----------|
| 概率配对质量影响训练 | 消融不同兼容性先验；温度参数调节 |
| 建筑潜空间表达能力不足 | 增加 d_latent；对比学习预训练 |
| 采样慢（联合维度增加） | 减少 timesteps；DDIM 加速 |
| 最近邻解码引入离散化误差 | 评估解码前后 z_bldg 距离分布 |

---

## 附录：与现有代码的关系

- `src/synthpop/model/diffusion_tabular.py` → 扩展为 `JointDiffusionModel`
- `src/synthpop/spatial/building_allocation.py` → 废弃（不再需要后处理分配）
- `src/synthpop/features/condition_vectors.py` → 扩展为多源条件编码
- `tools/poc_tabddpm_pums_buildingcond.py` → 新建 `poc_joint_diffusion.py`

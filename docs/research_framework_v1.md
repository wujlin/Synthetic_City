# 研究框架：数据驱动的建筑物尺度人口画像生成

> **核心命题**：通过多源数据融合与扩散模型，实现建筑物尺度的合成人口生成，其中人-建筑配对关系从数据中学习，而非规则设定。

---

## 0. 研究路径与方案演进

本研究采用渐进式验证策略，通过 Scheme B → Scheme C 的演进，逐步解决建筑物尺度合成人口的核心挑战：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Scheme B (Baseline): 分离生成 + 规则分配                                │
│                                                                          │
│  目的: 验证扩散模型能否学习人口属性联合分布                               │
│                                                                          │
│  方法:                                                                   │
│    • 训练: P(AGEP, PINCP, SEX | PUMA)  ← 只用 PUMS 数据                 │
│    • 分配: income_price_match 规则分配到 building                        │
│                                                                          │
│  验证结果:                                                               │
│    • PUMA 级 AGEP_bin TVD = 0.058 ✅ (目标 < 0.08)                      │
│    • Tract 级 AGEP_bin TVD = 0.243 ❌ (结构性偏高)                       │
│                                                                          │
│  结论: 扩散模型有效学习属性分布，但规则分配无法捕捉 tract 结构            │
│        → 需要 Veraset 数据提供 person-building 配对信号                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Scheme C (Target): 多源对齐 + 联合生成                                  │
│                                                                          │
│  目的: 通过 Veraset 数据学习 person-building 联合结构                    │
│                                                                          │
│  方法:                                                                   │
│    • Stage 1: 对齐学习 (无监督，构建 shared latent space)               │
│    • Stage 2: 软配对构造 (基于对齐后的相似度)                            │
│    • Stage 3: 联合扩散训练 P([z_person, z_building] | PUMA)             │
│                                                                          │
│  预期:                                                                   │
│    • Tract 级 TVD 显著下降 (因模型学到了联合结构)                        │
│    • 无需手工规则分配                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心区别**：

| 维度 | Scheme B | Scheme C |
|------|----------|----------|
| 训练数据 | PUMS only | PUMS + Veraset + Buildings |
| person-building 配对 | 规则设定 (income→price) | 数据学习 (通过 device 桥接) |
| 模型输出 | 只生成 person attrs | 联合生成 (person, building) |
| Tract 精度来源 | 分配器 (0% 模型贡献) | 模型学习 (100% 模型贡献) |

---

## 1. 问题定义与核心挑战

### 1.1 研究目标

生成建筑物尺度（building-level）的合成人口，满足：
- **统计一致性**：与官方普查边际分布（ACS）一致
- **空间锚定**：每个合成个体被分配到具体建筑物
- **联合合理性**：人口属性与建筑特征的联合分布符合现实

### 1.2 核心挑战：联合分布不可识别

| 可观测数据 | 粒度 | 内容 |
|-----------|------|------|
| PUMS | PUMA (~100k人) | 个体属性 (age, income, tenure...) |
| ACS | tract (~4k人) | 边际分布统计 |
| Buildings | 建筑物 | 建筑特征 (price, type, capacity) |

**关键缺失**：没有 `(person, building)` 的配对数据。

传统方法的困境：
- **规则驱动**：用"高收入→高价房"等规则分配，本质是把假设伪装成结果（循环论证）
- **随机配对**：训练时随机配对，配对关系不携带互信息，模型学到 P(person) × P(building) 独立分布
- **后处理修正**：生成后强行调整边际，破坏联合分布质量
- **采样引导到验证目标**：若引导目标 = 验证目标，则为保驾护航，TVD 下降不代表模型有效

> **关键洞察**：无论规则多精巧、随机多"公平"，都无法从训练数据中学到真实的 person-building 关联——因为训练数据里根本没有这个信息。

### 1.3 我们的突破口

**关键洞见**：移动定位数据（Veraset）提供了 `device → location` 的行为轨迹，可作为**弱监督信号**桥接人口属性与建筑位置。

```
PUMS (person attrs)     Veraset (device behavior)     Buildings
        ↓                         ↓                        ↓
   [Latent Space Z: 共享表示空间，通过多源数据对齐学习]
        ↓                         ↓                        ↓
   z_person    ←── 对齐学习 ──→   z_device   ←── 对齐学习 ──→   z_building
```

---

## 2. 数据逻辑：从孤立边际到联合分布

### 2.1 数据源与信息层级

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         多源数据金字塔                                    │
│                                                                          │
│  Layer 4: 建筑物级                                                       │
│    └── Buildings: polygon, price_tier, capacity, land_use               │
│                                                                          │
│  Layer 3: CBG/Geohash5 级 (~1km)                                         │
│    ├── Veraset Home: device_id → home_CBG                               │
│    ├── Veraset Visits: device_id → POI (lat/lon, category, time)        │
│    └── ACS Summary: demographic marginals at tract/BG                   │
│                                                                          │
│  Layer 2: PUMA 级 (~100k人)                                              │
│    └── PUMS: individual-level attributes (age, income, tenure...)       │
│                                                                          │
│  Layer 1: City/County 级                                                 │
│    └── Administrative boundaries, total population                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据桥接逻辑

**核心问题**：如何从"各自独立的边际数据"构造"联合分布的学习信号"？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         桥接逻辑                                         │
│                                                                          │
│   PUMS ─────────────────────────────────────────────────→ Buildings     │
│    │         没有直接配对！需要桥接                           │         │
│    │                                                          │         │
│    │   ┌─────────── Veraset 数据作为桥梁 ───────────┐        │         │
│    │   │                                             │        │         │
│    ▼   ▼                                             ▼        ▼         │
│                                                                          │
│  Person          Device                          Building               │
│  (attrs)         (behavior)                      (features)             │
│    │                │                                │                  │
│    │                │                                │                  │
│    ▼                ▼                                ▼                  │
│                                                                          │
│  桥接1:           桥接2:                          桥接3:                 │
│  Person↔Device    Device↔Building               Building特征           │
│  通过CBG级        通过Veraset Home               通过Parcel/OSM         │
│  demographic      (device→home_CBG)              Assessment             │
│  一致性约束       + 活动圈空间约束                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 三重桥接的具体机制

#### 桥接1：Person ↔ Device（统计对齐）

**信号来源**：同一 CBG 内，person 分布（来自 ACS）应与 device 分布一致。

```python
# 伪代码：Person-Device 对齐损失
def person_device_alignment_loss(z_persons, z_devices, cbg):
    """
    同一 CBG 的 persons 和 devices 应在 latent space 中分布相似
    """
    # 聚合到 CBG 级别
    person_dist = aggregate_to_cbg(z_persons, cbg)
    device_dist = aggregate_to_cbg(z_devices, cbg)
    
    # 分布距离（可以是 MMD, Wasserstein, 或 KL）
    return distribution_distance(person_dist, device_dist)
```

**关键约束**：Device 分布存在渗透率偏差（年轻人、高收入者设备渗透率更高），需通过 ACS 边际做校正。

#### 桥接2：Device ↔ Building（空间-行为对齐）

**信号来源**：
1. **地理约束**：Veraset Home 告诉我们 device 的 home_CBG
2. **活动圈约束**：device 的非工作时间 POI visits 倾向于靠近 home

```python
# 伪代码：Device-Building 对齐
def device_building_alignment(device, buildings_in_cbg):
    """
    在 device 的 home_CBG 内，根据活动圈给 buildings 打分
    """
    # 计算 device 的活动圈中心（非工作时间的 POI visits）
    activity_center = compute_activity_centroid(
        device.visits, 
        time_filter='evening_weekend'
    )
    
    # 每栋 building 与活动圈中心的距离
    scores = []
    for bldg in buildings_in_cbg:
        dist = haversine(bldg.centroid, activity_center)
        scores.append(1.0 / (1.0 + dist))
    
    return softmax(scores)  # 概率配对分布
```

#### 桥接3：行为特征 → 隐含 Demographic

**信号来源**：POI visit pattern 隐含了 demographic 信息。

| Visit Pattern 特征 | 隐含的 Demographic 信号 |
|-------------------|------------------------|
| 高端餐厅/奢侈品店访问比例 | 收入水平 |
| 儿童相关 POI（学校、游乐场）| 家庭结构 |
| 活动时段分布 | 就业状态、年龄 |
| 活动空间范围 | 机动性、年龄 |
| 医疗/养老 POI 访问 | 年龄、健康 |

```python
# Device 行为特征编码
def encode_device_behavior(device_visits):
    """
    将 device 的 POI visit pattern 编码为 latent vector
    """
    features = {
        # POI 类别分布
        'dining_upscale_ratio': ratio(visits, 'Fine Dining'),
        'dining_casual_ratio': ratio(visits, 'Fast Food'),
        'retail_luxury_ratio': ratio(visits, 'Luxury Retail'),
        'retail_discount_ratio': ratio(visits, 'Discount Store'),
        'childcare_ratio': ratio(visits, ['School', 'Playground', 'Daycare']),
        'healthcare_ratio': ratio(visits, 'Healthcare'),
        
        # 时间模式
        'weekday_daytime_ratio': time_ratio(visits, 'weekday_9to17'),
        'weekend_ratio': time_ratio(visits, 'weekend'),
        
        # 空间模式
        'activity_radius_km': compute_activity_radius(visits),
        'home_work_distance_km': compute_commute_distance(device),
        'unique_cbg_count': count_unique_cbgs(visits),
    }
    
    return feature_encoder(features)
```

---

## 3. 模型架构：Scheme C 分阶段训练流程

> **关键设计原则**：避免循环依赖。对齐学习、配对构造、联合扩散训练必须**严格分阶段**进行。

### 3.1 Stage 1: 对齐学习（无监督）

**目标**：构建 Person-Device-Building 的共享 Latent Space，使三类实体可比较。

**关键**：此阶段**不需要** (person, building) 配对数据，只用 CBG 级别的统计一致性约束。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Stage 1: 对齐学习 (无监督)                            │
│                                                                          │
│   输入:                                                                  │
│     • PUMS: person features (不含 tract/building 信息)                  │
│     • Veraset: device → home_CBG, POI visits                            │
│     • Buildings: building features + CBG 归属                           │
│                                                                          │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│   │    Person     │     │    Device     │     │   Building    │         │
│   │   Encoder     │     │   Encoder     │     │   Encoder     │         │
│   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘         │
│           │                     │                     │                  │
│           ▼                     ▼                     ▼                  │
│         z_p                   z_d                   z_b                  │
│                                                                          │
│   对齐损失 (不需要配对标签):                                              │
│     L = L_contrast(z_d, z_b | same_CBG)     ← Device-Building 空间对齐  │
│       + L_dist(z_p, z_d | CBG)              ← Person-Device 统计对齐    │
│       + L_spatial(activity_center, bldg)   ← 活动圈一致性               │
│                                                                          │
│   输出: 训练好的 E_p, E_d, E_b (三个 encoder)                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Stage 2: 软配对构造

**目标**：基于 Stage 1 对齐后的 latent space，构造 (person, building) 训练对。

**关键**：配对概率来自**学习到的相似度**，不是人为规则。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Stage 2: 软配对构造                                   │
│                                                                          │
│   输入: Stage 1 训练好的 encoders + 原始数据                             │
│                                                                          │
│   For each PUMA:                                                         │
│     1. 编码所有 persons → z_p                                           │
│     2. 编码所有 devices (home_CBG in PUMA) → z_d                        │
│     3. 编码所有 buildings → z_b                                         │
│                                                                          │
│     4. 构造软配对:                                                       │
│        For each z_p:                                                     │
│          • 计算 sim(z_p, z_d) 得到 device 权重                          │
│          • 对每个 device，计算其活动圈内 building 的权重                 │
│          • 采样 (z_p, z_b, weight) 配对                                 │
│                                                                          │
│   输出: {(z_person, z_building, puma, weight)} 训练数据集               │
│                                                                          │
│   ⚠️ 注意:                                                               │
│     • weight 可以是 soft (概率) 或 hard (采样)                          │
│     • 配对信号来自 Veraset 行为数据，不是人为规则                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Stage 3: 联合扩散训练

**目标**：训练扩散模型生成 (z_person, z_building) 联合分布。

**关键**：条件变量是 **PUMA**（训练数据中存在），不是 tract（训练数据中没有）。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Stage 3: 联合扩散训练                                 │
│                                                                          │
│   输入: Stage 2 构造的软配对数据                                         │
│                                                                          │
│   训练目标:                                                              │
│     x = [z_person, z_building] ∈ R^{2d}                                 │
│     c = PUMA one-hot  ← 注意：只用 PUMA 作为条件，不用 tract            │
│                                                                          │
│     L = E[||ε - ε_θ(x_t, t, c)||²]                                      │
│                                                                          │
│   模型学到:                                                              │
│     P([z_person, z_building] | PUMA)                                    │
│     ↑ 这个联合分布来自 Veraset 桥接的配对，不是规则                      │
│                                                                          │
│   Tract 精度来源:                                                        │
│     • 模型学到 person-building 的联合结构                                │
│     • 同类 person 倾向于配对到同类 building                              │
│     • 同类 building 往往在同一 tract → tract 一致性自然涌现              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Stage 4: 采样与约束

**关键区分**：
- ✅ **PUMA 级约束**：可以用 Distribution Guidance（模型训练时见过 PUMA 条件）
- ❌ **Tract 级约束**：不能用 Guidance 引导（否则是保驾护航，验证无效）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Stage 4: 采样与约束                                   │
│                                                                          │
│   采样流程:                                                              │
│     1. 指定 PUMA + 人口数量 → 生成 [z_person, z_building]               │
│     2. z_building → Nearest Neighbor → bldg_id → tract_geoid           │
│     3. z_person → Decoder → (age, income, sex, ...)                    │
│                                                                          │
│   约束层级 (避免循环论证):                                               │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  ✅ 可用的约束 (训练时见过):                                      │   │
│   │     • PUMA 级 Distribution Guidance                              │   │
│   │     • 硬规则约束 (age < 16 → income = 0)                         │   │
│   │     • 建筑容量约束 (Feasibility Projection)                      │   │
│   │                                                                  │   │
│   │  ❌ 不可用的约束 (会导致循环论证):                                 │   │
│   │     • Tract 级 Distribution Guidance                             │   │
│   │     • 任何以验证目标作为引导目标的约束                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   输出: {(person_attributes, building_id, tract_geoid)}                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Stage 1 详细实现：对齐学习

```python
class MultiSourceAligner:
    """
    多源数据对齐学习 (Stage 1)
    
    关键：此阶段不需要 (person, building) 配对标签
    所有对齐信号来自 CBG 级别的统计约束
    """
    
    def alignment_loss(self, batch):
        """
        总对齐损失 = 对比损失 + 分布匹配损失
        
        注意：没有 person-building 直接配对损失
        person-building 关联通过 device 作为桥梁隐式学习
        """
        z_persons = self.person_encoder(batch['person_features'])
        z_devices = self.device_encoder(batch['device_features'])
        z_buildings = self.building_encoder(batch['building_features'])
        
        # 损失1：Device-Building 对比学习（空间对齐）
        # 正样本：同一 CBG 的 (device, building)  ← 来自 Veraset home_CBG
        # 负样本：不同 CBG 的 (device, building)
        # 这个信号来自观测数据，不是规则
        L_contrast = self.contrastive_loss(
            z_devices, z_buildings,
            positive_mask=batch['same_cbg_mask']
        )
        
        # 损失2：Person-Device 分布匹配（统计对齐）
        # 同一 CBG 的 persons 和 devices 在 latent space 中分布应相似
        # 这是弱监督：假设同 CBG 的人和设备有相似的 demographic 分布
        L_dist = self.distribution_matching_loss(
            z_persons, z_devices, 
            cbg_ids=batch['cbg_ids']
        )
        
        # 损失3：活动圈一致性（空间约束）
        # Device 的活动圈中心应接近同 CBG 的 buildings
        # 这个信号来自 Veraset POI visits，不是规则
        L_spatial = self.activity_center_loss(
            batch['device_activity_center'],
            batch['buildings_in_cbg_centroids']
        )
        
        return L_contrast + λ1 * L_dist + λ2 * L_spatial
```

**为什么这不是循环依赖？**

| 阶段 | 输入 | 输出 | 依赖 |
|------|------|------|------|
| Stage 1 | 原始特征 + CBG 标签 | 对齐后的 encoders | 无 |
| Stage 2 | 对齐后的 encoders | (person, building) 配对 | Stage 1 |
| Stage 3 | 配对数据 | 联合扩散模型 | Stage 2 |

每个阶段只依赖前一阶段的输出，不存在循环。

### 3.6 Stage 2 详细实现：软配对构造

```python
def construct_training_pairs(aligned_encoders, data):
    """
    Stage 2: 构造 (z_person, z_building) 训练对
    
    关键设计:
    1. 使用 Stage 1 训练好的 encoders (frozen)
    2. 配对概率来自 latent space 相似度，不是人为规则
    3. 按 PUMA 组织 (不是 tract)，因为 PUMS 只有 PUMA 标识
    """
    pairs = []
    
    # 注意：按 PUMA 组织，不是 tract
    # 因为 PUMS 只有 PUMA 地理标识，没有 tract
    for puma in data.pumas:
        # 获取该 PUMA 内的 persons (来自 PUMS)
        persons = data.get_persons_in_puma(puma)
        
        # 获取该 PUMA 内的 devices (home_CBG in PUMA)
        devices = data.get_devices_with_home_in_puma(puma)
        
        # 获取该 PUMA 内的 buildings
        buildings = data.get_buildings_in_puma(puma)
        
        if len(devices) == 0:
            # 无 Veraset 覆盖的 PUMA，跳过
            # 或降级到 Scheme B (规则分配)
            continue
        
        # 使用 Stage 1 训练好的 encoders (frozen, 不再更新)
        with torch.no_grad():
            z_persons = aligned_encoders.person(persons)
            z_devices = aligned_encoders.device(devices)
            z_buildings = aligned_encoders.building(buildings)
        
        # 预计算 device-building 亲和度 (基于 Veraset 活动圈)
        device_building_affinity = compute_activity_affinity(
            devices, buildings  # 基于活动圈距离
        )
        
        # 构造配对：通过 device 作为桥梁
        for i, z_p in enumerate(z_persons):
            # Step 1: person → device (基于 latent 相似度)
            device_scores = cosine_similarity(z_p.unsqueeze(0), z_devices)
            matched_device_idx = sample_from_scores(device_scores)
            
            # Step 2: device → building (基于活动圈亲和度)
            building_scores = device_building_affinity[matched_device_idx]
            matched_building_idx = sample_from_scores(building_scores)
            
            z_b = z_buildings[matched_building_idx]
            
            # 记录配对，条件是 PUMA (不是 tract)
            pairs.append({
                'z_person': z_p,
                'z_building': z_b,
                'puma': puma,
                'weight': device_scores[matched_device_idx].item()
            })
    
    return pairs
```

**为什么 Veraset 是必需的？**

| 配对方法 | person-building 关联来源 | 问题 |
|---------|------------------------|------|
| 随机配对 | 无 | 学到独立分布 P(p) × P(b) |
| 规则配对 | 人为规则 (income→price) | 循环论证 |
| **Veraset 配对** | **行为数据 (device→POI→building)** | **数据驱动，可验证** |

Veraset 的独特价值：它提供了**微观行为轨迹**，让模型能学到"什么样的人倾向于住在什么样的地方"——这是 ACS 和 PUMS 都无法提供的信息。

### 3.7 约束引导的适用边界

> ⚠️ **关键原则**：引导目标 ≠ 验证目标。若两者相同，则为循环论证。

```python
def guided_sampling(model, puma, n_samples, acs_puma_marginals=None):
    """
    采样时引导 (仅限 PUMA 级别)
    
    关键：只能用 PUMA 级别的 ACS 作为引导目标
          不能用 Tract 级别的 ACS (会导致循环论证)
    
    参考：Parihar et al. (CVPR 2024) "Distribution-Guided Debiasing"
    """
    # 初始化噪声
    x_T = torch.randn(n_samples, d_person + d_building)
    c = encode_puma_condition(puma)  # 条件是 PUMA，不是 tract
    
    for t in reversed(range(T)):
        # 标准去噪
        eps_pred = model(x_t, t, c)
        x_0_pred = predict_x0_from_eps(x_t, eps_pred, t)
        
        # ✅ PUMA 级引导 (可用，因为训练时见过 PUMA 条件)
        if acs_puma_marginals is not None:
            person_part = x_0_pred[:, :d_person]
            current_age_dist = soft_histogram(decode_age(person_part))
            
            # 引导到 PUMA 级 ACS 边际
            grad = gradient_of_tvd(current_age_dist, acs_puma_marginals['age'])
            guidance = guidance_scale(t) * grad
            x_t = x_t + guidance
        
        # 硬约束投影 (规则约束，非循环论证)
        x_t = feasibility_projection(x_t)
        
        # 正常去噪步
        x_t = denoise_step(x_t, eps_pred, t)
    
    return decode(x_0)
```

**约束层级表**：

| 约束 | 层级 | 可用性 | 原因 |
|------|------|--------|------|
| ACS @ PUMA | 训练条件 | ✅ 可引导 | 模型训练时见过 |
| ACS @ Tract | 验证目标 | ❌ 不可引导 | 引导=验证=循环论证 |
| 硬规则 (age<16→income=0) | 领域知识 | ✅ 可投影 | 非统计约束 |
| 建筑容量 | 物理约束 | ✅ 可投影 | 非统计约束 |

---

## 4. 核心 Insights

### Insight 1：弱监督信号的层级融合

> **不需要完美的配对数据**。通过多源数据的层级融合（PUMA→CBG→Building），可以从"弱监督信号"中学习联合分布。

| 层级 | 信号 | 强度 |
|------|------|------|
| PUMA | Person 属性分布 | 强（直接观测）|
| CBG | Device-Building 共现 | 中（地理约束）|
| Building | 活动圈一致性 | 弱（行为推断）|

**关键**：每一层的信号都是**从数据中观测的**，而非人为设定。

### Insight 2：行为即 Demographic

> **POI 访问模式是 demographic 的行为代理**。与其假设"高收入→高价房"，不如让模型学习"高端 POI 访问者→？"。

这避免了：
- 循环论证（假设 A，验证 A）
- 规则爆炸（特征增多时 cost 函数如何设计？）
- 假设争议（为什么 income-price 是线性的？）

### Insight 3：约束注入的层级原则

> **只能引导到训练时见过的条件层级**。引导目标 ≠ 验证目标。

| 约束类型 | 注入时机 | 方式 | 是否可作为验证指标？ |
|---------|---------|------|--------------------|
| PUMA 边际 (ACS) | 采样时 | Distribution Guidance | ✅ 可以 (但意义有限) |
| **Tract 边际 (ACS)** | **不可注入** | - | ✅ 核心验证指标 |
| 硬规则 (age<16→income=0) | 采样时 | Feasibility Projection | 违反率应为 0% |
| 联合结构 (person-building) | 训练时 | 对齐学习 + 联合扩散 | 通过 Tract TVD 间接验证 |

**核心洞察**：Tract 级验证之所以有意义，正是因为我们**没有**在采样时引导到 Tract marginals。如果引导了，TVD 下降只说明引导有效，不说明模型学到了联合结构。

### Insight 4：生态推断的现代解法

> **联合分布的识别需要额外信号**——移动数据提供了这个信号。

经典生态推断问题：从聚合数据推断个体行为，存在不可识别性。我们的解法：
- 不试图从"纯聚合数据"识别联合分布
- 引入"设备行为数据"作为**辅助变量**
- 通过 latent space 对齐，让模型学习 person-building 的隐式关联

### Insight 5：为什么 Veraset 不可替代

> **Veraset 提供的是微观配对信号，不是边际分布**。

| 数据源 | 提供什么 | 能学到的联合结构 |
|--------|---------|----------------|
| PUMS | P(attrs \| PUMA) 个体属性边际 | 属性间相关 (age-income) |
| ACS | P(attr \| tract) 边际统计 | 无（只有边际）|
| Buildings | P(features) 建筑特征 | 无（没有 person 信息）|
| **Veraset** | **device → home_CBG → behavior** | **person-building 联合** |

**替代方案的失败**：

| 方案 | 思路 | 为什么失败 |
|------|------|------------|
| 随机配对 | 随机分配 person→building | 配对不携带互信息，学到独立分布 |
| 规则配对 | income→price 规则 | 循环论证，学到的是注入的规则 |
| Building 特征条件 | P(attrs \| building_features) | 训练时配对仍需规则/随机 |
| Tract 聚合 | 用 tract 级统计 | 只有边际，没有微观配对 |

**Veraset 的独特性**：它提供了**设备级的行为轨迹**（device → POI visits → home），这是唯一能让模型学到"什么样的人住在什么样的地方"的信号。

---

## 5. 与现有方法的对比

| 方法 | 人-建筑配对 | 联合结构来源 | 约束满足 |
|------|------------|-------------|---------|
| **IPF/SIPP** | 规则分配 | 人为假设 | 后处理 |
| **传统 Microsimulation** | 规则分配 | 人为假设 | 迭代优化 |
| **TabDDPM (Scheme B)** | 不配对（分离生成） | 无 | 后处理 |
| **OT-Guided Diffusion** | OT coupling | Cost function 设计 | 训练时 |
| **本研究（Scheme C-v2）** | Latent space 对齐 | 多源数据学习 | 采样时 Guidance |

**核心区别**：我们的联合结构来自**数据观测**（行为-空间关联），而非**规则设定**。

---

## 6. 验证策略

### 6.1 验证层级与指标

> **核心原则**：验证层级必须高于训练条件层级，否则无法证明模型泛化能力。

| 层级 | 训练时角色 | 验证时角色 | 核心指标 |
|------|-----------|-----------|----------|
| PUMA | 条件变量 | 基础验证 | TVD < 0.08 (已达成) |
| **Tract** | **不可见** | **核心验证** | TVD < 0.10 (目标) |
| Building | 联合生成 | 内部一致性 | 无外部 ground truth |

### 6.2 统计验证

| 指标 | 目标 | 数据源 | 注意事项 |
|------|------|--------|----------|
| TVD @ PUMA | < 0.08 | ACS 边际 | Scheme B 已达成 |
| **TVD @ Tract** | < 0.10 | ACS 边际 | **核心指标，不可引导** |
| 联合一致性 (income×tenure) | 接近 ACS 交叉表 | ACS B25118 | 二阶验证 |
| 规则违反率 | 0% | 硬约束规则 | 可投影满足 |

### 6.2 空间验证

| 指标 | 目标 | 说明 |
|------|------|------|
| Moran's I (空间自相关) | 显著正值 | 相似人群应空间聚集 |
| 建筑容量满足率 | 100% | 不超过建筑容量 |
| CBG 级分布一致性 | 接近 Veraset Home 分布 | 外部验证 |

### 6.3 消融实验

| 消融 | 目的 |
|------|------|
| 移除 Device-Building 对齐 | 验证 Veraset 数据的贡献 |
| 移除 Distribution Guidance | 验证采样时约束的贡献 |
| 随机配对 baseline | 验证对齐学习的必要性 |

---

## 7. 研究贡献总结

1. **方法贡献**：提出基于多源数据对齐的联合扩散生成框架，实现建筑物尺度人口画像，其中人-建筑配对从数据学习而非规则设定。

2. **数据贡献**：首次将移动定位数据（Veraset）作为"行为-空间桥梁"引入合成人口生成，突破传统方法的配对数据缺失困境。

3. **理论贡献**：将生态推断问题重新表述为多源数据的 latent space 对齐问题，并通过扩散模型的 Distribution Guidance 机制实现"约束内生满足"。

4. **应用价值**：为城市模拟、疫情建模、应急规划等需要细粒度人口空间分布的应用提供数据基础。

# 研究框架：数据驱动的建筑物尺度人口画像生成

> **核心命题**：通过多源数据融合与扩散模型，实现建筑物尺度的合成人口生成，其中人-建筑配对关系从数据中学习，而非规则设定。

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
- **规则驱动**：用"高收入→高价房"等规则分配，本质是把假设伪装成结果
- **随机配对**：训练时随机配对，模型学到的是独立分布，无法捕捉联合结构
- **后处理修正**：生成后强行调整，破坏生成质量

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

## 3. 模型架构：共享 Latent Space + 联合扩散

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Phase 1: 构建共享 Latent Space                        │
│                                                                          │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│   │    Person     │     │    Device     │     │   Building    │         │
│   │   Features    │     │   Features    │     │   Features    │         │
│   │  (from PUMS)  │     │ (from Visits) │     │(from Parcels) │         │
│   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘         │
│           │                     │                     │                  │
│           ▼                     ▼                     ▼                  │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│   │   Encoder     │     │   Encoder     │     │   Encoder     │         │
│   │    E_p(·)     │     │    E_d(·)     │     │    E_b(·)     │         │
│   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘         │
│           │                     │                     │                  │
│           ▼                     ▼                     ▼                  │
│         z_p                   z_d                   z_b                  │
│           │                     │                     │                  │
│           └─────────┬───────────┼───────────┬─────────┘                  │
│                     │           │           │                            │
│                     ▼           ▼           ▼                            │
│               ┌─────────────────────────────────┐                        │
│               │     Shared Latent Space Z       │                        │
│               │         (dimension d)           │                        │
│               │                                 │                        │
│               │   对齐学习：                     │                        │
│               │   • Person-Device 统计对齐      │                        │
│               │   • Device-Building 空间对齐    │                        │
│               │   • 对比学习 + 分布匹配         │                        │
│               └─────────────────────────────────┘                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Phase 2: 联合扩散生成                                  │
│                                                                          │
│   输入条件 c：                                                            │
│     • tract_id（空间条件）                                                │
│     • ACS 边际分布（统计条件）                                            │
│     • 空间证据：POI 密度、夜间灯光、土地利用                               │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   x = [z_person, z_building] ∈ R^{2d}                           │   │
│   │                                                                  │   │
│   │   Forward:  x_0 → x_1 → ... → x_T  (加噪)                       │   │
│   │   Reverse:  x_T → x_{T-1} → ... → x_0  (去噪)                   │   │
│   │                                                                  │   │
│   │   去噪网络: ε_θ(x_t, t, c)                                       │   │
│   │                                                                  │   │
│   │   训练目标: L = E[||ε - ε_θ(x_t, t, c)||²]                      │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Phase 3: 约束引导采样                                  │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │   A. Distribution Guidance（聚合约束）                           │   │
│   │      • 计算 batch 内生成样本的边际分布                           │   │
│   │      • 与 ACS 目标分布计算距离梯度                               │   │
│   │      • 引导采样轨迹向目标分布靠近                                │   │
│   │                                                                  │   │
│   │   B. Feasibility Projection（硬约束）                            │   │
│   │      • 规则约束：age < 18 → income = 0                          │   │
│   │      • 容量约束：household_size ≤ building_capacity             │   │
│   │      • 在每个去噪步投影到可行域                                  │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    Phase 4: 解码与输出                                    │
│                                                                          │
│   z_person → Decoder_p → (age, sex, income, tenure, ...)                │
│                                                                          │
│   z_building → Nearest Neighbor Lookup → building_id                    │
│                                                                          │
│   输出: {(person_attributes, building_id)} for all synthetic persons    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 对齐学习：对比学习 + 分布匹配

```python
class MultiSourceAligner:
    """
    多源数据对齐学习
    """
    
    def alignment_loss(self, batch):
        """
        总对齐损失 = 对比损失 + 分布匹配损失
        """
        z_persons = self.person_encoder(batch['person_features'])
        z_devices = self.device_encoder(batch['device_features'])
        z_buildings = self.building_encoder(batch['building_features'])
        
        # 损失1：Device-Building 对比学习（空间对齐）
        # 正样本：同一 CBG 的 (device, building)
        # 负样本：不同 CBG 的 (device, building)
        L_contrast = self.contrastive_loss(
            z_devices, z_buildings,
            positive_mask=batch['same_cbg_mask']
        )
        
        # 损失2：Person-Device 分布匹配（统计对齐）
        # 同一 CBG 的 persons 和 devices 分布应相似
        L_dist = self.distribution_matching_loss(
            z_persons, z_devices, 
            cbg_ids=batch['cbg_ids']
        )
        
        # 损失3：活动圈一致性（空间约束）
        # Device 的活动圈中心应接近其配对的 building
        L_spatial = self.activity_center_loss(
            batch['device_activity_center'],
            batch['building_location']
        )
        
        return L_contrast + λ1 * L_dist + λ2 * L_spatial
```

### 3.3 联合扩散训练

训练数据构造（基于对齐后的 latent space）：

```python
def construct_training_pairs(aligned_encoders, data):
    """
    构造 (z_person, z_building) 训练对
    
    关键：配对概率来自学习到的对齐，不是人为规则
    """
    pairs = []
    
    for tract in data.tracts:
        # 获取该 tract 内的 persons（来自 PUMS，通过 PUMA 映射）
        persons = data.get_persons_in_tract(tract)
        
        # 获取该 tract 内的 buildings
        buildings = data.get_buildings_in_tract(tract)
        
        # 获取该 tract 内的 devices（有 home_CBG 在该 tract 内）
        devices = data.get_devices_with_home_in_tract(tract)
        
        # 编码到 latent space
        z_persons = aligned_encoders.person(persons)
        z_devices = aligned_encoders.device(devices)
        z_buildings = aligned_encoders.building(buildings)
        
        # 构造配对：通过 device 作为桥梁
        for z_p in z_persons:
            # 找最相似的 device
            device_scores = cosine_similarity(z_p, z_devices)
            matched_device_idx = sample_from_scores(device_scores)
            
            # 该 device 对应的 building 分布（基于活动圈）
            building_scores = device_building_affinity[matched_device_idx]
            matched_building_idx = sample_from_scores(building_scores)
            
            z_b = z_buildings[matched_building_idx]
            pairs.append((z_p, z_b, tract))
    
    return pairs
```

### 3.4 Distribution Guidance（采样时聚合约束）

```python
def distribution_guided_sampling(model, tract, acs_marginals, n_samples):
    """
    聚合引导采样：让生成的 batch 边际接近 ACS
    
    参考：Parihar et al. (CVPR 2024) "Distribution-Guided Debiasing"
    """
    # 初始化噪声
    x_T = torch.randn(n_samples, d_person + d_building)
    c = encode_condition(tract)
    
    for t in reversed(range(T)):
        # 标准去噪
        eps_pred = model(x_t, t, c)
        x_0_pred = predict_x0_from_eps(x_t, eps_pred, t)
        
        # 计算当前 batch 的边际分布
        person_part = x_0_pred[:, :d_person]
        current_age_dist = soft_histogram(decode_age(person_part))
        current_income_dist = soft_histogram(decode_income(person_part))
        
        # 分布引导梯度
        grad_age = gradient_of_tvd(current_age_dist, acs_marginals['age'])
        grad_income = gradient_of_tvd(current_income_dist, acs_marginals['income'])
        
        # 组合：标准去噪 + 引导
        guidance = guidance_scale(t) * (grad_age + grad_income)
        x_t = denoise_step(x_t, eps_pred, t) + guidance
        
        # 硬约束投影
        x_t = feasibility_projection(x_t)
    
    return decode(x_0)
```

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

### Insight 3：约束的正确注入时机

> **宏观约束在采样时注入，联合结构在训练时学习**。

| 约束类型 | 注入时机 | 方式 |
|---------|---------|------|
| 边际分布（ACS） | 采样时 | Distribution Guidance |
| 硬规则（age<18→income=0） | 采样时 | Feasibility Projection |
| 联合结构（person-building 关联） | 训练时 | 对齐学习 + 联合扩散 |

### Insight 4：生态推断的现代解法

> **联合分布的识别需要额外信号**——移动数据提供了这个信号。

经典生态推断问题：从聚合数据推断个体行为，存在不可识别性。我们的解法：
- 不试图从"纯聚合数据"识别联合分布
- 引入"设备行为数据"作为**辅助变量**
- 通过 latent space 对齐，让模型学习 person-building 的隐式关联

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

### 6.1 统计验证

| 指标 | 目标 | 数据源 |
|------|------|--------|
| TVD (边际) | < 0.05 | ACS 边际分布 |
| 联合一致性 (income×tenure) | 接近 ACS 交叉表 | ACS B25118 |
| 规则违反率 | 0% | 硬约束规则 |

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

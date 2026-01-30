代码仓库修改建议（Scheme C-v2: 数据驱动联合生成）
1. 架构对比：当前 vs 目标
当前实现 (Scheme B)	目标实现 (Scheme C-v2)
分离式：diffusion 学 person attrs，规则分配 building	联合式：diffusion 在 latent space 联合生成
assign_buildings_within_bg：容量加权随机	Latent space 对齐 + NN lookup
条件：PUMA/tract embedding only	条件：tract + device profile + building features
无 Veraset 数据支持	Device-Building 对齐作为核心桥梁
2. 新增模块建议
2.1 src/synthpop/encoders/ (新建目录)
目的：三路 Encoder，映射到共享 Latent Space
src/synthpop/encoders/
├── __init__.py
├── person_encoder.py      # PUMS attrs → z_person
├── device_encoder.py      # Veraset visit patterns → z_device  
├── building_encoder.py    # Building features → z_building
└── shared_latent.py       # 对齐学习逻辑（对比学习 + 分布匹配）
各模块职责：

文件	输入	输出	关键设计
person_encoder.py	PUMS 属性 (age, income, sex...)	z_person ∈ R^d	MLP，与 device encoder 对齐
device_encoder.py	POI visit pattern 特征向量	z_device ∈ R^d	特征：POI 类别分布、活动时段、空间范围
building_encoder.py	Building 特征 (price, type, capacity...)	z_building ∈ R^d	MLP 或对比学习预训练
shared_latent.py	三路 encoder + 对齐数据	对齐损失函数	InfoNCE + MMD + 活动圈一致性
2.2 src/synthpop/data/veraset.py (新建)
目的：处理 Veraset 移动数据

# 关键函数
def load_veraset_home(path) -> pd.DataFrame:
    """加载 device → home_CBG 映射"""
    
def load_veraset_visits(path) -> pd.DataFrame:
    """加载 device POI visits"""
    
def compute_device_features(visits: pd.DataFrame) -> pd.DataFrame:
    """
    从 POI visits 提取 device 行为特征：
    - POI 类别分布 (fine_dining_ratio, discount_store_ratio, ...)
    - 时间模式 (weekday_ratio, evening_ratio, ...)
    - 空间模式 (activity_radius, unique_cbg_count, ...)
    """
    
def compute_activity_center(visits: pd.DataFrame, time_filter='evening') -> pd.DataFrame:
    """计算 device 的活动圈中心（用于 building 配对先验）"""

2.3 src/synthpop/model/joint_diffusion.py (新建)
目的：替代当前 diffusion_tabular.py，在 latent space 做联合扩散

class JointDiffusionModel:
    """
    在 [z_person, z_building] 的联合 latent space 做扩散
    
    与 DiffusionTabularModel 的区别：
    - 输入不是原始属性，而是 latent vectors
    - 支持 Distribution Guidance 采样
    """
    
    def __init__(self, latent_dim: int, cond_dim: int, ...):
        ...
        
    def train_step(self, z_person, z_building, condition):
        """联合训练"""
        x = concat(z_person, z_building)
        ...
        
    def sample_with_guidance(self, condition, acs_marginals, n_samples):
        """
        Distribution Guidance 采样：
        - 在每个去噪步计算 batch 边际
        - 引导向 ACS 目标分布
        """

2.4 src/synthpop/alignment/ (新建目录)
目的：多源数据对齐学习

src/synthpop/alignment/
├── __init__.py
├── contrastive.py         # 对比学习损失（Device-Building 同 CBG）
├── distribution_match.py  # 分布匹配损失（Person-Device 统计对齐）
├── spatial_prior.py       # 活动圈一致性损失
└── training_pairs.py      # 从对齐后的 latent 构造训练配对

3. 修改现有模块
3.1 assign_buildings.py
当前：容量加权随机采样（规则驱动）

修改为：Latent space nearest neighbor lookup

def assign_buildings_latent_nn(
    *,
    z_person_generated: np.ndarray,      # 生成的 person latent
    z_buildings: np.ndarray,              # 预计算的 building latent
    buildings: pd.DataFrame,              # building 元数据
    tract_col: str = "tract_geoid",
    building_id_col: str = "bldg_id",
) -> pd.DataFrame:
    """
    基于 latent space 的 building 分配：
    1. 在同一 tract 内
    2. 找 latent 距离最近的 building
    
    替代 capacity-weighted random 分配
    """
3.2 soft_guidance.py
当前：post-hoc importance reweighting

修改为：支持采样时 Distribution Guidance

def distribution_guidance_step(
    *,
    x_t: torch.Tensor,           # 当前噪声样本
    x_0_pred: torch.Tensor,      # 预测的干净样本
    target_marginals: dict,      # ACS 边际分布
    guidance_scale: float,
    t: int,
) -> torch.Tensor:
    """
    单步 Distribution Guidance：
    1. 计算当前 batch 的边际（soft histogram）
    2. 计算与目标的距离梯度
    3. 返回 guidance 向量
    
    参考：Parihar et al. (CVPR 2024) Distribution-Guided Debiasing
    """

新增：

def soft_histogram(x: torch.Tensor, bins: torch.Tensor, sigma: float) -> torch.Tensor:
    """可微直方图，用于计算边际分布"""
    
def tvd_gradient(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TVD 距离对当前分布的梯度"""

3.3 condition_vectors.py
当前：空实现 (NotImplementedError)

修改为：实现条件向量构建

def build_tract_conditions(
    *,
    geo_units: pd.DataFrame,
    marginals: pd.DataFrame,
    poi_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建 tract 级条件向量：
    - 人口边际摘要 (age_median, income_median, ...)
    - 建成环境摘要 (poi_density, residential_ratio, ...)
    """
    
def build_device_features(
    *,
    veraset_visits: pd.DataFrame,
    device_home: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建 device 行为特征向量：
    - POI 类别分布
    - 时间模式
    - 空间模式
    """

4. 新增 Pipeline
4.1 src/synthpop/pipeline/detroit_v1.py (新建)
目的：Scheme C-v2 的完整 pipeline

def run_detroit_v1(config: Config) -> dict:
    """
    Phase 1: Encoder 预训练
    - 加载 Buildings, Veraset, PUMS
    - 预训练 Building Encoder（对比学习）
    - 预训练 Device Encoder（自监督）
    
    Phase 2: 多源对齐学习
    - Device-Building 对比学习（同 CBG 为正样本）
    - Person-Device 分布匹配（CBG 级统计对齐）
    
    Phase 3: 构造训练配对
    - 基于对齐后的 latent，为每个 person 找配对 building
    - 输出 (z_person, z_building, tract) 训练集
    
    Phase 4: 联合扩散训练
    - 在 [z_person, z_building] 上训练 JointDiffusionModel
    
    Phase 5: 采样
    - Distribution Guidance 采样
    - Feasibility Projection
    - Decode + Building Lookup
    
    Phase 6: 验证
    - 边际一致性 (TVD vs ACS)
    - 空间一致性 (Moran's I)
    - 敏感性分析
    """


5. 目录结构变更总览

src/synthpop/
├── __init__.py
├── cli.py
├── config.py
├── paths.py
+├── encoders/                    # 新增：三路 Encoder
+│   ├── __init__.py
+│   ├── person_encoder.py
+│   ├── device_encoder.py
+│   ├── building_encoder.py
+│   └── shared_latent.py
+├── alignment/                   # 新增：对齐学习
+│   ├── __init__.py
+│   ├── contrastive.py
+│   ├── distribution_match.py
+│   ├── spatial_prior.py
+│   └── training_pairs.py
+├── data/                        # 新增：数据加载
+│   ├── __init__.py
+│   ├── veraset.py              # Veraset 数据处理
+│   └── acs_crosstab.py         # ACS 交叉表加载
├── model/
│   ├── __init__.py
│   ├── diffusion_tabular.py     # 保留（可用于 baseline）
+│   └── joint_diffusion.py      # 新增：联合扩散
├── features/
│   ├── __init__.py
│   └── condition_vectors.py     # 修改：实现条件构建
├── constraints/
│   ├── __init__.py
│   ├── hard_rules.py
│   ├── projection.py
│   └── soft_guidance.py         # 修改：支持 Distribution Guidance
├── spatial/
│   ├── __init__.py
│   └── assign_buildings.py      # 修改：支持 latent NN lookup
├── pipeline/
│   ├── __init__.py
│   ├── detroit_v0.py            # 保留（baseline）
+│   └── detroit_v1.py           # 新增：Scheme C-v2 pipeline
└── validation/
    ├── __init__.py
    ├── stats.py
    ├── spatial.py
    └── temporal.py

6. 数据目录变更

data/detroit/
├── raw/
│   ├── tiger/
│   ├── acs/
│   ├── pums/
│   ├── buildings/
│   ├── poi/
+│   └── mobility/               # 新增：Veraset 数据
+│       ├── veraset_home/
+│       ├── veraset_visits/
+│       └── veraset.metadata.json
├── processed/
│   ├── geo_units/
│   ├── buildings/
│   ├── pums/
│   ├── marginals/
+│   ├── device_features/        # 新增：device 行为特征
+│   │   └── device_features.parquet
+│   ├── acs_crosstab/           # 新增：ACS 交叉表
+│   │   └── income_tenure.parquet
+│   └── latent_encodings/       # 新增：预计算的 latent
+│       ├── z_buildings.parquet
+│       └── z_devices.parquet

7. 实施优先级建议
优先级	任务	文件	说明
P0	创建 encoders 模块骨架	encoders/*.py	定义接口，先用简单 MLP
P0	实现 veraset 数据加载	data/veraset.py	解析 Veraset 格式，提取特征
P1	实现 Device-Building 对比学习	alignment/contrastive.py	核心对齐逻辑
P1	实现 Distribution Guidance	soft_guidance.py	可微直方图 + 梯度计算
P2	实现 JointDiffusionModel	model/joint_diffusion.py	在 latent space 做联合扩散
P2	修改 assign_buildings	assign_buildings.py	添加 latent NN lookup
P3	完整 detroit_v1 pipeline	pipeline/detroit_v1.py	串联所有模块
8. 关键接口定义
为确保 partner 实现与架构一致，以下是关键接口的签名：

# === encoders/shared_latent.py ===
class SharedLatentSpace:
    def __init__(self, latent_dim: int = 32):
        self.person_encoder: nn.Module
        self.device_encoder: nn.Module
        self.building_encoder: nn.Module
        
    def encode_person(self, person_features: torch.Tensor) -> torch.Tensor:
        """PUMS 属性 → z_person"""
        
    def encode_device(self, device_features: torch.Tensor) -> torch.Tensor:
        """Visit pattern 特征 → z_device"""
        
    def encode_building(self, building_features: torch.Tensor) -> torch.Tensor:
        """Building 特征 → z_building"""
        
    def alignment_loss(
        self,
        z_persons: torch.Tensor,
        z_devices: torch.Tensor,
        z_buildings: torch.Tensor,
        cbg_ids: torch.Tensor,
        activity_centers: torch.Tensor,
        building_locations: torch.Tensor,
    ) -> torch.Tensor:
        """总对齐损失 = 对比 + 分布匹配 + 空间一致性"""

# === model/joint_diffusion.py ===
class JointDiffusionModel:
    def __init__(
        self,
        latent_dim: int,        # z_person 和 z_building 的维度
        cond_dim: int,          # 条件向量维度
        config: TabDDPMConfig,
    ): ...
    
    def train_step(
        self,
        z_person: torch.Tensor,
        z_building: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """返回 loss"""
        
    def sample(
        self,
        condition: torch.Tensor,
        n_samples: int,
        acs_marginals: dict | None = None,  # 如果提供，启用 Distribution Guidance
        guidance_scale: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (z_person, z_building)"""


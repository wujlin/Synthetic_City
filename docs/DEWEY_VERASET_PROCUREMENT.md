# Dewey/Veraset 数据采购需求说明（面向本研究｜Detroit v0→v1）

## Thought（核心结论）

本研究的“生成式建筑物尺度人口画像”主线以 **建筑/地块级房价（评估值）作为收入代理**为核心空间条件，生成端不依赖移动数据。  
但若希望把“静态画像”扩展为可检验的“时空画像”（日/夜、通勤、活动强度），并建立评审可接受的**外部证据闭环**，则需要引入移动数据来提供 **CBG/网格级**的动态统计约束与验证基准。

`docs/deway_data/` 中提供的是 Dewey 平台下的 **Veraset** 系列数据集说明（Home/Work/Visits 等），均标注为 **仅机构许可（institutional licenses）可用**。这些数据的空间精度以 `CENSUS_BLOCK_GROUP` 与 `GEOHASH_5` 为主，属于 km 级/统计区尺度，不是建筑物尺度“真值”，适合做 **动态约束/验证** 而非建筑级监督标签。

---

## 1) 采购目标与研究用途映射

### 1.1 我们要解决的“可检验需求”

| 研究模块 | 需要的可检验量 | 需要的数据形态 |
|---|---|---|
| 时间动态（v1） | 日/夜人口空间分布差异；居住地→工作地通勤结构 | `Home` + `Work`（月度推断位置）及其在 CBG/网格上的聚合 |
| POI 活动验证（v1.5，可选） | 不同功能区的访问强度、时段分布与画像一致性 | `Visits`（事件级 POI visit） |
| 合规与可复现 | 不触碰 PII；输出仅以聚合统计/可视化呈现 | Vendor 条款 + 本地访问控制 + 聚合导出策略 |

### 1.2 推荐采购优先级（最小可用闭环）

**P1（推荐，动态闭环最小集）**
- `Home`（2023–present，月更）：居住地网格/CBG
- `Work`（2023–present，月更）：工作地网格/CBG

**P2（可选，POI 强度/功能验证）**
- `Visits`（2019–present，月更）：POI visit 事件级（注意 2024-06-01 统计口径变更）

**P3（补充/替代，若只需要历史到 2024）**
- `Home Visit` / `Work Visit`（2019–2024，月更）：对 home/work 的 visit 事件（仍是 geohash5/CBG 粒度）

---

## 2) 数据集总览（来源：Dewey 平台说明，见 `docs/deway_data/`）

> 说明：以下字段定义与覆盖期来自 `docs/deway_data/*.md`。实际交付格式（CSV/Parquet）、分区方式（按月/州/县）与可选字段以 Dewey 导出与合同为准。

### 2.1 Veraset Visits（POI Visits）

- **Source**：Veraset（POI provider 使用 SafeGraph 的 places/categorization）
- **Coverage**：US；2019–present；Monthly refresh
- **Spatial**：`GEOHASH_5`（网格） + `CENSUS_BLOCK_GROUP`（统计区）；并含 `STREET_ADDRESS/CITY/STATE/ZIPCODE`（POI 地址文本）
- **Time**：事件级 `UTC_TIMESTAMP`（Unix 秒）与 `LOCAL_TIMESTAMP`
- **Purpose（本研究）**：
  - 构建 CBG/网格级“活动强度/功能分布”的外部证据
  - 与 SafeGraph Places（我们已在 wsA 有）做 `SAFEGRAPH_PLACE_ID/PLACEKEY` 对齐，形成 POI 功能验证
  - 注意：**2024-06-01 起 visit 统计口径变化**，跨期对比需分段处理

### 2.2 Veraset Home / Work（推断 home/work 位置）

- **Source**：Veraset
- **Coverage**：US；2023–present；Monthly refresh
- **Spatial**：`GEOHASH_5` + `CENSUS_BLOCK_GROUP`（隐私要求，不提供地址）
- **Time**：无事件级时间戳（按月刷新，可视作“当月/周期的推断位置”）
- **Purpose（本研究）**：
  - 生成端：可作为 v1“日/夜（居住/工作）人口约束”的目标统计
  - 验证端：合成建筑人口聚合到 CBG/网格后，与 Home/Work 的空间分布一致性对齐

### 2.3 Home Visit / Work Visit（home/work 的 visits）

- **Source**：Veraset（同 visits 字段体系）
- **Coverage**：US；2019–2024；Monthly refresh
- **Spatial**：`GEOHASH_5` + `CENSUS_BLOCK_GROUP`
- **Time**：事件级 `UTC_TIMESTAMP/LOCAL_TIMESTAMP` + `MINIMUM_DWELL`
- **Purpose（本研究）**：
  - 构建日周期的“在家/在岗”强度指标（到 2024 为止）
  - 作为 `Home/Work` 的动态补充（但不覆盖 2025+）

---

## 3) 字段字典（字段意义 / 描述 / 用途）

### 3.1 通用字段（Visits 与 Home/Work Visits）

| 字段 | 含义/description | 用途（purpose） | 备注 |
|---|---|---|---|
| `UTC_TIMESTAMP` | UTC 时间（Unix 秒） | 构建日/周周期、事件聚合 | visits/home-work visits 有；home/work 无 |
| `LOCAL_TIMESTAMP` | 本地时间戳 | 与城市时区对齐的日周期分析 | 同上 |
| `CAID` | 设备伪匿名 ID | 设备级链路（如通勤链） | 仍属敏感数据，需严格访问控制 |
| `ID_TYPE` | Android/iOS 标识 | 设备结构诊断（可选） | |
| `GEOHASH_5` / `GEO_HASH5` | 5 位 geohash 网格单元 | 网格级聚合与对齐 | km 级网格；不同文件命名不一致需统一 |
| `CENSUS_BLOCK_GROUP` | 普查 CBG | 与 ACS/人口统计对齐 | 也是我们 v0 里已有的空间诊断尺度 |
| `MINIMUM_DWELL` | 最短停留时长（分钟） | 过滤噪声/快闪访问 | 仅 visit 类数据 |

### 3.2 POI Visits 特有字段（`visit.md`）

| 字段 | 含义/description | 用途（purpose） | 备注 |
|---|---|---|---|
| `LOCATION_NAME` | POI 名称 | 解释性展示（可选） | |
| `TOP_CATEGORY` / `SUB_CATEGORY` | POI 类别 | 功能区验证、POI 画像 | 与 SafeGraph 分类一致 |
| `NAICS_CODE` | 6 位 NAICS | 细粒度功能分析 | |
| `SAFEGRAPH_PLACE_ID` | SafeGraph POI ID | 与 SafeGraph Places 对齐 | 若为 home visit，= "home" |
| `PLACEKEY` | Placekey ID | 跨数据源 join 键 | |
| `STREET_ADDRESS/CITY/STATE/ZIPCODE` | POI 地址文本 | 地理解释/去重（可选） | 不是建筑物级“居住地址” |
| `BRANDS` | 汽车经销商品牌列表 | 特定行业分析（可选） | |

### 3.3 Home / Work（`home.md`, `work.md`）

| 字段 | 含义/description | 用途（purpose） | 备注 |
|---|---|---|---|
| `CAID` | 设备伪匿名 ID | 设备级 home-work 配对（可选） | 仅在合规允许且必要时使用 |
| `COUNTRY/REGION/CITY/ZIPCODE` | 行政区与邮编 | 过滤范围（Detroit/MI） | 可优先用 CBG 过滤 |
| `CENSUS_BLOCK_GROUP` | CBG | 空间聚合对齐 | |
| `GEOHASH_5` / `GEO_HASH5` | geohash5 网格 | 网格聚合对齐 | |

> `work.md` 额外描述了 WFH 概念：如果 work 与 home 落在同一 geohash，则 WFH=true（字段是否显式提供需向 Dewey 确认）。

---

## 4) 本研究的“最小交付清单”（写给采购/合同）

### 4.1 地域与时间范围（建议）

- 地域：Michigan → Wayne County → Detroit core（优先以 `CENSUS_BLOCK_GROUP` 过滤）
- 时间：
  - `Home/Work`：2023–至今（至少覆盖一个完整年度，便于季节性）
  - `Visits`：如用于动态验证，建议 2019–至今，但需注意 2024-06-01 的口径断点

### 4.2 必要字段（建议写进“必须交付”）

- `Home`：`CAID, CENSUS_BLOCK_GROUP, GEOHASH_5(or GEO_HASH5), REGION, CITY, ZIPCODE`
- `Work`：`CAID, CENSUS_BLOCK_GROUP, GEOHASH_5, REGION, CITY, ZIPCODE, ID_TYPE`
- `Visits`：`UTC_TIMESTAMP, LOCAL_TIMESTAMP, CAID, CENSUS_BLOCK_GROUP, GEOHASH_5, SAFEGRAPH_PLACE_ID, PLACEKEY, TOP_CATEGORY, SUB_CATEGORY, NAICS_CODE, MINIMUM_DWELL`

### 4.3 合规与输出限制（建议写进条款）

- 原始设备级数据（含 `CAID`）仅限授权成员在受控环境访问；不得入 git；不得外发给未授权人员。
- 对外发表/共享仅输出聚合统计（例如 CBG/tract 级分布、热力图、汇总表），不输出设备级记录。
- 允许将合成结果与 Veraset 聚合统计进行对齐验证（研究用途），并在论文/基金中引用数据来源与许可声明。

---

## 5) 交付与落盘建议（工程口径）

建议落盘到（示例）：

```text
$RAW_ROOT/synthetic_city/data/detroit/raw/mobility/veraset/
  home/        # 按月分区（建议）
  work/
  visits/
  home_visits/ # 可选
  work_visits/ # 可选
  dewey.metadata.json
```

并将可用于训练/验证的聚合产物输出到：

```text
$RAW_ROOT/synthetic_city/data/detroit/processed/mobility/
  cbg_home_pop_monthly.parquet
  cbg_work_pop_monthly.parquet
  cbg_visit_counts_hourly.parquet
```

---

## 6) 需向 Dewey 确认的关键问题（不确认会影响采购决策）

1) **导出格式与分区**：按月/州/县如何切分？是否可直接导出 Parquet？
2) **字段一致性**：`GEOHASH_5` 与 `GEO_HASH5` 是否同义？是否统一列名？
3) **WFH 字段**：是否提供显式 `WFH` 标识，还是需要我们自行从 home/work geohash 比较推断？
4) **口径断点**：对 `Visits` 的 2024-06-01 口径变化是否有官方修正/说明字段？
5) **范围过滤能力**：是否支持按 `CENSUS_BLOCK_GROUP` 白名单导出（仅 Detroit/Wayne）以降低成本与体量？
6) **合规边界**：研究发表允许展示到何种空间粒度（CBG/tract/geohash5）？是否允许发布 `building_portrait_points.geojson`（不含设备，只含合成画像）？


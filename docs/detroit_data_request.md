# 底特律（Detroit）落地：数据需求清单（给 Partner 检索用）

> 用途：请 Partner 按本清单检索“可获取的数据源链接 + 数据字典/字段说明 + 许可/使用限制 + 覆盖范围/年份 + 下载方式”。  
> 目标：先跑通 Detroit 的**建筑物尺度静态人口画像**（P0），再逐步增强到**户—人层级**与**日/夜或小时动态**（P1/P2）。

---

## A. 统一口径（先对齐）

1) **研究范围（默认）**：Detroit city boundary（后续可扩到 Wayne County / Metro）。  
2) **控制量空间层级（默认）**：Census **tract** 为主，**block group（BG）**为辅（若数据质量允许）。  
3) **目标年份（默认）**：以“最新可用的 ACS 5-year”为主；PUMS 与 ACS 年份尽量一致。  
4) **文件格式（交付建议）**：表格 `parquet/csv`；地理 `gpkg/GeoParquet`；附 `metadata.md`（来源、版本、许可、下载日期）。

---

## P0（必须）——没有这些就无法跑通“静态建筑物尺度画像”

### P0-1 行政边界与Census地理单元（Geo）

- **Detroit city boundary**
  - 需求字段：`geometry`、`name`、`GEOID`（若有）
  - 备注：用于裁剪所有数据到统一研究区
  - 可能来源：US Census TIGER/Line（place boundary）、Detroit open data（若有官方边界）

- **Census tracts / block groups 边界（含 GEOID）**
  - 需求字段：`GEOID`、`geometry`、`NAME`（可选）
  - 备注：用于构建宏观约束与空间叠加（building→BG/tract）
  - 可能来源：US Census TIGER/Line

### P0-2 宏观约束（ACS Summary / Decennial Summary）

> 目标是生成 `marginals_long`：`(geo_level, geoid, constraint_name, category, target, year, source)`。

请检索“底特律 tract/BG 级”可用的控制量表（至少包括）：

- 人口边际：年龄×性别（或年龄段、性别）
- 家庭边际：家庭规模（hh size）、家庭类型（可选）
- 社会经济：收入分箱（household income bins，或个人收入/贫困状态可选）
- 就业：就业状态（employed/unemployed/not in labor force）；职业/行业（可选，若 tract/BG 可得）
- 通勤（可选但很有价值）：通勤方式、通勤时间分箱

输出要求（给 Partner）：每张表请提供 **表号/变量名**（例如 ACS table id）、空间层级可用性（tract/BG）、年份范围。

### P0-3 微观种子样本（PUMS 或可替代种子）

> 若我们走“生成式+控制量约束”路线，最稳的是 PUMS 作为 household/person seed。

- **ACS PUMS（household + person）**
  - 需求：可下载入口、年份、Michigan 覆盖、PUMA 边界与 `PUMA↔tract/BG` 可用的 crosswalk（如果需要）
  - 关键字段（最小）：`AGEP/SEX`、`HINCP`（或收入）、`HHSIZE`、就业/职业（可选）、通勤方式（可选）、权重（`PWGTP/WGTP`）
  - 备注：后续会做分箱与标准化字段

> 如果不打算用 PUMS，请给替代方案：例如“本地调查/合成样本库/公开微观调查”，并说明覆盖范围与许可。

### P0-4 建筑物 footprint（Buildings）

- **Building footprints（Detroit 全覆盖）**
  - 需求字段：`geometry`，最好有 `source_id`
  - 必要派生字段：`bldg_id`、`footprint_area_m2`、`centroid`、`bg_geoid/tract_geoid`（后续空间叠加产生）
  - 可能来源（请优先检索开放）：Microsoft US Building Footprints、OSM、Detroit/Wayne open data（如有）

> 重要：请同时检索“是否有建筑物用途/居住属性/层数高度”等字段（即使是 P1，也先确认可得性）。

### P0-5 POI / 用地功能代理（POI / Landuse）

- **POI（点或面）**
  - 需求字段：`geometry`、`category`（可标准化到若干大类）
  - 用途：作为建筑物 landuse/功能强度的弱监督条件；也用于空间真实性验证
  - 可能来源：OSM POI、政府开放POI、商业POI（若能合作）

---

## P1（强建议）——显著提升空间锚定与可解释性

### P1-1 Parcel / 土地利用 / 分区（用于更稳的 residential 判定）

- **Parcel（地块）与属性**
  - 需求字段：`geometry`、`land_use`/`zoning`、`residential_units`（若有）、`year_built`（可选）
  - 用途：估计居住容量（res_capacity）、区分居住/非居住建筑
  - 可能来源：Detroit/Wayne assessor/open data、SEMCOG（若开放）

### P1-2 建筑高度/层数（用于容量先验）

- **Building height / floors（任一可用即可）**
  - 来源可能性：城市三维建筑数据、LiDAR 派生、OSM `building:levels`
  - 用途：把 footprint area → volume → capacity 先验

### P1-3 就业分布与通勤OD（用于“居住地–就业地”约束/验证）

- **LEHD LODES（WAC/RAC/OD）**
  - 用途：就业岗位空间分布、居住—工作地流（可作为约束或验证）
  - 备注：请确认可用年份、空间层级（通常到 block）

---

## P2（可选增强）——时空动态与更强验证

### P2-1 动态人口/热力/信令（外部验证）

- **平台聚合热力/迁徙/驻留指数**（腾讯/百度等）
  - 用途：日周期/热点迁移的弱真值参照
  - 需求：获取方式（公开页面/API/合作）、空间粒度与时间粒度说明

- **运营商信令/第三方LBS聚合统计**（若可合作）
  - 用途：小时级动态验证与 OD 验证
  - 需求：可提供到何种聚合层级（网格/基站/街道/BG）

### P2-2 交通供给数据（解释通勤方式/可达性）

- 路网（OSM）、GTFS（公交）
  - 用途：构建可达性特征，辅助通勤方式与就业分配

### P2-3 夜间灯光/遥感产品（空间强度代理）

- VIIRS 夜光（或其他）
  - 用途：作为夜间活动强度代理，用于空间/时间验证的弱证据

---

## B. Partner 回传格式（请按这个模板整理）

对每个数据源，请提供：

1) **数据集名称**（含年份/版本）  
2) **覆盖范围**（Detroit city / Wayne / MI / US）与**空间分辨率**（tract/BG/building/block/网格）  
3) **时间分辨率**（静态/日夜/小时）  
4) **下载入口**（URL）+ 获取方式（直接下载/申请/付费/合作）  
5) **许可与使用限制**（是否可用于基金申报、是否可公开发布派生结果）  
6) **字段字典/数据字典**（至少列出关键字段与含义）  
7) **文件格式与大致体量**（MB/GB）  


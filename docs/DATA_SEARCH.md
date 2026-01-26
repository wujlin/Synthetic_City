# Detroit（底特律）落地：数据检索清单（Buildings 已由 TUM 提供）

> 用途：请 Partner 按本清单检索“可获取的数据源链接 + 数据字典/字段说明 + 许可/使用限制 + 覆盖范围/年份 + 下载方式”。  
> 目标：先跑通 Detroit 的**建筑物尺度静态人口画像**（P0），再逐步增强到**户—人层级**与**日/夜或小时动态**（P1/P2）。  
> 交付建议：表格 `parquet/csv`；地理 `gpkg/GeoParquet`；附 `metadata.md`（来源、版本、许可、下载日期、处理步骤）。

---

## 0) 已有数据（不需 Partner 再检索）
- ✅ **Buildings（footprint + height）**：TUM 公开建筑物数据（polygon + height）
- ✅ **POI**：Deway 平台的 SafeGraph POI 数据  
  - 备注：请补齐“许可/使用限制（是否允许公开派生结果、是否允许基金申报、是否允许论文/开源发布）”到 `metadata.md`。

---

## A) 统一口径（先对齐）
1) **研究范围（默认）**：Detroit city boundary（后续可扩到 Wayne County / Metro）。  
2) **控制量空间层级（默认）**：Census **tract** 为主，**block group（BG）**为辅（若数据质量允许）。  
3) **目标年份（默认）**：以“最新可用的 ACS 5-year”为主；PUMS 与 ACS 年份尽量一致。  
4) **统一输出字段建议**：
   - `year`：数据年份或 5-year window（例如 `2019-2023`）
   - `source`：数据源名称（Census/Detroit ODP/LEHD 等）
   - `download_date`：下载日期（YYYY-MM-DD）
   - `license`：许可摘要 + 链接

---

# P0（必须）——没有这些就无法跑通“静态建筑物尺度画像”

## P0-1 行政边界与 Census 地理单元（Geo）

### 数据源 P0-1a：Detroit city boundary（首选：US Census TIGER/Line Places）
1) **数据集名称**：TIGER/Line Shapefiles – Places（建议用最新 TIGER 年份，例如 2025）  
2) **覆盖范围**：US（全美）；筛选 Michigan（statefp=26）内的 Detroit（place）。  
3) **空间分辨率**：place boundary（polygon）。  
4) **下载入口（URL）**：
   - TIGER/Line 总入口（Web/FTP）：https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
   - FTP（示例，具体按年/目录选择）：https://www2.census.gov/geo/tiger/TIGER2025/PLACE/
5) **许可与使用限制**：联邦公开地理产品；请在 `metadata.md` 中写明“来源 + TIGER 年份 + 下载日期”。  
6) **字段字典/关键字段（建议检查并保留）**：
   - `GEOID`, `NAME`, `NAMELSAD`, `STATEFP`, `PLACEFP`, `ALAND`, `AWATER`, `INTPTLAT`, `INTPTLON`, `geometry`
7) **文件格式与体量**：Shapefile(zip)；Partner 回传写明 zip 大小（MB）与解压后大小。

---

### 数据源 P0-1b：Detroit city boundary（备选：Detroit Open Data Portal / DetroitData）
> 用途：用于对比/核验边界；若官方业务边界与 TIGER 不一致，可记录差异。

1) **数据集名称**：City of Detroit Boundary（ODP / DetroitData）  
2) **覆盖范围**：Detroit city boundary（polygon）  
3) **下载入口（URL）**：
   - Detroit ODP（地图/图层）：https://data.detroitmi.gov/maps/detroitmi%3A%3Acity-of-detroit-boundary
   - DetroitData 数据集页：https://detroitdata.org/dataset/city-of-detroit-boundary
4) **许可与使用限制**：
   - 多数 DetroitData/ODP 条目会显示 “No License Provided / Request permission to use” 或要求遵循 Portal 免责声明/条款；**必须在 metadata 里逐条记录**。
5) **字段字典/关键字段**：至少保留 `name/NAME`（若有）、`geometry`；如无 `GEOID` 则以 TIGER 的 place GEOID 为主口径。  
6) **文件格式与体量**：ArcGIS Hub/Feature layer，可导出 GeoJSON/Shapefile；记录导出格式与大小。

---

## P0-1c Census Tracts / Block Groups 边界（含 GEOID）
1) **数据集名称**：TIGER/Line Shapefiles – Census Tracts；Block Groups  
2) **覆盖范围**：US（全美）；筛选 Michigan→Wayne County→裁剪 Detroit。  
3) **下载入口（URL）**：
   - TIGER/Line 总入口：https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
   - FTP（示例，按年选择）：  
     - Tracts：https://www2.census.gov/geo/tiger/TIGER2025/TRACT/
     - Block Groups：https://www2.census.gov/geo/tiger/TIGER2025/BG/
4) **许可与使用限制**：联邦公开地理产品；写明 TIGER 年份与下载日期。  
5) **字段字典/关键字段**：
   - Tract：`GEOID`（11 位）、`STATEFP`、`COUNTYFP`、`TRACTCE`、`NAME`、`ALAND`、`AWATER`、`geometry`
   - BG：`GEOID`（12 位）、`BLKGRPCE` 等、`geometry`
6) **文件格式与体量**：Shapefile(zip)；记录下载包大小。

---

## P0-2 宏观约束（ACS Summary / Decennial Summary）

### 数据源 P0-2a：ACS 5-year Detailed Tables（用于 tract/BG 控制量）
1) **数据集名称**：ACS 5-Year Data（Detailed Tables via API / data.census.gov）  
2) **覆盖范围/空间层级**：**Detailed Tables 可到 block group**；tract/BG 均可用。  
3) **推荐年份**：
   - 当前落地建议：先用 **2019–2023 ACS 5-year**（稳定可用）  
   - 待更新：**2020–2024 ACS 5-year** 预计在 **2026-01-29** 发布（发布后可整体升级口径）  
4) **下载入口（URL）**：
   - ACS 5-year API 页面：https://www.census.gov/data/developers/data-sets/acs-5year.html
   - data.census.gov（表格检索/导出）：https://data.census.gov/
5) **许可与使用限制**：联邦公开统计数据；需在 metadata 中记录表号、年份、下载日期、抽取参数。  
6) **控制量表（建议最小集，输出成 `marginals_long`）**：
   - 人口：年龄×性别  
     - `B01001` Sex by Age
   - 家庭：家庭规模/类型  
     - `B11016` Household Type by Household Size（推荐）  
     - （可选）`B11001` Household Type
   - 收入：家庭收入分箱  
     - `B19001` Household Income in the Past 12 Months
   - 就业：就业状态  
     - `B23025` Employment Status for the Population 16 Years and Over
   - 通勤：通勤方式、通勤时间分箱（强建议）  
     - `B08301` Means of Transportation to Work  
     - `B08303` Travel Time to Work
   - （可选）贫困  
     - `B17001` Poverty Status by Sex by Age
7) **Partner 输出要求**：
   - 每张表：提供 `table_id`、变量列表（含 label）、可用空间层级（tract/BG）、年份、抽取方法（API 参数或导出步骤）
   - 建议输出：
     - `acs_<year>_<table_id>_bg.parquet`
     - `acs_<year>_<table_id>_tract.parquet`
     - `acs_<year>_variable_dictionary.csv`

> 附：API 抽取提示（示例，Partner 可按需修改）
- Block group（全 Wayne County）：`for=block group:*&in=state:26 county:163`  
- Tract（全 Wayne County）：`for=tract:*&in=state:26 county:163`  
- 再用 Detroit boundary（place）做空间裁剪，或用 tract/BG 的几何与 boundary 叠加裁剪。

---

### 数据源 P0-2b：2020 Decennial（用于“硬约束/校准”，可选但推荐）
> 目的：与 ACS 估计值做一致性校验；必要时可作为“总量锚”。

1) **数据集名称**：2020 Census Redistricting File (P.L. 94-171) Summary File / Shapefiles  
2) **覆盖范围/分辨率**：全美；通常可到 block（非常细）  
3) **下载入口（URL）**：
   - 2020 TIGER/Line + PL94-171 信息页：https://www.census.gov/geographies/mapping-files/2020/geo/tiger-line-file.html
4) **许可与使用限制**：公开；在 metadata 里记录版本/下载日期。  
5) **字段字典**：Partner 需附官方技术文档/record layout 链接（或下载包内说明文件）。  
6) **格式与体量**：体量较大；建议仅抽 Michigan/Wayne/Detroit 相关部分并记录抽取方式。

---

## P0-3 微观种子样本（PUMS）

### 数据源 P0-3a：ACS PUMS（household + person）
1) **数据集名称**：ACS PUMS（1-year + 5-year；建议与 ACS 5-year 控制量对齐）  
2) **地理分辨率**：最细到 **PUMA**（不直接到 tract/BG）  
3) **下载入口（URL）**：
   - PUMS access 页：https://www.census.gov/programs-surveys/acs/microdata/access.html
   - PUMS 总览页：https://www.census.gov/programs-surveys/acs/microdata.html
   - FTP 入口说明：https://www.census.gov/programs-surveys/acs/data/data-via-ftp.html
   - FTP（示例：2019 5-year 目录，会有各州压缩包）：https://www2.census.gov/programs-surveys/acs/data/pums/2019/5-Year/
4) **关键字段（最小集）**：
   - Person：`AGEP`, `SEX`, `PWGTP`（人权重），就业/通勤相关字段（可选）
   - Household/Housing：`HHSIZE`（或派生）、`HINCP`（或收入相关）、`WGTP`（户权重）
   - 注意：通胀调整字段（如 `ADJINC`/`ADJHSG`）若使用必须在 metadata 记录
5) **字段字典/数据字典**：
   - PUMS Documentation 总页：https://www.census.gov/programs-surveys/acs/microdata/documentation.html
   - 2019–2023 ACS 5-year PUMS Data Dictionary（PDF）：https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2019-2023.pdf
6) **许可与使用限制**：公开样本；须遵守不识别个体的使用原则（在 metadata 写明用途与隐私合规说明）。  
7) **文件格式与体量**：州级压缩包可能较大；Partner 回传需写明“只下载 Michigan 还是全国？下载包大小？解压后大小？”

---

## P0-4 PUMA 边界与 PUMA↔Tract crosswalk（让 PUMS 可对齐 tract/BG 控制量）

### 数据源 P0-4a：PUMA20 边界（TIGER/Line）
1) **数据集名称**：TIGER/Line – 2020 PUMA（PUMA20）  
2) **下载入口（URL）**：
   - TIGER/Line 总入口：https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
   - FTP（示例，按年）：https://www2.census.gov/geo/tiger/TIGER2025/PUMA20/
3) **关键字段**：`GEOID`/`GEOID20`（依版本）、`NAME`/`NAMELSAD`、`geometry`  
4) **备注**：需要与 PUMS 年份的 PUMA 口径一致（通常用 2020 PUMA）。

---

### 数据源 P0-4b：Census Tract → PUMA Relationship File（官方）
1) **数据集名称**：Census Tract to PUMA Relationship Files（2020 boundaries）  
2) **下载入口（URL）**：
   - PUMAs 官方说明页（含下载链接）：https://www.census.gov/programs-surveys/geography/guidance/geo-areas/pumas.html
   - Relationship Files（2020 系列总页）：https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.2020.html
3) **用途**：把 tract 归属到 PUMA（或做 tract→PUMA 权重映射）；再结合 tract/BG 控制量约束生成。  
4) **字段字典**：Partner 附上官方 record layout/说明链接或下载包内说明文件。

---

### 数据源 P0-4c（可选增强）：NHGIS Geographic Crosswalks（更适合跨年统一口径）
1) **数据集名称**：IPUMS NHGIS Geographic Crosswalks  
2) **下载入口（URL）**：https://www.nhgis.org/geographic-crosswalks  
3) **备注**：通常需要注册/同意条款；适合做时序或地理口径变化时的高质量映射。  
4) **Partner 回传**：说明 crosswalk 类型、权重口径、限制条款。

---

# P1（强建议）——显著提升空间锚定与可解释性（Residential 判定、容量先验、就业约束）

## P1-1 Parcel / 分区 / 用地（Detroit 官方）
> 用途：更稳的 residential 判定、估计居住容量、解释建成环境差异。

### 数据源 P1-1a：Parcels (Current)（Assessor）
1) **数据集名称**：Parcels (Current) – City of Detroit Office of the Assessor  
2) **覆盖范围/分辨率**：Detroit 全市 parcel polygon  
3) **下载入口（URL）**：
   - ODP about 页：https://data.detroitmi.gov/maps/detroitmi%3A%3Aparcels-current-1/about
   - ODP Explore（可看表）：https://data.detroitmi.gov/datasets/3c784c118e5c4083b37038e9b38573df_0/explore?showTable=true
   - DetroitData 镜像页（便于记录 license）：https://detroitdata.org/dataset/parcels
   - **ArcGIS FeatureServer（推荐给工程用，可分页拉全量）**：

```text
https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/Parcels_Current/FeatureServer/0
```

4) **许可与使用限制**：常见显示 “No License Provided / Request permission to use”；另需遵循 ODP 免责声明/条款（Partner 回传时必须截图或摘录关键句到 metadata）。  
5) **关键字段（建议至少保留/核验是否存在）**：
   - 标识：`parcel_number`/`parcel_id`（实际字段名以表为准）
   - 评估值/交易（用于房价代理）：`assessed_value`、`taxable_value`、`sale_price`、`sale_date`
   - 地址：`address`、`zip`
   - 用途/分类：`use_code`、`use_code_desc`、`property_class(_desc)`、`zoning`
   - 容量/属性：`num_bldgs`、`total_floor_area`/`total_square_footage`、`year_built`、`is_improved`
6) **格式与体量**：Feature layer 导出（GeoJSON/GPKG/Shapefile）；记录导出格式与大小。

**工程下载代码（推荐｜可断点续跑，chunked GeoJSON）**

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
python tools/detroit_fetch_public_data.py parcels-detroit \
  --out_root "$RAW_ROOT/synthetic_city/data" \
  --out_fields "parcel_number,assessed_value,taxable_value,sale_price,sale_date"

# 输出目录：$RAW_ROOT/synthetic_city/data/detroit/raw/parcels/detroit_parcels_current/
```

---

### 数据源 P1-1b：Zoning（分区）
1) **数据集名称**：Zoning – City of Detroit（CPC 更新，2021-04 口径说明）  
2) **覆盖范围/分辨率**：Detroit 全市 zoning polygon  
3) **下载入口（URL）**：
   - ODP about：https://data.detroitmi.gov/datasets/detroitmi%3A%3Azoning/about
   - DetroitData（含 license 字段）：https://detroitdata.org/dataset/zoning
4) **许可与使用限制**：同上（No License Provided / Request permission to use + ODP 条款/免责声明）  
5) **关键字段**：`zoning`/`ZONING_REV`、`ZDESCR`（描述）、`geometry`（以实际字段为准）  
6) **格式与体量**：Feature layer 导出；记录导出方式、大小与字段字典来源。

---

### 数据源 P1-1c（可选）：Master Plan / Future General Land Use
1) **数据集名称**：Current Master Plan Future General Land Use（City of Detroit ODP）  
2) **下载入口（URL）**：
   - ODP about：https://data.detroitmi.gov/datasets/current-master-plan-future-general-land-use/about

---

### 数据源 P1-1d（备选/补充）：Wayne County Annual Assessment Data（County 级评估数据）
> 说明：Wayne County 的年度 assessment 包通常是表格压缩包；若要用于空间 join，需要再配套 parcel 几何（且 join key 必须一致）。  
> 我们 v0 更推荐用 Detroit 的 Parcels_Current（自带 geometry + assessed_value），这条作为备用证据链。

1) **数据集名称**：Wayne County Annual Assessment Data (2025)  
2) **入口页**：https://www.waynecountymi.gov/Government/Departments/Management-Budget/Assessment-Equalization/Annual-Assessment-Data  
3) **直接下载（2025）**：

```text
Detroit: https://www.waynecountymi.gov/files/assets/mainsite/v/1/management-amp-budget/documents/assessment-data/2025/detroit.zip
Full county: https://www.waynecountymi.gov/files/assets/mainsite/v/1/management-amp-budget/documents/assessment-data/2025/2025-82-wayne-county-foia.zip
```

4) **工程下载代码**：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
python tools/detroit_fetch_public_data.py wayne-assessment-2025 --scope detroit --out_root "$RAW_ROOT/synthetic_city/data"
```
3) **用途**：对 POI/parcel 的 land-use 分类做补充与解释（尤其是 residential/industrial/commercial 区）。  
4) **Partner 回传**：分类字段字典、更新日期、许可限制。

---

## P1-2 就业分布与通勤 OD（LEHD LODES）
1) **数据集名称**：LEHD LODES（WAC/RAC/OD；LODES8）  
2) **覆盖范围/分辨率**：US；通常细到 census block  
3) **下载入口（URL）**：
   - LEHD 数据入口：https://lehd.ces.census.gov/data/
   - LODES 下载工具页：https://lehd.ces.census.gov/php/inc_lodesDownloadTool.php
   - LODES 技术文档（目录结构/字段）：https://lehd.ces.census.gov/data/lodes/LODES8/LODESTechDoc8.3.pdf
4) **许可与使用限制**：公开；注意官方建议不要把不同 release 混用（Partner 回传必须记录 release 版本）。  
5) **关键字段/文件类型**：
   - WAC（workplace area characteristics）
   - RAC（residence area characteristics）
   - OD（origin-destination flows）
6) **输出建议**：
   - `lodes_<release>_<year>_wac_mi.parquet`
   - `lodes_<release>_<year>_rac_mi.parquet`
   - `lodes_<release>_<year>_od_mi.parquet`
   - `lodes_<release>_techdoc.pdf`

---

# P2（可选增强）——时空动态与更强验证

## P2-1 动态人口/移动性（开放数据优先）
### 数据源 P2-1a：ACS 发布节奏（便于你决定何时升级到 2020–2024）
- 2026 ACS Program Updates（提示 2020–2024 5-year 预计 2026-01-29 发布）  
  - URL：https://www.census.gov/programs-surveys/acs/news/updates/2026.html

### 数据源 P2-1b：BTS “Trips by Distance”（县级/日尺度，适合 Wayne County 验证）
- URL：https://www.bts.gov/browse-statistical-products-and-data/covid-related/distribution-trips-distance-national-state-and

### 数据源 P2-1c：Meta / HDX Movement Range Maps（疫情期间日尺度，弱真值参照）
- URL：https://data.humdata.org/dataset/movement-range-maps

---

## P2-2 交通供给数据（解释通勤方式/可达性）
### 数据源 P2-2a：路网（OSM）
1) **数据集名称**：OpenStreetMap（路网/POI/部分 landuse 标签）  
2) **下载入口（URL）**：
   - Geofabrik Michigan Extract：https://download.geofabrik.de/north-america/us/michigan.html
   - OSM 版权与许可摘要：https://www.openstreetmap.org/copyright
3) **许可**：ODbL（share-alike）；派生数据库/公开发布需遵循 ODbL（在 metadata 中写清楚）。  

### 数据源 P2-2b：GTFS（Detroit 公交）
> 你可以先用 Detroit ODP 提供的 GTFS 文档/下载作为统一入口（便于记录许可与版本）。

- DDOT GTFS file（ODP 文档页）：https://data.detroitmi.gov/documents/1de3fec8cc894fdbbc03c5d31bca32d4  
- Detroit People Mover GTFS（ODP 文档页）：https://data.detroitmi.gov/documents/08c451e3814143dd804539a4da2c3527

Partner 回传需要补齐：
- feed 的 zip 下载链接（若文档页内给出）
- GTFS 生效日期范围（calendar.txt / calendar_dates.txt）
- 是否有 GTFS-RT（实时）及其入口与限制

---

## P2-3 夜间灯光 / 遥感（空间强度代理）
### 数据源 P2-3a：NASA Black Marble（daily/monthly/yearly 夜光）
- Black Marble 主页：https://blackmarble.gsfc.nasa.gov/
- LAADS 产品域（夜光说明）：https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/science-domain/nighttime-lights/
- 示例产品（年尺度 VNP46A4）：https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A4/

### 数据源 P2-3b：VIIRS 夜光（工具链友好版本：Google Earth Engine 数据集）
- NOAA VIIRS DNB Monthly（GEE Catalog）：https://developers.google.com/earth-engine/datasets/catalog/NOAA_VIIRS_DNB_MONTHLY_V1_VCMCFG

Partner 回传需要补齐：
- 产品版本号、时间范围、空间分辨率、投影与 nodata
- 下载方式（直接下载 / GEE 导出 / 其他平台）

---

# P2-4 GUS-3D（全球 500m 网格 3D 城市形态）——供宏观先验/背景引用
> 备注：该数据是 **500m 网格**层级的建筑结构统计（不是建筑物 footprint）。对 Detroit building-level portrait 不能直接替代 GBA，但可用于：  
> - 宏观背景叙事（城市形态与不平等）  
> - coarse-level 的 sanity check / prior（例如高度/体量与夜光的一致性）  

1) **论文（来源）**：Global Mapping of 3D Urban Structure Inequality: Unveiling the Urban Form’s Role in Unequal Outcomes. *The Engineering* (2024). DOI: `10.1016/j.eng.2024.01.025`  
2) **数据内容（文中描述）**：以全球 urban clusters 为单位，在 500m 网格尺度提供 `average building height`、`building volume`、`building footprint`、`building coverage ratio (BCR)`、`building volume density (BVD)` 等建筑结构指标（文中以 2015 年为例）。  
3) **开放数据入口（Figshare，需核验可达性与文件清单）**：

```text
https://figshare.com/articles/dataset/Global_Mapping_of_Three-Dimensional_3D_Urban_Structures_Inequality_Unveiling_the_Urban_Form_s_Role_in_Unequal_Outcomes/21507537
```

4) **关键点（回应 reviewer 可能的追问）**：该工作属于“多源遥感 + 机器学习”反演网格化 3D 城市形态；文中明确用带高度信息的建筑 footprint 数据作为参考/校准来源之一，但最终公开产品是网格化结构指标，而非逐栋建筑几何。

---

# B) Partner 回传格式（请按模板整理）

对每个数据源，请提供：

1) **数据集名称**（含年份/版本）  
2) **覆盖范围**（Detroit city / Wayne / MI / US）与**空间分辨率**（tract/BG/parcel/block/网格）  
3) **时间分辨率**（静态/日/小时）  
4) **下载入口（URL）** + 获取方式（直接下载/申请/付费/合作）  
5) **许可与使用限制**（是否可用于基金申报、是否可公开发布派生结果、是否可商用）  
6) **字段字典/数据字典**（关键字段与含义；附链接或随包文件）  
7) **文件格式与大致体量**（MB/GB）+ 处理记录（裁剪/投影/字段清洗）

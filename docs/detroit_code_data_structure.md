# 底特律（Detroit）案例落地蓝图：代码结构与数据结构（Draft）

> 目的：把“基于扩散的多层次人口生成方法”在底特律落地所需的**代码目录结构**与**数据目录/表结构**先定义清楚，便于后续按步骤实现与迭代。  
> 说明：本文档是“构思方案”版本，包含若干待确认项；确认后再进入实现与数据落地。

---

## 0. 范围与输出（我们到底要产出什么）

**目标城市**：Detroit（建议先以 Detroit city boundary 为试点；后续可扩展到 Wayne County / Metro Detroit）。  
**目标粒度**：建筑物尺度（building-level）的人口画像；可选扩展到户—人层级与时段动态。  

**核心输出（建议最小可用集）**
- `persons`：合成人口个体表（人口属性）。
- `households`（可选但强建议）：合成家庭表（户属性），并与 persons 绑定以保证户—人一致性。
- `assignments`：个体/家庭 → 建筑物（或地块/格网）空间落点。
- `time_profiles`（可选）：日间/夜间或小时级的人口位置/活动强度分配。
- `metrics`：统计一致性、空间真实性、时间动态一致性三层验证指标与可视化产物。

---

## 1. 需要先确认的关键问题（避免后期返工）

1) **研究区域**：仅 Detroit city？还是扩到 Wayne County / Metro Detroit？  
2) **空间锚定对象**：建筑物 footprints（推荐）还是 parcel / 100m 网格？  
3) **时间维度**：仅静态常住人口？还是日间/夜间？还是小时级（24h）？  
4) **属性集合**：底特律首版建议先做“年龄×性别×收入×职业/就业状态×家庭规模×通勤方式（可选）”。是否需要教育、种族、车拥有量等？  
5) **种子微观样本**：是否使用 ACS PUMS（美国可公开获取）作为 household/person seed？若不用，替代种子是什么？  
6) **宏观约束层级**：用 tract 还是 block group（BG）做主要控制量？（BG 细但噪声更大、表更稀疏）  
7) **建筑物容量先验**：是否有 floors/height/landuse/parcel usage 等字段？若没有，容量只用 footprint area + POI 代理是否可接受？  
8) **动态验证参照数据**：是否有（或能合作拿到）信令/平台热力/OD？否则时间层验证只能做弱证据链。

> 建议默认决策（除非你另有要求）：Detroit city + building footprint；静态 + 日/夜两时段；PUMS 作种子；控制量以 tract/BG 混合（主 tract，BG 用于局部细化）。

---

## 2. 数据目录结构（Data Structure）

采用“城市维度隔离 + raw/interim/processed/outputs”四层结构，避免不同城市/不同run相互污染。

工作站落盘口径（与 `docs/WORKSTATION_GUIDE.md` 一致）：仓库只放代码与文档；**大数据不进 git**。实践上建议把仓库内的 `data/` 做成软链接，指向外置目录（例如 `$RAW_ROOT/synthetic_city/data`），从而保持代码引用路径稳定，同时避免误把大文件提交到 GitHub（本仓库已在 `.gitignore` 忽略 `data/`）。

### 2.1 目录树（建议）

```text
data/
  detroit/
    raw/                      # 原始数据：不改动，仅记录来源与版本
      geo/                    # 行政边界/地理单元（TIGER 等）
      census/                 # ACS Summary / Decennial 等统计表
      pums/                   # ACS PUMS household/person
      buildings/              # building footprints / height / landuse
      parcels/                # parcel（可选）
      poi/                    # POI（OSM/商业/政府开放）
      mobility/               # LBS/信令/热力/OD（可选）
      transport/              # 路网/GTFS/可达性（可选）
      README.md               # 每类数据的获取方式/版本/许可

    interim/                  # 中间结果：清洗但未统一schema
      geo/
      buildings/
      pums/
      poi/

    processed/                # 统一schema后的“可建模数据”
      geo_units/              # tract/BG/city boundary（统一 CRS 与字段）
      buildings/              # building 主表 + 几何 + 特征
      crosswalks/             # building→BG/tract；POI→building（可选）
      pums/                   # household/person 清洗后表
      marginals/              # 控制量（长表/宽表）
      rules/                  # 规则与结构性零（yaml/json）
      features/               # 条件向量、空间先验、时段特征等

    outputs/
      runs/
        2026-01-17_detroit_v1_<gitsha>/   # 每次生成一个 run 目录
          config.yaml
          metadata.json
          synthetic/
            persons.parquet
            households.parquet
            assignments_building.parquet
            time_profiles.parquet
          metrics/
            stats_metrics.json
            spatial_metrics.json
            temporal_metrics.json
            figures/
          logs/
```

### 2.2 格式建议

- 表格：优先 `parquet`（必要时 `csv`）；地理：优先 `GeoParquet`（或 `gpkg`）。  
- 大文件命名：`<dataset>_<year>_<geolevel>.<ext>`；所有文件使用 UTF-8。  
- 坐标系：存储可用 `EPSG:4326`；涉及面积/距离计算时统一投影到 Detroit 合适的投影（如 UTM 17N，或地方投影），并在 `metadata.json` 记录。

### 2.3 wsA 拉取公开数据（P0 最小集）

> 目标：把 `raw/` 的公共数据先落盘，确保后续 `processed/` 的裁剪/叠加/约束构建都可复现。

推荐把仓库内 `data/` 软链到外置目录（不进 git）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
mkdir -p "$RAW_ROOT/synthetic_city/data"
ln -snf "$RAW_ROOT/synthetic_city/data" data
```

拉取 TIGER/ACS/PUMS/OSM（以及注册 SafeGraph 现有目录）：

```bash
python tools/detroit_fetch_public_data.py tiger --out_root "$RAW_ROOT/synthetic_city/data"
python tools/detroit_fetch_public_data.py acs --out_root "$RAW_ROOT/synthetic_city/data" --acs_year 2023
python tools/detroit_fetch_public_data.py pums --out_root "$RAW_ROOT/synthetic_city/data" --pums_year 2023
python tools/detroit_fetch_public_data.py osm --out_root "$RAW_ROOT/synthetic_city/data" --region michigan
python tools/detroit_fetch_public_data.py safegraph --out_root "$RAW_ROOT/synthetic_city/data" \
  --safegraph_dir "$RAW_ROOT/safegraph/safegraph_unzip"
```

---

## 3. 关键数据表（Data Schema）

> 原则：先定义“最小可用字段”，后续按需求扩展；所有表必须有稳定主键与来源字段。

### 3.1 地理单元（`processed/geo_units/`）

**`geo_units.parquet`（长表，统一管理 city/tract/BG 等）**
- `geo_level`：`city|county|tract|bg|block`（可扩展）
- `geoid`：对应 US Census GEOID（字符串）
- `name`：可读名称（可选）
- `geometry`：GeoParquet geometry
- `area_m2`：可选（投影计算）

### 3.2 建筑物（`processed/buildings/`）

**`buildings.parquet`**
- `bldg_id`：建筑物唯一ID（建议 hash 或连续整数）
- `geometry`
- `centroid_lon` `centroid_lat`
- `footprint_area_m2`
- `height_m` / `floors`（若有）
- `landuse`（标准化类别：residential/commercial/industrial/mixed/unknown）
- `bg_geoid` `tract_geoid`：空间落在哪个 BG/tract（由空间叠加得到）
- `res_capacity`：居住容量先验（可为空；后续由模型估计）
- `work_capacity`：就业容量先验（可选）
- `data_source`：OSM/Microsoft/City open data 等

**推荐数据源：GlobalBuildingAtlas（GBA）LoD1（你已在工作站下载北美 tiles）**
- Detroit 预计 tile：`w085_n45_w080_n40.geojson`（需以实际文件为准）
- 工作站路径（你提供的现状）：`/home/jinlin/DATASET/LoD1/northamerica/`
- 最小核验（请在工作站执行并把输出贴回）：  
  - `ls /home/jinlin/DATASET/LoD1/northamerica | rg "w085_n45_w080_n40\\.geojson" || true`  
  - `TILE=/home/jinlin/DATASET/LoD1/northamerica/w085_n45_w080_n40.geojson; ls -lh "$TILE"`  
  - `rg -n -m 1 "\"height\"" "$TILE" || true`（只为快速判断是否含高度字段；最终以 schema 为准）

**已核验（你回传的样例 feature）**
- `properties`：`source`（示例值 `ms`）、`id`、`height`、`var`、`region`
- `geometry`：`Polygon`，坐标为 2D 平面坐标；数值量级与 Web Mercator（`EPSG:3857`）一致（需在 `metadata.md` 明确 CRS）

> 处理建议（后续实现时）：先按 Detroit 边界裁剪 LoD1 tile，再统一投影计算 `footprint_area_m2` 与派生 `volume`/`capacity` 先验，最后写入 `processed/buildings/buildings.parquet`（或 GeoParquet）。

### 3.3 种子微观样本（`processed/pums/`）

> 若使用 ACS PUMS：建议保留原字段 + 我们的标准化字段（suffix `_std`），并记录 `year`、`state`、`puma`。

**`pums_households.parquet`**
- `hh_id_src`：原始 household id（或行号hash）
- `puma`：PUMA code
- `hh_size`
- `hh_income`（原值或分箱）
- `tenure`（自有/租住等，可选）
- `vehicles`（可选）
- `weight`：PUMS 权重（若使用）
- `year` `source`

**`pums_persons.parquet`**
- `person_id_src`
- `hh_id_src`
- `age`
- `sex`
- `education`（可选）
- `employment_status` / `occupation`（可选）
- `commute_mode`（可选）
- `weight`
- `year` `source`

### 3.4 宏观约束（`processed/marginals/`）

建议使用“长表约束”，便于统一多张 ACS 表、不同地理层级与不同变量：

**`marginals_long.parquet`**
- `geo_level`：`tract|bg|...`
- `geoid`
- `constraint_name`：如 `age_sex`、`hh_size`、`income_bin`、`occupation`、`day_night_pop`（可选）
- `category`：类别编码（如 `age_0_4_male`）
- `target`：目标人数/户数
- `year`
- `source`：ACS/Decennial/自定义估计

### 3.5 规则与结构性零（`processed/rules/`）

**`rules.yaml`（示例）**
- 年龄–就业逻辑：`age<16 => employment_status!=full_time`
- 驾照与年龄：`age<16 => has_driver_license=false`（若使用）
- 家庭规模一致性：`household.hh_size == count(persons in household)`
- 结构性零组合清单（可选）：用规则表达优先于硬编码列表

---

## 4. 代码结构（Code Structure）

> 当前仓库以申报文档为主，代码可以从“最小可执行管线”起步：数据准备 → 约束构建 → 训练/采样 → 验证与输出。

### 4.1 目录树（建议）

```text
src/
  synthpop/
    __init__.py
    cli.py                   # 统一入口：prepare / train / sample / validate
    config.py                # 读取/校验 YAML 配置

    data/
      ingest.py              # 读 raw → interim
      standardize.py         # interim → processed（统一schema）
      pums.py                # PUMS 清洗与编码
      census.py              # ACS/Decennial 约束表构建

    geo/
      crs.py
      overlay.py             # building↔BG/tract 空间叠加
      boundaries.py

    features/
      building_features.py   # capacity/landuse/POI 聚合
      condition_vectors.py   # 生成条件向量（geo+space+time）

    constraints/
      marginals.py           # 约束接口：读取/汇总/对齐
      rules.py               # 规则校验与修复（投影）

    model/
      diffusion_tabular.py   # 表格扩散核心（混合类型）
      schedule.py

    sampling/
      generate.py            # 采样：条件注入 + 规则投影

    validation/
      stats.py               # 边际/二阶关联一致性
      spatial.py             # 建筑物/热点/用地一致性
      temporal.py            # 日/夜或小时曲线一致性（若有数据）

configs/
  detroit.yaml
scripts/
  detroit_pipeline.sh        # 可选：一键串联
docs/
  detroit_code_data_structure.md
```

### 4.2 配置文件（`configs/detroit.yaml`）建议字段

- `study_area`：边界文件路径/行政区筛选规则  
- `geo_levels`：主控制层级（tract/BG）  
- `attributes`：个体/家庭属性集合与分箱定义  
- `constraints`：`marginals_long.parquet` 路径 + 使用哪些约束  
- `buildings`：building 数据路径 + capacity 估计策略  
- `training`：样本数、batch、epoch、seed、GPU/CPU  
- `sampling`：生成规模、条件强度、规则投影开关  
- `validation`：要跑哪些指标与外部参照数据路径  

---

## 5. 最小可执行工作流（MVP Pipeline）

1) **准备 Detroit 数据**：`raw/` → `processed/`（统一 CRS/字段）  
2) **构建控制量（marginals）与规则（rules）**：形成可被模型调用的约束接口  
3) **训练表格扩散模型**：以种子样本学习联合分布结构（可先无条件/弱条件）  
4) **条件化采样 + 规则投影**：输出 persons/households + building assignment  
5) **三层验证**：统计一致性 → 空间真实性 → 时间动态（若有参照）  

---

## 6. 下一步：请你确认的最小信息

为了把本文档从 Draft 变成可直接开工的“任务书”，请你回复以下最小信息：

1) Detroit 范围：`city` / `Wayne County` / `metro`？  
2) 首版要不要做 households（户—人层级）？  
3) 时间维度：`静态` / `日夜` / `24小时`？  
4) 你们手里已有的数据清单（尤其 buildings、POI、mobility/OD）。  

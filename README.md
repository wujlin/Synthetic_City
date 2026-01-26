# Synthetic_city

本仓库用于“融合多源大数据的生成式建筑物尺度时空化人口画像方法研究”的基金申报写作与方法落地准备（基于扩散的多层次人口生成方法）。

## 关键文档

- 申报正文：`NSFC/基金申报文稿.md`
- 扩散方法分稿：`NSFC/生成扩散.md`
- 文献证据链（v0.3）：`docs/literature_review_popdyn_partner_v0.3_2026-01-17.md`
- Detroit 落地蓝图（代码/数据结构）：`docs/detroit_code_data_structure.md`
- 合成人口代码架构（方法路线→模块映射）：`docs/synthpop_architecture.md`
- Detroit 数据需求清单（给检索用）：`docs/detroit_data_request.md`
- Detroit 数据检索结果（来源证据链）：`docs/DATA_SEARCH.md`
- 工作站运行与数据落盘口径：`docs/WORKSTATION_GUIDE.md`

## 建议目录约定

```text
NSFC/    # 申报文稿与分稿
docs/    # 文献梳理、落地蓝图、数据需求等支撑材料
data/    # 城市数据（默认不入库，见 .gitignore）
```

## Detroit 公开数据下载（wsA）

推荐把仓库内 `data/` 软链到外置目录（不进 git）：

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
mkdir -p "$RAW_ROOT/synthetic_city/data"
ln -snf "$RAW_ROOT/synthetic_city/data" data
```

拉取 P0 公共数据（TIGER/ACS/PUMS/OSM）并注册 SafeGraph（已存在于 wsA）：

```bash
python tools/detroit_fetch_public_data.py tiger --out_root "$RAW_ROOT/synthetic_city/data"
python tools/detroit_fetch_public_data.py acs --out_root "$RAW_ROOT/synthetic_city/data" --acs_year 2023
python tools/detroit_fetch_public_data.py pums --out_root "$RAW_ROOT/synthetic_city/data" --pums_year 2023
python tools/detroit_fetch_public_data.py osm --out_root "$RAW_ROOT/synthetic_city/data" --region michigan
python tools/detroit_fetch_public_data.py parcels-detroit --out_root "$RAW_ROOT/synthetic_city/data" \
  --out_fields "parcel_number,assessed_value,taxable_value,sale_price,sale_date"
python tools/detroit_fetch_public_data.py safegraph --out_root "$RAW_ROOT/synthetic_city/data" \
  --safegraph_dir "$RAW_ROOT/safegraph/safegraph_unzip"
```

注册完成后 SafeGraph 的统一入口：
- `$RAW_ROOT/synthetic_city/data/detroit/raw/poi/safegraph/safegraph_unzip/`
- `$RAW_ROOT/synthetic_city/data/detroit/raw/poi/safegraph/safegraph.metadata.json`

## 代码入口（Detroit v0）

打印项目/数据根目录解析（便于排查路径口径）：

```bash
python -m src.synthpop paths
```

查看 Detroit 原始数据是否到位（只做路径存在性检查，不做重验证）：

```bash
python -m src.synthpop detroit status
```

初始化 Detroit 目录结构（只创建目录，不移动/删除文件）：

```bash
python -m src.synthpop detroit init-dirs
```

将 4 个 TIGER zip（`tl_2023_26_{place,tract,bg,puma20}.zip`）落到标准目录（默认 copy，不会删除源文件）：

```bash
python -m src.synthpop detroit stage-tiger --tiger_year 2023 --src_dir data
```

将 PUMS zip（常见命名：`psam_p26.zip` 或 `csv_pmi.zip`）落到标准目录（默认 copy，不会删除源文件）：

```bash
python -m src.synthpop detroit stage-pums --pums_year 2023 --pums_period 5-Year --src_dir data
```

## 最小技术验证（TabDDPM + PUMS）

在 wsA 上用 PUMS person 子集跑一个 PoC（验证“混合类型扩散 + 条件化”可训练可采样）：

```bash
python tools/poc_tabddpm_pums.py --data_root "$RAW_ROOT/synthetic_city/data" --pums_year 2023 --epochs 5 --timesteps 200
```

也可用 CLI wrapper（等价调用 tools 脚本）：

```bash
python -m src.synthpop detroit poc-train --data_root "$RAW_ROOT/synthetic_city/data"
python -m src.synthpop detroit poc-sample --data_root "$RAW_ROOT/synthetic_city/data"
```

## 建筑条件注入 PoC（Detroit｜building-level portrait）

PI review 强调：建筑落点不能只是事后分配；应把建筑作为条件注入生成过程。对应的机制 PoC：

1) 从 GBA LoD1 + TIGER 生成 Detroit 建筑特征表（`bldg_id+puma+tract+area/height/cap_proxy/dist_cbd+centroid`）
2) 叠加 Wayne County parcel assessment，得到 `price_per_sqft` 并在 tract 内分位数分档 `price_tier(Q1..Q5)`（作为建筑级收入代理）
3) 训练条件扩散：`cond = [PUMA one-hot, building_feature(price_tier + 辅助特征)]`
3) 采样输出 `samples_building.csv` 并聚合得到 `building_portrait.csv`（pop/age/income）

```bash
export DATA_DIR="$RAW_ROOT/synthetic_city/data"

# 1) 生成建筑特征表（需要安装 geopandas/pyproj/shapely）
python tools/prepare_detroit_buildings_gba.py \
  --gba_tile "/home/jinlin/DATASET/LoD1/northamerica/w085_n45_w080_n40.geojson" \
  --tiger_place_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_place.zip" \
  --tiger_puma_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_puma20.zip" \
  --tiger_tract_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_tract.zip" \
  --out_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features.csv"

# 2) join parcel assessment → price_per_sqft + price_tier（parcels_path 由 partner 下载落盘）
python tools/join_detroit_buildings_parcel_assessment.py \
  --buildings_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features.csv" \
  --parcels_path "$DATA_DIR/detroit/raw/parcels/wayne_county/parcel_assessment.gpkg" \
  --group_for_tier tract --n_tiers 5 \
  --out_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features_price.csv"

# 3) building-conditioned PoC（会输出 samples_building.csv / building_portrait.csv）
python tools/poc_tabddpm_pums_buildingcond.py \
  --mode train-sample \
  --data_root "$DATA_DIR" \
  --buildings_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features_price.csv" \
  --pairing price_tier --n_tiers 5 \
  --epochs 5 --timesteps 200 --n_samples 50000 --device cuda
```

## GitHub 首次同步（常见报错修复）

若 `git push -u origin main` 报错 `src refspec main does not match any`，通常是**还没有任何 commit**。

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

若 HTTPS 认证失败：GitHub 通常需要使用 **Personal Access Token**（PAT）作为密码。

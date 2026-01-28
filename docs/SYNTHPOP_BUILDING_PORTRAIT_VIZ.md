# SynthPop：建筑物尺度人口画像可视化（Detroit｜v0.1）

## Thought（核心结论）

我们要可视化的不是“生成了多少人”，而是回答一个更硬的问题：

> **建筑物尺度的人口画像是否是“在生成过程中”被空间条件塑形出来的？**

因此可视化必须直接服务于方法主线：  
**\(P(\text{attrs}, \text{bldg} \mid \text{macro constraints}, \text{building features})\)** 的联合生成，而不是“先生成人再事后分配建筑”的后处理。

---

## Layer 0：输入与数据契约（先把口径钉死）

本阶段的最小输入是两张表：

1) **建筑特征表（含房价代理）**（由 GBA LoD1 + TIGER + Wayne County parcel assessment 生成）
- `bldg_id`：建筑唯一ID
- `puma`：建筑所属 PUMA（用于宏观条件）
- `footprint_area_m2, height_m, cap_proxy, dist_cbd_km`：空间条件特征（辅助）
- `price_per_sqft, price_tier`：**建筑级收入代理（核心维度）**  
  - `price_per_sqft`：parcel assessed value 与建筑面积匹配后的价格密度代理  
  - `price_tier`：在 tract 内的分位数档位（Q1–Q5）
- `centroid_lon, centroid_lat`：用于轻量可视化（点）

2) **合成人口样本**（条件扩散采样输出）
- `bldg_id, PUMA, AGEP, PINCP, SEX, ESR`

对应 PoC 脚本会额外输出：
- `building_portrait.csv`：按 `bldg_id` 聚合的建筑物尺度画像（pop_count/age/income）

---

## Layer 1：建筑物尺度“人口数量”分布（pop_count）

目的不是“热力图好看”，而是检查两件事：

1) **容量驱动是否生效**：`pop_count` 与 `cap_proxy` 是否正相关  
2) **宏观条件是否守恒**：按 PUMA 聚合的总人数分布是否与训练期 `puma_prob` 一致（PoC 用 PUMS 经验分布）

建议产物：
- `fig_pop_count_scatter.png`：`cap_proxy` vs `pop_count`（log尺度）
- `fig_puma_mass_bar.png`：按 PUMA 的人口占比对比（target vs synth）

---

## Layer 2：建筑物尺度“收入”分布（income）

核心问题：空间条件是否改变了生成的属性结构？  
最直接的诊断是：**不同建筑条件下的 income 分布是否出现系统性差异**。

建议产物：
- `fig_income_map.png`：以 `income_p50` 着色的建筑点图（centroid）
- `fig_income_vs_dist.png`：`dist_cbd_km` vs `income_p50`

---

## Layer 3：建筑物尺度“年龄结构”分布（age）

目的：检查“空间条件→人群结构”是否同时作用于另一个关键维度，而不是只在收入上起效。

建议产物：
- `fig_age_map.png`：`age_p50` 着色的建筑点图
- `fig_age_vs_dist.png`：`dist_cbd_km` vs `age_p50`

---

## 在 wsA 上跑通这套最小闭环（命令）

> 说明：以下命令对应的是**机制 PoC**，用于验证“建筑条件注入”是否可行；并非最终 Detroit 全流程。

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export DATA_DIR="$RAW_ROOT/synthetic_city/data"
cd ~/projects/Synthetic_City

# 1) 生成 Detroit 建筑特征表（GBA + TIGER；需要 geopandas/pyproj/shapely）
python tools/prepare_detroit_buildings_gba.py \
  --gba_tile "/home/jinlin/DATASET/LoD1/northamerica/w085_n45_w080_n40.geojson" \
  --tiger_place_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_place.zip" \
  --tiger_puma_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_puma20.zip" \
  --tiger_tract_zip "$DATA_DIR/detroit/raw/geo/tiger/TIGER2023/tl_2023_26_tract.zip" \
  --out_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features.csv"

# 2) 叠加 Parcels_Current（City of Detroit），得到 price_per_sqft + tract 内 price_tier（Q1-Q5）
#    parcels_path 推荐使用 tools/detroit_fetch_public_data.py parcels-detroit 的输出目录
python tools/join_detroit_buildings_parcel_assessment.py \
  --buildings_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features.csv" \
  --parcels_path "$DATA_DIR/detroit/raw/parcels/detroit_parcels_current" \
  --group_for_tier tract --n_tiers 5 \
  --out_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features_price.csv"

# 3) building-conditioned diffusion PoC：输出 samples_building + building_portrait
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$DATA_DIR/detroit/outputs/runs/${RUN_TAG}_poc_tabddpm_pums_buildingcond"

PYTHONUNBUFFERED=1 python -u tools/poc_tabddpm_pums_buildingcond.py \
  --mode train-sample \
  --data_root "$DATA_DIR" \
  --buildings_csv "$DATA_DIR/detroit/processed/buildings/buildings_detroit_features_price.csv" \
  --pairing price_tier --n_tiers 5 \
  --epochs 1000 --batch_size 4096 --timesteps 200 --n_samples 50000 --device cuda \
  --out_dir "$OUT_DIR" \
  |& tee "$OUT_DIR/run.log"

ls -lah "$OUT_DIR" | rg "samples_building|building_portrait|summary|model.pt|encoder.json"
```

## 最小可视化产物（PNG + GeoJSON）

> 目的：把“建筑尺度画像”变成可 review、可进 QGIS/Kepler 的产物，而不是只留在 CSV。

若缺少绘图依赖（常见于纯训练环境），先安装：

```bash
conda install -c conda-forge matplotlib
```

```bash
python tools/viz_building_portrait.py \
  --portrait_csv "$OUT_DIR/building_portrait.csv" \
  --out_dir "$OUT_DIR/viz" \
  --max_points 20000

ls -lah "$OUT_DIR/viz"
ls -lah "$OUT_DIR/viz/figures"
```

产物说明：
- `building_portrait_points.geojson`：建筑点数据（可直接拖进 QGIS/Kepler）
- `figures/map_*`：建筑点图（income/age/pop_count），同时导出 `.png` 与 `.pdf`
- `figures/box_*`：按 `price_tier` 分档的分布图（income 使用 log1p 便于跨档对比），同时导出 `.png` 与 `.pdf`
- `figures/hist_*`：建筑尺度的分布直方图（log1p(income)、log1p(pop)、age），同时导出 `.png` 与 `.pdf`
- `viz_summary.json`：相关性与分档统计（用于 PI 快速 review）

---

## 应用场景（把可视化变成“用得上”）

建筑物尺度的人口画像至少支撑三类应用叙事（从“为什么要做”到“做成能干什么”）：

1) **城市风险与韧性**：热浪/洪涝/污染暴露评估需要建筑尺度的年龄与收入结构（脆弱性画像），而不是 tract 级均值。
2) **公共服务与公平**：学校、医疗、应急避难的服务覆盖，本质是“人在哪 + 是什么人”，建筑尺度画像能把结构性不平等显性化。
3) **与动态证据闭环**（后续）：将合成人口与 POI/轨迹/夜光等动态证据对齐，形成静态—动态一致的数字孪生人口底座。

> 重要：应用不是“画图”，而是验证我们的方法确实在生成过程中把空间条件内生化了。

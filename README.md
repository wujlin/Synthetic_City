# Synthetic_city

本仓库用于“融合多源大数据的生成式建筑物尺度时空化人口画像方法研究”的基金申报写作与方法落地准备（基于扩散的多层次人口生成方法）。

## 关键文档

- 申报正文：`NSFC/基金申报文稿.md`
- 扩散方法分稿：`NSFC/生成扩散.md`
- 文献证据链（v0.3）：`docs/literature_review_popdyn_partner_v0.3_2026-01-17.md`
- Detroit 落地蓝图（代码/数据结构）：`docs/detroit_code_data_structure.md`
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
python tools/detroit_fetch_public_data.py safegraph --out_root "$RAW_ROOT/synthetic_city/data" \
  --safegraph_dir "$RAW_ROOT/safegraph/safegraph_unzip"
```

## GitHub 首次同步（常见报错修复）

若 `git push -u origin main` 报错 `src refspec main does not match any`，通常是**还没有任何 commit**。

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

若 HTTPS 认证失败：GitHub 通常需要使用 **Personal Access Token**（PAT）作为密码。

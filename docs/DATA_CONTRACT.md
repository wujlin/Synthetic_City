# 数据契约（Data Contract）

> 目标：用最少的约束把“数据怎么落盘、怎么引用、怎么复现”说清楚，避免多人协作时口径漂移。

## 0) 总原则

- 仓库只放代码、文档与小体量产物；大数据一律不进 git。
- 推荐把仓库内的 `data/` 作为软链接指向外置目录（例如 `$RAW_ROOT/synthetic_city/data`），并通过配置文件统一引用。

## 1) 分层落盘

- `raw/`：原始数据，只做整理不改内容；必须附 `metadata.md`（来源/版本/下载日期/许可/覆盖范围/CRS/文件体量）。
- `interim/`：清洗但未统一 schema 的中间产物；可删可再生，但建议保留关键日志。
- `processed/`：统一 schema 后的“可建模数据”；字段稳定、可复现、可被多模块复用。
- `outputs/`：每次 run 的产物（合成结果 + 指标 + 图表 + 日志），目录名包含日期与 git sha。

公开数据抓取建议：先用 `tools/detroit_fetch_public_data.py` 把 TIGER/ACS/PUMS/OSM 等公共数据下载到 `raw/`，并自动生成最小 `*.metadata.json`（URL/时间/参数）。后续如需更严格的来源记录，再补充 `metadata.md`。

## 2) 格式与坐标

- 表格优先 `parquet`；地理优先 `GeoParquet` 或 `gpkg`。
- 存储可用 `EPSG:4326`；涉及面积/距离计算时使用合适投影并在 `metadata.json` 记录投影 EPSG 与计算口径。

## 3) 标识与最小字段

- 地理单元统一用 `geoid`（字符串）；层级用 `geo_level`（如 `city/tract/bg`）。
- 建筑物必须有稳定 `bldg_id`；建议保留 `data_source` 与原始 id（若存在）。

## 4) Detroit 细节

- Detroit 的具体目录树与表结构以 `docs/detroit_code_data_structure.md` 为准。

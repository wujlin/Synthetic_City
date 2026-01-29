#!/usr/bin/env python3
"""
Detroit (Synthetic_city) public data fetcher.

Design goals (KISS):
- Only handles "public, direct-link" downloads + registering existing local datasets.
- Writes raw files + small metadata JSON (source URL, date, params).
- Avoids heavy dependencies; outputs CSV.GZ by default.

Typical usage on wsA:
  export RAW_ROOT=/home/jinlin/data/geoexplicit_data
  python tools/detroit_fetch_public_data.py tiger --out_root "$RAW_ROOT/synthetic_city/data"
  python tools/detroit_fetch_public_data.py acs --out_root "$RAW_ROOT/synthetic_city/data" --acs_year 2023
  python tools/detroit_fetch_public_data.py pums --out_root "$RAW_ROOT/synthetic_city/data" --pums_year 2023
  python tools/detroit_fetch_public_data.py osm --out_root "$RAW_ROOT/synthetic_city/data"
  python tools/detroit_fetch_public_data.py safegraph --out_root "$RAW_ROOT/synthetic_city/data" \
    --safegraph_dir "$RAW_ROOT/safegraph/safegraph_unzip"
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import json
import os
import pathlib
import random
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: pathlib.Path, obj: object) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _http_head(url: str, *, timeout_s: int = 30) -> dict[str, str] | None:
    req = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Synthetic_City/1.0 (+https://github.com/wujlin/Synthetic_City)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _download(url: str, dest: pathlib.Path, *, overwrite: bool = False, timeout_s: int = 60) -> None:
    _ensure_dir(dest.parent)
    if dest.exists() and not overwrite:
        print(f"[skip] exists: {dest}", file=sys.stderr)
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Synthetic_City/1.0 (+https://github.com/wujlin/Synthetic_City)"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, open(tmp, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if sys.stderr.isatty() and total_bytes:
                pct = 100.0 * downloaded / total_bytes
                sys.stderr.write(
                    f"\r[dl] {dest.name}: {downloaded/1e6:.1f}MB/{total_bytes/1e6:.1f}MB ({pct:.1f}%)"
                )
                sys.stderr.flush()

    if sys.stderr.isatty() and total_bytes:
        sys.stderr.write("\n")
    tmp.replace(dest)
    print(f"[ok] downloaded: {dest}", file=sys.stderr)


def _http_get(url: str, *, timeout_s: int = 120, retries: int = 6, backoff_s: float = 1.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Synthetic_City/1.0 (+https://github.com/wujlin/Synthetic_City)",
            # Some servers/proxies are less flaky with explicit close.
            "Connection": "close",
        },
    )

    retries = int(retries)
    if retries < 0:
        retries = 0
    backoff_s = float(backoff_s)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except (urllib.error.URLError, ssl.SSLError, ConnectionResetError, TimeoutError) as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_s = min(60.0, max(0.0, backoff_s) * (2**attempt) + random.random() * 0.2)
            print(f"[warn] HTTP error (attempt {attempt+1}/{retries+1}): {e}; sleep {sleep_s:.1f}s", file=sys.stderr)
            time.sleep(sleep_s)
    raise urllib.error.URLError(last_err)


def _http_get_json(url: str, *, timeout_s: int = 120, retries: int = 6, backoff_s: float = 1.0) -> dict:
    payload = _http_get(url, timeout_s=timeout_s, retries=retries, backoff_s=backoff_s)
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to decode json from url={url}") from e


def _arcgis_query(
    layer_url: str,
    params: dict[str, str],
    *,
    timeout_s: int = 180,
    retries: int = 6,
    backoff_s: float = 1.0,
) -> dict:
    qs = urllib.parse.urlencode(params, safe=":,=*")
    url = layer_url.rstrip("/") + "/query?" + qs
    data = _http_get_json(url, timeout_s=timeout_s, retries=retries, backoff_s=backoff_s)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"ArcGIS query error: url={url}, error={data.get('error')}")
    return data


def _acs_select_estimate_variables(
    variables: dict[str, dict], table_id: str
) -> tuple[list[str], dict[str, dict]]:

    prefix = table_id + "_"
    estimate_vars = []
    estimate_meta: dict[str, dict] = {}
    for name, meta in variables.items():
        if not name.startswith(prefix):
            continue
        # Keep only estimates (E). MOE (M) is optional and can be added later.
        if not name.endswith("E"):
            continue
        # Typical pattern: B01001_001E
        if not re.fullmatch(rf"{re.escape(table_id)}_\d{{3}}E", name):
            continue
        estimate_vars.append(name)
        estimate_meta[name] = meta

    estimate_vars.sort()
    return estimate_vars, estimate_meta


def _acs_fetch(
    *,
    year: int,
    dataset: str,
    state_fips: str,
    county_fips: str,
    geo_level: str,
    get_vars: list[str],
    api_key: str | None,
) -> list[list[str]]:
    if geo_level == "tract":
        for_clause = "tract:*"
        in_clause = f"state:{state_fips} county:{county_fips}"
    elif geo_level == "bg":
        for_clause = "block group:*"
        in_clause = f"state:{state_fips} county:{county_fips} tract:*"
    else:
        raise ValueError(f"unsupported geo_level: {geo_level}")

    params = {"get": ",".join(get_vars), "for": for_clause, "in": in_clause}
    if api_key:
        params["key"] = api_key

    qs = urllib.parse.urlencode(params, safe=":,*")
    url = f"https://api.census.gov/data/{year}/{dataset}?{qs}"

    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"unexpected ACS response, url={url}")
    return payload


def _write_csv_gz(path: pathlib.Path, rows: list[list[str]]) -> None:
    _ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _cmd_tiger(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root).resolve()
    tiger_year = int(args.tiger_year)
    statefp = args.statefp

    layer_map = {
        "place": ("PLACE", "place"),
        "tract": ("TRACT", "tract"),
        "bg": ("BG", "bg"),
        "puma20": ("PUMA20", "puma20"),
    }

    layers = args.layers.split(",")
    unknown = [x for x in layers if x not in layer_map]
    if unknown:
        raise SystemExit(f"Unknown TIGER layer(s): {unknown}. Supported: {sorted(layer_map)}")

    base_dir = out_root / "detroit" / "raw" / "geo" / "tiger" / f"TIGER{tiger_year}"
    for layer in layers:
        dir_name, file_layer = layer_map[layer]
        filename = f"tl_{tiger_year}_{statefp}_{file_layer}.zip"
        url = f"https://www2.census.gov/geo/tiger/TIGER{tiger_year}/{dir_name}/{filename}"

        dest = base_dir / filename
        _download(url, dest, overwrite=args.overwrite)
        _write_json(
            base_dir / f"{filename}.metadata.json",
            {
                "dataset": "US Census TIGER/Line",
                "year": tiger_year,
                "layer": layer,
                "statefp": statefp,
                "url": url,
                "download_utc": _utc_now_iso(),
                "license": "Public domain (US Census TIGER/Line).",
            },
        )


def _cmd_parcels_detroit(args: argparse.Namespace) -> None:
    """
    Download Detroit "Parcels_Current" from the City of Detroit Assessor ArcGIS FeatureServer.

    Why this source:
    - Parcel polygons + assessed_value are available directly from the service (no extra joins).
    - Supports GeoJSON query output.
    - We download in chunks for resumability (offset-based pagination).
    """

    out_root = pathlib.Path(args.out_root).resolve()
    service_url = str(args.service_url).strip().rstrip("/")
    where = str(args.where).strip()
    out_fields = str(args.out_fields).strip()

    out_dir = out_root / "detroit" / "raw" / "parcels" / "detroit_parcels_current"
    _ensure_dir(out_dir)

    retries = int(args.retries)
    backoff_s = float(args.backoff_s)
    sleep_s = float(args.sleep_s)
    timeout_s = int(args.timeout_s)

    # 1) Count
    count_payload = _arcgis_query(
        service_url,
        {"where": where, "returnCountOnly": "true", "f": "json"},
        timeout_s=180,
        retries=retries,
        backoff_s=backoff_s,
    )
    total = int(count_payload.get("count", 0) or 0)
    if total <= 0:
        raise SystemExit(f"ArcGIS returned count={total}. Check service_url/where: {service_url} | {where}")

    record_count = int(args.record_count)
    if record_count <= 0:
        record_count = 2000

    max_chunks = int(args.max_chunks)

    downloaded = 0
    for offset in range(0, total, record_count):
        if max_chunks > 0 and downloaded >= max_chunks:
            break

        dest = out_dir / f"parcels_offset{offset:07d}.geojson"
        if dest.exists() and not args.overwrite:
            downloaded += 1
            continue

        data = _arcgis_query(
            service_url,
            {
                "where": where,
                "outFields": out_fields,
                "returnGeometry": "true",
                "f": "geojson",
                "outSR": "4326",
                "resultOffset": str(offset),
                "resultRecordCount": str(record_count),
            },
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
        )

        features = data.get("features", [])
        if not isinstance(features, list) or not features:
            # No more rows (or service doesn't support resultOffset); stop to avoid infinite loops.
            break

        dest.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
        downloaded += 1
        print(f"[ok] chunk {downloaded}: {dest.name} (features={len(features)})", file=sys.stderr)
        if sleep_s > 0:
            time.sleep(sleep_s)

    _write_json(
        out_dir / "parcels_current.metadata.json",
        {
            "dataset": "City of Detroit Parcels_Current (Assessor) via ArcGIS FeatureServer",
            "service_url": service_url,
            "where": where,
            "out_fields": out_fields,
            "record_count": record_count,
            "count": total,
            "chunks_written": downloaded,
            "download_utc": _utc_now_iso(),
            "license": "UNKNOWN (check Detroit Open Data / ArcGIS Hub terms; do not redistribute without review).",
        },
    )
    print(f"[ok] wrote {downloaded} chunk(s) under: {out_dir}", file=sys.stderr)


def _cmd_wayne_assessment_2025(args: argparse.Namespace) -> None:
    """
    Download Wayne County annual assessment data (2025).

    Note:
    - These URLs are from Wayne County official site and may change in future years.
    - If this download returns 403/blocked on the workstation, fallback is manual browser download + rsync.
    """

    out_root = pathlib.Path(args.out_root).resolve()
    scope = str(args.scope).strip().lower()
    url_map = {
        "full": "https://www.waynecountymi.gov/files/assets/mainsite/v/1/management-amp-budget/documents/assessment-data/2025/2025-82-wayne-county-foia.zip",
        "detroit": "https://www.waynecountymi.gov/files/assets/mainsite/v/1/management-amp-budget/documents/assessment-data/2025/detroit.zip",
    }
    if scope not in url_map:
        raise SystemExit(f"unknown --scope={scope} (use: full|detroit)")
    url = url_map[scope]

    out_dir = out_root / "detroit" / "raw" / "parcels" / "wayne_county_assessment" / "2025"
    _ensure_dir(out_dir)
    filename = pathlib.Path(urllib.parse.urlparse(url).path).name
    dest = out_dir / filename

    _download(url, dest, overwrite=args.overwrite, timeout_s=300)
    _write_json(
        dest.with_suffix(dest.suffix + ".metadata.json"),
        {
            "dataset": "Wayne County Annual Assessment Data",
            "year": 2025,
            "scope": scope,
            "url": url,
            "download_utc": _utc_now_iso(),
            "license": "UNKNOWN (check Wayne County terms; do not redistribute without review).",
        },
    )


def _cmd_acs(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root).resolve()
    acs_year = int(args.acs_year)
    dataset = "acs/acs5"
    state = args.statefp
    county = args.countyfp
    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")

    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    if not tables:
        raise SystemExit("--tables cannot be empty")

    geo_levels = [g.strip() for g in args.geo_levels.split(",") if g.strip()]
    for g in geo_levels:
        if g not in {"tract", "bg"}:
            raise SystemExit(f"unsupported geo level: {g} (use tract,bg)")

    out_dir = out_root / "detroit" / "raw" / "census" / "acs" / f"acs5_{acs_year}"
    _ensure_dir(out_dir)

    # Cache full variables.json for reproducibility and to avoid repeated network calls.
    var_dict_all = out_dir / f"acs5_{acs_year}_variables_all.json"
    if not var_dict_all.exists() or args.overwrite:
        url = f"https://api.census.gov/data/{acs_year}/{dataset}/variables.json"
        _download(url, var_dict_all, overwrite=args.overwrite)

    variables_payload = json.loads(var_dict_all.read_text(encoding="utf-8"))
    variables_all: dict[str, dict] = variables_payload.get("variables", {})

    for table_id in tables:
        vars_e, vars_meta = _acs_select_estimate_variables(variables_all, table_id)
        if not vars_e:
            raise SystemExit(f"no estimate variables found for table {table_id} in {acs_year}/{dataset}")

        # Always include NAME; keep vars count under a conservative cap.
        get_vars = ["NAME"] + vars_e
        if len(get_vars) > 60:
            raise SystemExit(
                f"too many vars for {table_id} ({len(get_vars)}). "
                "KISS: split table or implement chunking."
            )

        # Write a small variable dictionary for this table (estimate only).
        var_dict_path = out_dir / f"acs5_{acs_year}_{table_id}_variables.csv"
        if not var_dict_path.exists() or args.overwrite:
            with open(var_dict_path, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "label", "concept", "predicateType"])
                for name in vars_e:
                    meta = vars_meta.get(name, {})
                    w.writerow([name, meta.get("label", ""), meta.get("concept", ""), meta.get("predicateType", "")])

        for geo_level in geo_levels:
            rows = _acs_fetch(
                year=acs_year,
                dataset=dataset,
                state_fips=state,
                county_fips=county,
                geo_level=geo_level,
                get_vars=get_vars,
                api_key=api_key,
            )
            out_path = out_dir / f"acs5_{acs_year}_{table_id}_{geo_level}_state{state}_county{county}.csv.gz"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] exists: {out_path}", file=sys.stderr)
                continue

            _write_csv_gz(out_path, rows)
            _write_json(
                out_dir / f"{out_path.name}.metadata.json",
                {
                    "dataset": "ACS 5-year (Detailed Tables)",
                    "acs_year": acs_year,
                    "acs_window_hint": f"{acs_year-4}-{acs_year}",
                    "table_id": table_id,
                    "geo_level": geo_level,
                    "statefp": state,
                    "countyfp": county,
                    "n_vars": len(get_vars),
                    "download_utc": _utc_now_iso(),
                    "license": "Public domain (US Census ACS).",
                    "api_key_used": bool(api_key),
                },
            )


def _cmd_pums(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root).resolve()
    year = int(args.pums_year)
    period = args.pums_period
    state = str(args.statefp).zfill(2)

    # Detroit v0 only needs Michigan (MI=26). Keep it explicit to avoid silent mistakes.
    state_postal_lower = "mi" if state == "26" else None
    if state_postal_lower is None:
        raise SystemExit(
            f"Unsupported --statefp={args.statefp}. This detroit helper currently only supports MI (26)."
        )

    base_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/{period}/"
    candidates = [
        # Older naming (common in older releases / mirrors)
        f"psam_h{state}.zip",
        f"psam_p{state}.zip",
        # Newer naming seen in some recent releases (postal abbreviation)
        f"csv_h{state_postal_lower}.zip",
        f"csv_p{state_postal_lower}.zip",
    ]

    out_dir = out_root / "detroit" / "raw" / "pums" / f"pums_{year}_{period}"
    _ensure_dir(out_dir)
    _write_json(
        out_dir / "pums_source.metadata.json",
        {
            "dataset": "ACS PUMS",
            "year": year,
            "period": period,
            "statefp": state,
            "base_url": base_url,
            "download_utc": _utc_now_iso(),
            "license": "Public microdata sample (US Census). Follow non-identification principle.",
            "note": "File naming may change across years; script tries psam_hXX/psam_pXX and csv_h{state}/csv_p{state}.",
        },
    )

    missing = []
    for name in candidates:
        url = base_url + name
        if _http_head(url) is None:
            missing.append(url)
            continue
        _download(url, out_dir / name, overwrite=args.overwrite)

    if missing:
        print("[warn] Some expected PUMS files not found (404):", file=sys.stderr)
        for url in missing:
            print(f"  - {url}", file=sys.stderr)
        print(
            "[hint] open the directory in a browser and confirm filenames; "
            "then download manually or extend candidates list.",
            file=sys.stderr,
        )


def _cmd_osm(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root).resolve()
    region = args.region

    url = f"https://download.geofabrik.de/north-america/us/{region}-latest.osm.pbf"
    out_dir = out_root / "detroit" / "raw" / "transport" / "osm"
    _ensure_dir(out_dir)
    dest = out_dir / f"{region}-latest.osm.pbf"
    _download(url, dest, overwrite=args.overwrite)
    _write_json(
        out_dir / f"{dest.name}.metadata.json",
        {
            "dataset": "OpenStreetMap (Geofabrik extract)",
            "region": region,
            "url": url,
            "download_utc": _utc_now_iso(),
            "license": "ODbL (OpenStreetMap). See https://www.openstreetmap.org/copyright",
        },
    )


def _cmd_safegraph(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root).resolve()
    safegraph_dir = pathlib.Path(args.safegraph_dir).resolve()
    out_dir = out_root / "detroit" / "raw" / "poi" / "safegraph"
    _ensure_dir(out_dir)

    if not safegraph_dir.exists():
        raise SystemExit(f"safegraph_dir not found: {safegraph_dir}")

    link_path = out_dir / "safegraph_unzip"
    if link_path.exists():
        print(f"[skip] exists: {link_path}", file=sys.stderr)
    else:
        link_path.symlink_to(safegraph_dir)
        print(f"[ok] symlink: {link_path} -> {safegraph_dir}", file=sys.stderr)

    # Minimal sanity: count shards if they exist.
    shards = list(safegraph_dir.glob("Global_Places_POI_Data-*.csv"))
    _write_json(
        out_dir / "safegraph.metadata.json",
        {
            "dataset": "SafeGraph Places (existing on wsA)",
            "source_dir": str(safegraph_dir),
            "n_shards_like_Global_Places_POI_Data": len(shards),
            "download_utc": _utc_now_iso(),
            "license": "SEE metadata.md (fill in based on Deway/SafeGraph terms).",
            "license_file": "metadata.md",
        },
    )

    md_path = out_dir / "metadata.md"
    if not md_path.exists():
        md_path.write_text(
            "# SafeGraph Places：许可与使用声明（需人工填写）\n"
            "\n"
            "> 本目录通过 symlink 引用 SafeGraph 原始数据。请务必根据 Deway/供应方合同条款填写并确认。\n"
            "> 默认原则：**不公开发布原始数据**，对外仅发布经过聚合/脱敏的派生结果，并在论文/报告中按要求致谢与标注。\n"
            "\n"
            "## 1. 数据来源\n"
            "\n"
            "- 数据集：SafeGraph Places\n"
            "- 获取渠道：Deway 平台（或内部共享）\n"
            "- 原始数据路径（本机）：`safegraph_unzip/`\n"
            "- 覆盖范围：________\n"
            "- 版本/日期：________\n"
            "\n"
            "## 2. 许可条款（请粘贴关键条款/链接）\n"
            "\n"
            "- 条款链接/文件：________\n"
            "- 是否允许论文发表：________（是/否/需审批）\n"
            "- 是否允许公开发布派生结果：________（是/否/需审批）\n"
            "- 是否允许共享给第三方：________（是/否/需审批）\n"
            "- 必要的引用/致谢格式：________\n"
            "\n"
            "## 3. 风险与合规说明\n"
            "\n"
            "- 是否包含敏感字段：________\n"
            "- 脱敏/聚合策略（如有）：________\n"
            "- 仅在受控环境使用的限制：________\n"
            "\n"
            "## 4. 确认记录\n"
            "\n"
            "- 最后确认日期：________\n"
            "- 确认人：________\n"
            "- 备注：________\n"
            ,
            encoding="utf-8",
        )
        print(f"[ok] wrote template: {md_path}", file=sys.stderr)

    print(f"[info] shards matched: {len(shards)}", file=sys.stderr)
    if shards:
        print(f"[info] example: {shards[0].name}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(prog="detroit_fetch_public_data")
    parser.add_argument(
        "--out_root",
        default=None,
        help="Output root directory (recommended: external data root, e.g. $RAW_ROOT/synthetic_city/data).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Allow placing common args *after* subcommand as well:
    #   python detroit_fetch_public_data.py safegraph --out_root ...   (works)
    #   python detroit_fetch_public_data.py --out_root ... safegraph   (also works)
    def _add_common_after(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--out_root",
            default=argparse.SUPPRESS,
            help="Output root directory (same as top-level --out_root).",
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Overwrite existing files (same as top-level --overwrite).",
        )

    p_tiger = sub.add_parser("tiger", help="Download TIGER/Line shapefiles (MI).")
    _add_common_after(p_tiger)
    p_tiger.add_argument("--tiger_year", default="2023")
    p_tiger.add_argument("--statefp", default="26", help="State FIPS (MI=26).")
    p_tiger.add_argument(
        "--layers",
        default="place,tract,bg,puma20",
        help="Comma-separated: place,tract,bg,puma20",
    )
    p_tiger.set_defaults(func=_cmd_tiger)

    p_acs = sub.add_parser("acs", help="Download ACS 5-year tables via API (Wayne County, MI).")
    _add_common_after(p_acs)
    p_acs.add_argument("--acs_year", default="2023", help="ACS end-year (e.g. 2023 for 2019-2023).")
    p_acs.add_argument("--statefp", default="26", help="State FIPS (MI=26).")
    p_acs.add_argument("--countyfp", default="163", help="County FIPS (Wayne=163).")
    p_acs.add_argument(
        "--tables",
        default="B01001,B11016,B19001,B23025,B08301,B08303",
        help="Comma-separated ACS table IDs (estimates only).",
    )
    p_acs.add_argument(
        "--geo_levels",
        default="tract,bg",
        help="Comma-separated: tract,bg",
    )
    p_acs.add_argument("--api_key", default=None, help="Census API key (or set env CENSUS_API_KEY).")
    p_acs.set_defaults(func=_cmd_acs)

    p_pums = sub.add_parser("pums", help="Download ACS PUMS (MI state files).")
    _add_common_after(p_pums)
    p_pums.add_argument("--pums_year", default="2023", help="PUMS release year (end-year).")
    p_pums.add_argument("--pums_period", default="5-Year", help='Usually "5-Year".')
    p_pums.add_argument("--statefp", default="26", help="State FIPS (MI=26).")
    p_pums.set_defaults(func=_cmd_pums)

    p_osm = sub.add_parser("osm", help="Download Geofabrik OSM PBF.")
    _add_common_after(p_osm)
    p_osm.add_argument("--region", default="michigan", help="Geofabrik region name (e.g., michigan).")
    p_osm.set_defaults(func=_cmd_osm)

    p_safe = sub.add_parser("safegraph", help="Register existing SafeGraph directory via symlink.")
    _add_common_after(p_safe)
    p_safe.add_argument(
        "--safegraph_dir",
        default="/home/jinlin/data/geoexplicit_data/safegraph/safegraph_unzip",
        help="Existing SafeGraph unzip directory on wsA.",
    )
    p_safe.set_defaults(func=_cmd_safegraph)

    p_parcels = sub.add_parser("parcels-detroit", help="Download City of Detroit parcel polygons (Parcels_Current).")
    _add_common_after(p_parcels)
    p_parcels.add_argument(
        "--service_url",
        default="https://services2.arcgis.com/qvkbeam7Wirps6zC/arcgis/rest/services/Parcels_Current/FeatureServer/0",
        help="ArcGIS layer URL for Parcels_Current (layer 0).",
    )
    p_parcels.add_argument("--where", default="1=1", help="ArcGIS where clause (default: 1=1).")
    p_parcels.add_argument(
        "--out_fields",
        default="parcel_number,assessed_value",
        help="Comma-separated fields to include (default: parcel_number,assessed_value).",
    )
    p_parcels.add_argument(
        "--record_count",
        default="2000",
        help="Records per chunk (ArcGIS maxRecordCount is typically 2000).",
    )
    p_parcels.add_argument("--timeout_s", default="300", help="Per-chunk request timeout seconds (default: 300).")
    p_parcels.add_argument("--retries", default="6", help="Retry count on transient SSL/connection errors (default: 6).")
    p_parcels.add_argument("--backoff_s", default="1.0", help="Retry backoff base seconds (default: 1.0).")
    p_parcels.add_argument("--sleep_s", default="0.2", help="Politeness sleep between chunks (default: 0.2).")
    p_parcels.add_argument(
        "--max_chunks",
        default="0",
        help="For debugging: limit number of chunks (0 = no limit).",
    )
    p_parcels.set_defaults(func=_cmd_parcels_detroit)

    p_assess = sub.add_parser("wayne-assessment-2025", help="Download Wayne County 2025 assessment data (ZIP).")
    _add_common_after(p_assess)
    p_assess.add_argument("--scope", choices=["full", "detroit"], default="detroit", help="Download full county or Detroit only.")
    p_assess.set_defaults(func=_cmd_wayne_assessment_2025)

    args = parser.parse_args()
    if not getattr(args, "out_root", None):
        parser.error("--out_root is required (you can put it before or after the subcommand).")
    if not hasattr(args, "overwrite"):
        args.overwrite = False
    args.func(args)


if __name__ == "__main__":
    main()

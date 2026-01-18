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
import re
import sys
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
    state = args.statefp

    base_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/{period}/"
    candidates = [
        f"psam_h{state}.zip",
        f"psam_p{state}.zip",
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
            "note": "File naming may change across years; script tries psam_hXX.zip/psam_pXX.zip.",
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
            "license": "UNKNOWN (must be filled in metadata.md according to Deway terms).",
        },
    )

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
    p_tiger.add_argument("--tiger_year", default="2025")
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

    args = parser.parse_args()
    if not getattr(args, "out_root", None):
        parser.error("--out_root is required (you can put it before or after the subcommand).")
    if not hasattr(args, "overwrite"):
        args.overwrite = False
    args.func(args)


if __name__ == "__main__":
    main()

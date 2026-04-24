#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ID = 190971
PROJECT_API = f"https://api.figshare.com/v2/projects/{PROJECT_ID}/articles?page=1&page_size=100"
ROOT = Path(os.environ.get("GREENSPACE_ROOT", "/home/jinlin/data/Greenspace_Seasonality_Data_Cube"))
MAX_TOTAL_BYTES = int(
    os.environ.get("GREENSPACE_MAX_TOTAL_BYTES", str(240 * 1024**3))
)
PROXY_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
]


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in PROXY_KEYS:
        env.pop(key, None)
    return env


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^0-9A-Za-z _.-]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text


def session_with_retries() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def fetch_json(session: requests.Session, url: str) -> dict | list:
    for attempt in range(10):
        try:
            resp = session.get(url, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2 + attempt)
    raise RuntimeError(f"unreachable: {url}")


def current_payload_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in {"article_meta.json", "project_manifest.json"}:
            total += path.stat().st_size
    return total


def download_file(dest: Path, url: str) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    wget_cmd = [
        "wget",
        "-c",
        "-nv",
        "--tries=20",
        "--timeout=60",
        "--waitretry=2",
        "--retry-connrefused",
        "--show-progress",
        "--progress=dot:giga",
        "-O",
        str(dest),
        url,
    ]
    rc = subprocess.run(wget_cmd, env=clean_env()).returncode
    if rc == 0:
        return 0

    curl_cmd = [
        "curl",
        "-L",
        "--retry",
        "10",
        "--retry-delay",
        "2",
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    return subprocess.run(curl_cmd, env=clean_env()).returncode


def build_project_manifest(session: requests.Session) -> list[dict]:
    ROOT.mkdir(parents=True, exist_ok=True)
    articles = fetch_json(session, PROJECT_API)
    project_manifest = []
    total_bytes = 0

    for idx, article in enumerate(articles, start=1):
        meta = fetch_json(session, article["url_public_api"])
        files = meta.get("files", [])
        article_total = sum(f.get("size", 0) for f in files)
        total_bytes += article_total

        article_dir = ROOT / f"{idx:02d}_{slugify(article['title'])}"
        article_dir.mkdir(parents=True, exist_ok=True)
        (article_dir / "article_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        project_manifest.append(
            {
                "index": idx,
                "article_id": article["id"],
                "title": article["title"],
                "article_dir": str(article_dir),
                "file_count": len(files),
                "total_bytes": article_total,
            }
        )

    (ROOT / "project_manifest.json").write_text(
        json.dumps(
            {
                "project_id": PROJECT_ID,
                "article_count": len(project_manifest),
                "total_bytes": total_bytes,
                "articles": project_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return project_manifest


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    session = session_with_retries()
    project_manifest = build_project_manifest(session)

    print(f"[config] root={ROOT}")
    print(f"[config] max_total_bytes={MAX_TOTAL_BYTES} ({MAX_TOTAL_BYTES / 1024**3:.2f} GiB)")
    sys.stdout.flush()

    for item in project_manifest:
        article_dir = Path(item["article_dir"])
        meta = json.loads((article_dir / "article_meta.json").read_text(encoding="utf-8"))
        files = meta.get("files", [])

        if not files:
            print(f"[skip] {item['title']}: no downloadable files")
            continue

        print(
            f"[article {item['index']:02d}/{len(project_manifest)}] "
            f"{item['title']} | files={len(files)} | bytes={item['total_bytes']}"
        )
        sys.stdout.flush()

        for file_idx, file_meta in enumerate(files, start=1):
            dest = article_dir / file_meta["name"]
            expected = int(file_meta.get("size", 0))
            have = dest.stat().st_size if dest.exists() else 0

            if dest.exists() and have == expected:
                print(f"  [ok] {file_idx:04d}/{len(files):04d} {dest.name}")
                continue

            payload_now = current_payload_bytes(ROOT)
            additional = max(expected - have, 0)
            if payload_now + additional > MAX_TOTAL_BYTES:
                print(
                    f"  [stop] capacity guard hit before {dest.name} | "
                    f"payload={payload_now / 1024**3:.2f} GiB | "
                    f"need+={additional / 1024**3:.2f} GiB | "
                    f"limit={MAX_TOTAL_BYTES / 1024**3:.2f} GiB"
                )
                return 0

            print(
                f"  [get] {file_idx:04d}/{len(files):04d} {dest.name} "
                f"({expected} bytes)"
            )
            sys.stdout.flush()
            rc = download_file(dest, file_meta["download_url"])
            if rc != 0:
                print(f"  [err] download failed rc={rc} for {dest.name}")
                return rc

    print("[done] download completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

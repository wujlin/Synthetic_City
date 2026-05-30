#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.release.export_paper1_release_csv import _export_state, _statefp_from_parquet_path  # noqa: E402


_STATEFP_TO_POSTAL = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


def _state_postal(statefp: str) -> str:
    key = str(statefp).zfill(2)
    try:
        return _STATEFP_TO_POSTAL[key]
    except KeyError as exc:
        raise ValueError(f"unsupported statefp={statefp}") from exc


def _release_file_stem(statefp: str) -> str:
    return f"synthetic_individuals_{_state_postal(statefp)}"


@dataclass(frozen=True)
class OsfFolder:
    name: str
    api_children_url: str
    upload_url: str
    new_folder_url: str
    materialized_path: str


class OsfClient:
    def __init__(self, *, node_id: str, token: str) -> None:
        self.node_id = node_id
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    def _get_json(self, url: str) -> dict[str, Any]:
        resp = self.session.get(url, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def _put(self, url: str, *, data: Any = b"", content_type: str | None = None) -> requests.Response:
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                if hasattr(data, "seek"):
                    data.seek(0)
                resp = self.session.put(url, data=data, headers=headers, timeout=None)
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else 0
                if status < 500 and status not in {408, 429}:
                    raise
            except requests.RequestException as exc:
                last_error = exc
            sleep_s = min(60, 2**attempt)
            print(f"[retry] OSF upload attempt={attempt} sleep={sleep_s}s error={last_error}", flush=True)
            time.sleep(sleep_s)
        if last_error:
            raise last_error
        raise RuntimeError("OSF upload failed without an exception")

    def provider_root_children_url(self) -> str:
        return f"https://api.osf.io/v2/nodes/{self.node_id}/files/osfstorage/"

    def find_child_folder(self, parent_children_url: str, name: str) -> OsfFolder | None:
        url = parent_children_url
        while url:
            payload = self._get_json(url)
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") == "folder" and attrs.get("name") == name:
                    rel = item.get("relationships", {})
                    links = item.get("links", {})
                    return OsfFolder(
                        name=str(attrs.get("name", "")),
                        api_children_url=str(rel["files"]["links"]["related"]["href"]),
                        upload_url=str(links["upload"]),
                        new_folder_url=str(links["new_folder"]),
                        materialized_path=str(attrs.get("materialized_path") or attrs.get("materialized") or ""),
                    )
            url = payload.get("links", {}).get("next")
        return None

    def find_child_file_upload_url(self, parent_children_url: str, name: str) -> str | None:
        url = parent_children_url
        while url:
            payload = self._get_json(url)
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") == "file" and attrs.get("name") == name:
                    return str(item.get("links", {}).get("upload", ""))
            url = payload.get("links", {}).get("next")
        return None

    def find_child_file(self, parent_children_url: str, name: str) -> dict[str, Any] | None:
        url = parent_children_url
        while url:
            payload = self._get_json(url)
            for item in payload.get("data", []):
                attrs = item.get("attributes", {})
                if attrs.get("kind") == "file" and attrs.get("name") == name:
                    return item
            url = payload.get("links", {}).get("next")
        return None

    def rename_file(self, folder: OsfFolder, old_name: str, new_name: str) -> str:
        if old_name == new_name:
            return "unchanged"
        old_item = self.find_child_file(folder.api_children_url, old_name)
        new_item = self.find_child_file(folder.api_children_url, new_name)
        if new_item and not old_item:
            return "already_new"
        if new_item and old_item:
            raise RuntimeError(f"cannot rename {old_name!r} to {new_name!r}: target already exists")
        if not old_item:
            return "missing_old"
        move_url = str(old_item.get("links", {}).get("move", ""))
        if not move_url:
            raise RuntimeError(f"OSF file has no move link: {old_name}")
        resp = self.session.post(move_url, json={"action": "rename", "rename": new_name}, timeout=120)
        resp.raise_for_status()
        return "renamed"

    def ensure_folder(self, parent: OsfFolder | None, name: str) -> OsfFolder:
        parent_children = parent.api_children_url if parent else self.provider_root_children_url()
        child = self.find_child_folder(parent_children, name)
        if child:
            return child
        if parent is None:
            base = f"https://files.osf.io/v1/resources/{self.node_id}/providers/osfstorage/?kind=folder"
        else:
            base = parent.new_folder_url
        sep = "&" if "?" in base else "?"
        url = base + sep + urllib.parse.urlencode({"name": name})
        self._put(url, data=b"")
        child = self.find_child_folder(parent_children, name)
        if not child:
            raise RuntimeError(f"created OSF folder {name!r}, but could not rediscover it")
        return child

    def upload_file(
        self,
        folder: OsfFolder,
        local_path: pathlib.Path,
        remote_name: str,
        *,
        overwrite_existing: bool = False,
    ) -> dict[str, Any]:
        sep = "&" if "?" in folder.upload_url else "?"
        url = folder.upload_url + sep + urllib.parse.urlencode({"kind": "file", "name": remote_name})
        with local_path.open("rb") as fh:
            try:
                resp = self._put(url, data=fh, content_type="application/octet-stream")
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 409:
                    if not overwrite_existing:
                        print(f"[skip-remote-exists] {folder.materialized_path}{remote_name}", flush=True)
                        return {"status_code": 409, "remote_exists": True}
                    existing_upload_url = self.find_child_file_upload_url(folder.api_children_url, remote_name)
                    if not existing_upload_url:
                        raise
                    print(f"[overwrite-remote] {folder.materialized_path}{remote_name}", flush=True)
                    resp = self._put(existing_upload_url, data=fh, content_type="application/octet-stream")
                else:
                    raise
        try:
            return resp.json()
        except Exception:
            return {"status_code": resp.status_code}


def _sha256(path: pathlib.Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _gzip_csv(csv_path: pathlib.Path, gz_path: pathlib.Path) -> None:
    if gz_path.exists() and gz_path.stat().st_mtime >= csv_path.stat().st_mtime and gz_path.stat().st_size > 0:
        return
    tmp = gz_path.with_suffix(gz_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as out:
        subprocess.run(["gzip", "-n", "-c", str(csv_path)], stdout=out, check=True)
    tmp.replace(gz_path)


def _selected_states(run_dir: pathlib.Path, value: str) -> list[str]:
    text = str(value).strip().lower()
    if text in {"", "ready", "completed", "all"}:
        paths = sorted((run_dir / "synthetic").glob("state=*/persons.parquet"))
        return [_statefp_from_parquet_path(p) for p in paths]
    return sorted({part.strip().zfill(2) for part in str(value).split(",") if part.strip()})


def _write_manifest(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "statefp",
        "state_postal",
        "rows",
        "csv_path",
        "gz_path",
        "gz_size_bytes",
        "sha256",
        "uploaded",
        "uploaded_utc",
        "remote_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_uploaded_manifest(path: pathlib.Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("uploaded", "")).lower() == "true":
            out[str(row.get("statefp", "")).zfill(2)] = row
    return out


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    ap = argparse.ArgumentParser(prog="upload_osf_release_incremental")
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--node_id", required=True)
    ap.add_argument("--remote_root", default="synthetic_population")
    ap.add_argument("--states", default="completed")
    ap.add_argument("--release_dir", default="")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--max_states", type=int, default=0)
    ap.add_argument("--skip_upload", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("OSF_TOKEN", "").strip()
    if not token and not args.skip_upload:
        raise SystemExit("OSF_TOKEN is required unless --skip_upload is used")

    run_dir = args.run_dir.expanduser().resolve()
    release_dir = pathlib.Path(args.release_dir).expanduser().resolve() if args.release_dir else run_dir / "release_csv_osf"
    manifest_dir = run_dir / "osf_upload"
    manifest_path = manifest_dir / "upload_manifest.csv"
    states = _selected_states(run_dir, str(args.states))
    if args.max_states > 0:
        states = states[: int(args.max_states)]
    if not states:
        raise SystemExit("no completed state parquet files found")

    client = None
    if not args.skip_upload:
        client = OsfClient(node_id=str(args.node_id), token=token)
        root = client.ensure_folder(None, str(args.remote_root))
        data = client.ensure_folder(root, "data")
        csv_by_state = client.ensure_folder(data, "csv_by_state")
        manifests = client.ensure_folder(root, "manifests")
    else:
        csv_by_state = None
        manifests = None

    rows: list[dict[str, Any]] = []
    previously_uploaded = _load_uploaded_manifest(manifest_path)
    for statefp in states:
        state_postal = _state_postal(statefp)
        file_stem = _release_file_stem(statefp)
        remote_name = f"{file_stem}.csv.gz"
        prior = previously_uploaded.get(statefp)
        if (
            prior
            and str(prior.get("sha256", ""))
            and str(prior.get("remote_name", "")) == remote_name
            and str(prior.get("uploaded", "")).lower() == "true"
        ):
            print(f"[skip-uploaded] state={statefp} remote_name={remote_name}", flush=True)
            row = dict(prior)
            row["statefp"] = statefp
            row["state_postal"] = state_postal
            rows.append(row)
            _write_manifest(rows, manifest_path)
            continue

        parquet_path = run_dir / "synthetic" / f"state={statefp}" / "persons.parquet"
        if not parquet_path.exists():
            print(f"[skip] state={statefp} missing {parquet_path}", flush=True)
            continue
        state_dir = release_dir / f"state={statefp}"
        csv_path = state_dir / f"{file_stem}.csv"
        gz_path = state_dir / f"{file_stem}.csv.gz"
        sha_path = state_dir / f"{file_stem}.csv.gz.sha256"

        if not csv_path.exists() or csv_path.stat().st_mtime < parquet_path.stat().st_mtime:
            print(f"[export] state={statefp} -> {csv_path}", flush=True)
            export_row = _export_state(parquet_path=parquet_path, out_csv=csv_path, chunksize=int(args.chunksize))
        else:
            export_row = {"statefp": statefp, "rows": int(sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1)}

        print(f"[gzip] state={statefp}", flush=True)
        _gzip_csv(csv_path, gz_path)
        digest = _sha256(gz_path)
        sha_path.write_text(f"{digest}  {gz_path.name}\n", encoding="utf-8")

        row = {
            "statefp": statefp,
            "state_postal": state_postal,
            "rows": int(export_row.get("rows", 0)),
            "csv_path": str(csv_path),
            "gz_path": str(gz_path),
            "gz_size_bytes": int(gz_path.stat().st_size),
            "sha256": digest,
            "uploaded": False,
            "uploaded_utc": "",
            "remote_name": remote_name,
        }
        if prior and str(prior.get("sha256", "")) == digest and str(prior.get("remote_name", "")) == remote_name:
            print(f"[skip-uploaded] state={statefp} already in manifest with matching sha256", flush=True)
            row["uploaded"] = True
            row["uploaded_utc"] = str(prior.get("uploaded_utc", ""))
        elif client and csv_by_state and manifests:
            print(f"[upload] state={statefp} {gz_path.stat().st_size / 1e9:.3f} GB", flush=True)
            client.upload_file(csv_by_state, gz_path, remote_name)
            client.upload_file(manifests, sha_path, sha_path.name, overwrite_existing=True)
            row["uploaded"] = True
            row["uploaded_utc"] = _utc_now()
        rows.append(row)
        _write_manifest(rows, manifest_path)

    if client and manifests and manifest_path.exists():
        client.upload_file(manifests, manifest_path, manifest_path.name, overwrite_existing=True)

    summary = {
        "created_utc": _utc_now(),
        "node_id": str(args.node_id),
        "remote_root": str(args.remote_root),
        "run_dir": str(run_dir),
        "release_dir": str(release_dir),
        "states_attempted": states,
        "states_uploaded": [r["statefp"] for r in rows if r.get("uploaded")],
        "rows_uploaded": int(sum(int(r.get("rows", 0)) for r in rows if r.get("uploaded"))),
        "bytes_uploaded": int(sum(int(r.get("gz_size_bytes", 0)) for r in rows if r.get("uploaded"))),
        "manifest_csv": str(manifest_path),
    }
    summary_path = manifest_dir / "upload_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if client and manifests:
        client.upload_file(manifests, summary_path, summary_path.name, overwrite_existing=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

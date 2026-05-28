#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Any


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.upload_osf_release_incremental import (  # noqa: E402
    OsfClient,
    _release_file_stem,
    _state_postal,
    _utc_now,
)


def _read_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def _selected_states(value: str, manifest_rows: list[dict[str, str]]) -> list[str]:
    text = str(value).strip().lower()
    if text in {"", "manifest", "uploaded"}:
        states = [str(r.get("statefp", "")).zfill(2) for r in manifest_rows if r.get("statefp")]
        return sorted(set(states))
    return sorted({part.strip().zfill(2) for part in str(value).split(",") if part.strip()})


def main() -> int:
    ap = argparse.ArgumentParser(prog="osf_rename_state_files_to_postal")
    ap.add_argument("--run_dir", required=True, type=pathlib.Path)
    ap.add_argument("--node_id", required=True)
    ap.add_argument("--remote_root", default="synthetic_population")
    ap.add_argument("--states", default="manifest")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("OSF_TOKEN", "").strip()
    if not token and not args.dry_run:
        raise SystemExit("OSF_TOKEN is required unless --dry_run is used")

    run_dir = args.run_dir.expanduser().resolve()
    manifest_dir = run_dir / "osf_upload"
    manifest_path = manifest_dir / "upload_manifest.csv"
    rows = _read_manifest(manifest_path)
    if not rows:
        raise SystemExit(f"missing or empty manifest: {manifest_path}")

    states = _selected_states(str(args.states), rows)
    by_state = {str(r.get("statefp", "")).zfill(2): dict(r) for r in rows if r.get("statefp")}
    client = None
    csv_by_state = None
    manifests = None
    if not args.dry_run:
        client = OsfClient(node_id=str(args.node_id), token=token)
        root = client.ensure_folder(None, str(args.remote_root))
        data = client.ensure_folder(root, "data")
        csv_by_state = client.ensure_folder(data, "csv_by_state")
        manifests = client.ensure_folder(root, "manifests")

    operations: list[dict[str, str]] = []
    for statefp in states:
        if statefp not in by_state:
            print(f"[skip] state={statefp} not in manifest", flush=True)
            continue
        row = by_state[statefp]
        postal = _state_postal(statefp)
        new_data_name = f"{_release_file_stem(statefp)}.csv.gz"
        legacy_data_name = f"synthetic_individuals_state{statefp}.csv.gz"
        manifest_data_name = str(row.get("remote_name") or legacy_data_name)
        new_sha_name = f"{new_data_name}.sha256"
        old_data_name = manifest_data_name
        old_sha_name = f"{old_data_name}.sha256"
        print(f"[rename] state={statefp} {manifest_data_name} -> {new_data_name}", flush=True)

        data_status = "dry_run"
        sha_status = "dry_run"
        if client and csv_by_state and manifests:
            if client.find_child_file(csv_by_state.api_children_url, new_data_name):
                data_status = "already_new"
            else:
                candidates = []
                for candidate in [manifest_data_name, legacy_data_name]:
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
                data_status = "missing_old"
                for candidate in candidates:
                    if client.find_child_file(csv_by_state.api_children_url, candidate):
                        old_data_name = candidate
                        data_status = client.rename_file(csv_by_state, candidate, new_data_name)
                        break

            if client.find_child_file(manifests.api_children_url, new_sha_name):
                sha_status = "already_new"
            else:
                candidates = []
                for candidate in [f"{manifest_data_name}.sha256", f"{legacy_data_name}.sha256"]:
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
                sha_status = "missing_old"
                for candidate in candidates:
                    if client.find_child_file(manifests.api_children_url, candidate):
                        old_sha_name = candidate
                        sha_status = client.rename_file(manifests, candidate, new_sha_name)
                        break
            sha_text = f"{row.get('sha256', '').strip()}  {new_data_name}\n"
            sha_local = manifest_dir / new_sha_name
            sha_local.write_text(sha_text, encoding="utf-8")
            client.upload_file(manifests, sha_local, new_sha_name, overwrite_existing=True)

        row["state_postal"] = postal
        row["remote_name"] = new_data_name
        row["uploaded"] = "True"
        row["uploaded_utc"] = row.get("uploaded_utc") or _utc_now()
        by_state[statefp] = row
        operations.append(
            {
                "statefp": statefp,
                "state_postal": postal,
                "old_data_name": old_data_name,
                "new_data_name": new_data_name,
                "data_status": data_status,
                "old_sha_name": old_sha_name,
                "new_sha_name": new_sha_name,
                "sha_status": sha_status,
            }
        )

    out_rows = [by_state[str(r.get("statefp", "")).zfill(2)] for r in rows if r.get("statefp")]
    if not args.dry_run:
        _write_manifest(out_rows, manifest_path)

    summary_path = manifest_dir / "rename_state_files_to_postal_summary.json"
    summary = {
        "created_utc": _utc_now(),
        "node_id": str(args.node_id),
        "remote_root": str(args.remote_root),
        "run_dir": str(run_dir),
        "states": states,
        "operations": operations,
    }
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if client and manifests:
        client.upload_file(manifests, manifest_path, manifest_path.name, overwrite_existing=True)
        client.upload_file(manifests, summary_path, summary_path.name, overwrite_existing=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

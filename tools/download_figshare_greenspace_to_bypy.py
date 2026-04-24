#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
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
ROOT = Path("/home/jinlin/data/Greenspace_Seasonality_Data_Cube")
BYPY_REMOTE_ROOT = "Greenspace_Seasonality_Data_Cube"
UPLOAD_STATE_PATH = ROOT / "upload_state.json"
DELETE_AFTER_UPLOAD = True
DOWNLOAD_WORKERS = int(os.environ.get("GREENSPACE_DOWNLOAD_WORKERS", "4"))
MAX_PENDING_BYTES = int(os.environ.get("GREENSPACE_MAX_PENDING_BYTES", str(8 * 1024**3)))
UPLOAD_VERIFY_RETRIES = int(os.environ.get("GREENSPACE_UPLOAD_VERIFY_RETRIES", "8"))
UPLOAD_VERIFY_SLEEP = float(os.environ.get("GREENSPACE_UPLOAD_VERIFY_SLEEP", "5"))
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


@dataclass(frozen=True)
class FileTask:
    article_index: int
    article_count: int
    article_title: str
    file_idx: int
    file_count: int
    dest: Path
    expected_size: int
    download_url: str
    remote_parent: str
    state_key: str


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


def run_command(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=clean_env())
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


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


def load_upload_state() -> dict[str, dict]:
    if not UPLOAD_STATE_PATH.exists():
        return {}
    return json.loads(UPLOAD_STATE_PATH.read_text(encoding="utf-8"))


def save_upload_state(state: dict[str, dict]) -> None:
    UPLOAD_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_remote_dir(remote_dir: str) -> None:
    remote_dir = remote_dir.strip("/.")
    if not remote_dir:
        return
    cur = ""
    for part in remote_dir.split("/"):
        cur = f"{cur}/{part}" if cur else part
        proc = run_command(["bypy", "mkdir", cur])
        if proc.returncode == 0:
            continue
        joined = f"{proc.stdout}\n{proc.stderr}".lower()
        if "exists" in joined or "already" in joined:
            continue
        raise RuntimeError(
            f"bypy mkdir failed for {cur}\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def upload_file(local_path: Path, remote_parent: str) -> None:
    ensure_remote_dir(remote_parent)
    proc = run_command(["bypy", "-e", "upload", str(local_path), remote_parent])
    if proc.returncode != 0:
        raise RuntimeError(
            f"bypy upload failed for {local_path}\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )


def list_remote_dir(remote_dir: str) -> dict[str, int]:
    proc = run_command(["bypy", "list", remote_dir])
    if proc.returncode != 0:
        raise RuntimeError(
            f"bypy list failed for {remote_dir}\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    file_pat = re.compile(r"^F\s+(.*?)\s+(\d+)\s+\d{4}-\d{2}-\d{2},")
    files: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        m = file_pat.match(line.strip())
        if m:
            files[m.group(1)] = int(m.group(2))
    return files


def verify_remote_file(remote_parent: str, filename: str, expected_size: int) -> bool:
    remote_path = f"{remote_parent}/{filename}"
    size_pat = re.compile(rf"^{re.escape(filename)}\|(\d+)(?:\|.*)?$")
    for attempt in range(UPLOAD_VERIFY_RETRIES):
        proc = run_command(["bypy", "meta", remote_path, "$f|$s|$m"])
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                m = size_pat.match(line.strip())
                if m and int(m.group(1)) == expected_size:
                    return True
        time.sleep(UPLOAD_VERIFY_SLEEP * (attempt + 1))
    return False


def reconcile_upload_state(upload_state: dict[str, dict]) -> dict[str, dict]:
    uploaded_items = {
        key: value
        for key, value in upload_state.items()
        if value.get("uploaded") and value.get("remote_parent")
    }
    if not uploaded_items:
        return upload_state

    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, value in uploaded_items.items():
        grouped[str(value["remote_parent"])].append((key, value))

    for remote_parent, items in grouped.items():
        remote_files = list_remote_dir(remote_parent)
        for state_key, value in items:
            filename = Path(state_key).name
            expected = int(value.get("size", 0))
            actual = remote_files.get(filename)
            if actual != expected:
                upload_state.pop(state_key, None)
    return upload_state


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
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
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


def build_tasks(project_manifest: list[dict], upload_state: dict[str, dict]) -> list[FileTask]:
    tasks: list[FileTask] = []
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

        remote_rel_dir = article_dir.relative_to(ROOT).as_posix()
        remote_parent = f"{BYPY_REMOTE_ROOT}/{remote_rel_dir}"

        for file_idx, file_meta in enumerate(files, start=1):
            dest = article_dir / file_meta["name"]
            state_key = str(dest.relative_to(ROOT))
            state = upload_state.get(state_key, {})
            if state.get("uploaded") and state.get("remote_parent") == remote_parent:
                print(f"  [up-ok] {file_idx:04d}/{len(files):04d} {dest.name}")
                continue
            tasks.append(
                FileTask(
                    article_index=item["index"],
                    article_count=len(project_manifest),
                    article_title=item["title"],
                    file_idx=file_idx,
                    file_count=len(files),
                    dest=dest,
                    expected_size=int(file_meta.get("size", 0)),
                    download_url=str(file_meta["download_url"]),
                    remote_parent=remote_parent,
                    state_key=state_key,
                )
            )
    return tasks


def download_task(task: FileTask) -> FileTask:
    dest = task.dest
    expected = task.expected_size
    if dest.exists() and dest.stat().st_size == expected:
        return task
    rc = download_file(dest, task.download_url)
    if rc != 0:
        raise RuntimeError(f"download failed rc={rc} for {dest.name}")
    if expected and dest.stat().st_size != expected:
        raise RuntimeError(
            f"size mismatch for {dest.name}: {dest.stat().st_size} != {expected}"
        )
    return task


def main() -> int:
    session = session_with_retries()
    project_manifest = build_project_manifest(session)
    upload_state = load_upload_state()
    upload_state = reconcile_upload_state(upload_state)
    save_upload_state(upload_state)
    tasks = build_tasks(project_manifest, upload_state)
    task_iter = iter(tasks)
    ready: deque[FileTask] = deque()
    inflight: dict[Future[FileTask], FileTask] = {}
    pending_bytes = 0
    exhausted = False

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        while True:
            while (
                not exhausted
                and len(inflight) < DOWNLOAD_WORKERS
                and pending_bytes < MAX_PENDING_BYTES
            ):
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                if task.dest.exists() and task.dest.stat().st_size == task.expected_size:
                    print(f"  [ok]  {task.file_idx:04d}/{task.file_count:04d} {task.dest.name}")
                    ready.append(task)
                    pending_bytes += task.dest.stat().st_size
                else:
                    print(
                        f"  [get] {task.file_idx:04d}/{task.file_count:04d} {task.dest.name} "
                        f"({task.expected_size} bytes)"
                    )
                    sys.stdout.flush()
                    inflight[pool.submit(download_task, task)] = task

            if ready:
                task = ready.popleft()
                dest = task.dest
                print(
                    f"  [push] {task.file_idx:04d}/{task.file_count:04d} "
                    f"{dest.name} -> {task.remote_parent}"
                )
                sys.stdout.flush()
                upload_file(dest, task.remote_parent)
                if not verify_remote_file(task.remote_parent, dest.name, dest.stat().st_size):
                    raise RuntimeError(
                        f"remote verification failed for {task.remote_parent}/{dest.name}"
                    )
                upload_state[task.state_key] = {
                    "uploaded": True,
                    "size": dest.stat().st_size,
                    "remote_parent": task.remote_parent,
                    "updated_at": int(time.time()),
                }
                save_upload_state(upload_state)
                if DELETE_AFTER_UPLOAD and dest.exists():
                    size_now = dest.stat().st_size
                    dest.unlink()
                    pending_bytes = max(0, pending_bytes - size_now)
                    print(f"  [drop] {task.file_idx:04d}/{task.file_count:04d} {dest.name}")
                else:
                    pending_bytes = max(0, pending_bytes - dest.stat().st_size)
                continue

            if inflight:
                done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task = inflight.pop(future)
                    future.result()
                    ready.append(task)
                    pending_bytes += task.dest.stat().st_size
                continue

            if exhausted:
                break

    print("[done] download + upload completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

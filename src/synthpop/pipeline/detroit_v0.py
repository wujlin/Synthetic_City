from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any

from ..detroit.constants import DEFAULT_TIGER_YEAR, STATEFP_MI
from ..detroit.paths import detroit_root, safegraph_dir, tiger_dir
from ..paths import ensure_dir

_DEFAULT_PUMS_YEAR = 2023
_DEFAULT_PUMS_PERIOD = "5-Year"
_DEFAULT_PUMS_STATE_POSTAL_LOWER = "mi"  # MI for Detroit v0


def make_run_id(*, prefix: str = "detroit_v0") -> str:
    ts = _dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "").replace("-", "")
    return f"{ts}_{prefix}"


def status(*, data_root: pathlib.Path) -> dict[str, Any]:
    """
    Return a lightweight “what do we have” snapshot.
    Only checks paths exist; avoids heavy validation by design.
    """
    det = detroit_root(data_root)
    tiger = tiger_dir(data_root, tiger_year=DEFAULT_TIGER_YEAR)
    sg = safegraph_dir(data_root)

    def _exists(p: pathlib.Path) -> bool:
        try:
            return p.exists()
        except OSError:
            return False

    expected_tiger = {
        "place": tiger / f"tl_{DEFAULT_TIGER_YEAR}_{STATEFP_MI}_place.zip",
        "tract": tiger / f"tl_{DEFAULT_TIGER_YEAR}_{STATEFP_MI}_tract.zip",
        "bg": tiger / f"tl_{DEFAULT_TIGER_YEAR}_{STATEFP_MI}_bg.zip",
        "puma20": tiger / f"tl_{DEFAULT_TIGER_YEAR}_{STATEFP_MI}_puma20.zip",
    }

    pums_root = det / "raw" / "pums"
    pums_default_psam = (
        pums_root / f"pums_{_DEFAULT_PUMS_YEAR}_{_DEFAULT_PUMS_PERIOD}" / f"psam_p{STATEFP_MI}.zip"
    )
    pums_default_csv = (
        pums_root
        / f"pums_{_DEFAULT_PUMS_YEAR}_{_DEFAULT_PUMS_PERIOD}"
        / f"csv_p{_DEFAULT_PUMS_STATE_POSTAL_LOWER}.zip"
    )
    pums_found = sorted(
        set(
            list(pums_root.glob(f"pums_*/*psam_p{STATEFP_MI}.zip"))
            + list(pums_root.glob(f"pums_*/*csv_p{_DEFAULT_PUMS_STATE_POSTAL_LOWER}.zip"))
        )
    )

    return {
        "data_root": str(data_root),
        "detroit_root": str(det),
        "raw": {
            "tiger_dir": str(tiger),
            "tiger_expected": {k: {"path": str(v), "exists": _exists(v)} for k, v in expected_tiger.items()},
            "pums": {
                "default_person_zips": {
                    "psam": {"path": str(pums_default_psam), "exists": _exists(pums_default_psam)},
                    "csv": {"path": str(pums_default_csv), "exists": _exists(pums_default_csv)},
                },
                "found_zips": [str(p) for p in pums_found[:10]],
            },
            "safegraph_dir": str(sg),
            "safegraph_unzip": {"path": str(sg / "safegraph_unzip"), "exists": _exists(sg / "safegraph_unzip")},
            "safegraph_meta": {
                "path": str(sg / "safegraph.metadata.json"),
                "exists": _exists(sg / "safegraph.metadata.json"),
            },
        },
    }


def print_status(*, data_root: pathlib.Path) -> None:
    print(json.dumps(status(data_root=data_root), ensure_ascii=False, indent=2))


def init_dirs(*, data_root: pathlib.Path) -> dict[str, str]:
    """
    Create canonical folders (non-destructive).
    No files are moved or deleted.
    """
    det = detroit_root(data_root)
    created = {}

    for rel in [
        "raw/geo/tiger",
        "raw/census",
        "raw/pums",
        "raw/buildings",
        "raw/poi",
        "raw/mobility",
        "interim",
        "processed",
        "outputs/runs",
    ]:
        p = ensure_dir(det / rel)
        created[rel] = str(p)

    return created


def assign_buildings_v0(
    *,
    persons_geo: Any,
    buildings: Any,
    seed: int = 0,
) -> Any:
    """
    Detroit v0 spatial anchoring (two-stage):
    - Input persons already carry BG/tract assignment (person-only v0).
    - Output person → building assignment within BG using capacity proxies.
    """
    from ..spatial.assign_buildings import assign_buildings_within_bg

    return assign_buildings_within_bg(persons=persons_geo, buildings=buildings, seed=seed)

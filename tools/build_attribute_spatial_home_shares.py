#!/usr/bin/env python3
from __future__ import annotations

"""
Aggregate synthetic person locations into tract-level residential shares for selected groups.

Why:
- Figure 4 visualizes attribute-conditioned residential patterns.
- The source person-level file can be very large, so this script reads it as a stream and
  writes a compact tract-level CSV.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


CHILD_BINS = {"[0.0, 5.0)", "[5.0, 18.0)"}
SENIOR_BINS = {"[65.0, 75.0)", "[75.0, 85.0)", "[85.0, 1000.0)"}


def _open_input(path: str):
    if path == "-":
        return sys.stdin
    return open(path, "r", newline="", encoding="utf-8")


def aggregate(persons_csv: str) -> list[dict[str, float | int | str]]:
    total = Counter()
    child = Counter()
    female = Counter()
    senior = Counter()
    bachelor = Counter()
    employed = Counter()
    high_income = Counter()

    with _open_input(persons_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tract = str(row.get("tract_geoid", "")).strip()
            if not tract:
                continue
            total[tract] += 1
            if row.get("AGEP_bin", "") in CHILD_BINS:
                child[tract] += 1
            if str(row.get("SEX", "")).strip() == "2":
                female[tract] += 1
            if row.get("AGEP_bin", "") in SENIOR_BINS:
                senior[tract] += 1
            if row.get("SCHL_allpop", "") == "bachelor_plus":
                bachelor[tract] += 1
            if row.get("ESR_allpop", "") == "employed":
                employed[tract] += 1
            if row.get("EARN_16p_bin", "") == "ge_100k":
                high_income[tract] += 1

    rows: list[dict[str, float | int | str]] = []
    for tract in sorted(total):
        denom = float(total[tract])
        rows.append(
            {
                "tract_geoid": tract,
                "total_residents": int(total[tract]),
                "child_share": child[tract] / denom,
                "female_share": female[tract] / denom,
                "senior_share": senior[tract] / denom,
                "bachelor_plus_share": bachelor[tract] / denom,
                "employed_share": employed[tract] / denom,
                "high_income_share": high_income[tract] / denom,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persons_csv", required=True, help="Local CSV path or '-' for stdin.")
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    rows = aggregate(args.persons_csv)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tract_geoid",
                "total_residents",
                "child_share",
                "female_share",
                "senior_share",
                "bachelor_plus_share",
                "employed_share",
                "high_income_share",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {args.out_csv}")


if __name__ == "__main__":
    main()

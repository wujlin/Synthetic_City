# Data Contract

This repository tracks code, lightweight documentation, schemas, and small validation artifacts. It does not track raw data, licensed data, checkpoints, national synthetic population files, or large experiment outputs.

## Storage Layout

Use external data roots for large assets:

- `raw/`: source data with source URL, date, license, coverage, coordinate reference system, and file size metadata.
- `interim/`: cleaned but not yet schema-stable intermediates.
- `processed/`: schema-stable model inputs.
- `outputs/`: run artifacts, metrics, logs, checkpoints, release exports, and figures.

The local `data/` and `outputs/` paths are machine-specific and are ignored by Git.

## Geographic Identifiers

- Store geographic identifiers as strings.
- Use zero-padded Federal Information Processing Series and Census GEOID values.
- Use explicit column names such as `statefp`, `puma`, `puma_uid`, `tract_geoid`, and `block_group_geoid` when intermediate files require them.
- Release files should expose only the public release schema described in the README.

## Release Schema

The public synthetic population release uses 10 columns:

```text
person_id, age, gender, education, employment, income,
home_lon, home_lat, work_lon, work_lat
```

State-level release files are compressed CSV files named with USPS state abbreviations, for example `synthetic_individuals_CA.csv.gz`.

## Public Dataset

The release dataset is hosted on OSF:

<https://osf.io/e7wp8/>

The OSF release includes state files, upload manifests, checksums, a README, and a license statement.

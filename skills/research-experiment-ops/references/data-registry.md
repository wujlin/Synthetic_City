# Data Registry

## Goal

Large datasets should be discoverable even when they are not stored in git.

## Register every important dataset with

- a human-readable name
- local full-data path
- remote full-data path, if different
- sample path inside the repository, if available
- file role
- whether it is versioned, local-only, or remote-only

## Minimum distinctions

Always distinguish:

- sample vs full data
- derived vs raw data
- local path vs remote path
- manuscript-facing assets vs infrastructure assets

## Sample policy

If the full dataset is too large for the repository:

- keep a sample, head, or profile artifact in git when it helps interpretation
- document how the sample relates to the full file
- avoid letting the sample become the only surviving description of the dataset

## Common failure mode

Do not assume that because a script knows a path, the project knows the dataset.

The registry should make it obvious:

- what the file is
- where it lives
- what it is used for

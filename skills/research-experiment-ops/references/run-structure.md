# Run Structure

## Goal

Every experiment run should answer one clearly named question and produce a predictable artifact set.

## Naming rules

Use directory names that encode:

- task or dataset
- variable setting or resolution
- split or evaluation mode
- timestamp

Good pattern:

- `outputs/_task_setting_split_YYYYMMDDThhmmssZ`

Avoid:

- `test1`
- `new-run`
- `final-final`
- names that only make sense today

## Minimum artifact set

Before calling a run “complete”, expect at least:

- `run.log`
- `run_summary.json`
- `metrics/` with the key evaluation JSON files
- `checkpoints/` if training occurred
- exported tables or figures if the run directly supports the manuscript

If a run is missing its summary or metrics, do not cite it yet.

## Frozen vs exploratory runs

Distinguish two states:

- **exploratory run**: useful for diagnosis, not yet safe to cite
- **frozen run**: artifact-complete, numerically checked, and safe to reference in text, tables, or figures

Do not let manuscript text depend on exploratory runs.

## Logging rules

- Log the exact command that launched the run.
- Keep condition setting, split mode, and key hyperparameters discoverable from either the directory name or `run_summary.json`.
- If a script emits table-ready metrics, store them under `metrics/` rather than burying them in logs.

---
name: research-experiment-ops
description: Use when planning, running, naming, logging, checking, syncing, or organizing research experiments across local and remote environments. This skill standardizes run directories, required artifacts, dataset registration, remote sync habits, manuscript-readiness checks, and worktree-based parallel development.
---

# Research Experiment Ops

Use this skill when the task involves any of the following:

- starting a new experiment and deciding how to name its output directory
- defining what logs, metrics, checkpoints, and summaries a run must produce
- checking whether a run is complete enough to be cited in a paper
- organizing sample data, full data, local paths, and remote paths
- syncing experiment outputs between workstation and local machine
- planning parallel work with `git worktree`
- separating exploratory runs from frozen results used in figures or tables

## Core workflow

1. Define the experiment unit: what question this run answers and what output directory should represent it.
2. Freeze the run structure before launching: logs, metrics, checkpoints, and summary files.
3. Treat large datasets and remote paths as registered assets, not as implicit tribal knowledge.
4. Before a result enters the manuscript, check artifact completeness and figure-text consistency.
5. Use worktrees to isolate task streams, not to multiply confusion.

## Reference map

- For run directory naming, artifact expectations, and logging rules, read [references/run-structure.md](references/run-structure.md).
- For local/remote data handling and synchronization rules, read [references/remote-and-sync.md](references/remote-and-sync.md).
- For registering datasets and large files, read [references/data-registry.md](references/data-registry.md).
- For deciding whether a result is ready to enter the manuscript, read [references/manuscript-readiness.md](references/manuscript-readiness.md).
- For parallel development with `git worktree`, read [references/worktree-usage.md](references/worktree-usage.md).

## Task routing

- If the user is about to launch or rename an experiment, load `run-structure.md` first.
- If the task involves a workstation, SSH, rsync, or pulling outputs back to local, load `remote-and-sync.md` first.
- If the user is dealing with large datasets or sample/full-data relationships, load `data-registry.md` first.
- If the user wants to know whether a result can be cited, plotted, or turned into a table, load `manuscript-readiness.md` first.
- If the user wants to parallelize work across branches or tasks, load `worktree-usage.md` first.

## Output defaults

- Prefer stable naming over short naming.
- Prefer explicit file expectations over implied conventions.
- Never store secrets such as passwords inside the skill or dataset registry.
- Distinguish exploratory results from frozen, citable results.

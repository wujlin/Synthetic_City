# Worktree Usage

## When to use `git worktree`

Use worktrees when the project has genuinely parallel task streams, for example:

- manuscript polishing
- experiment development
- slide or presentation preparation

## Recommended principle

Split by **task stream**, not by file type.

Good examples:

- `main-writing`
- `exp-dev`
- `slides`

Bad examples:

- one tree for `.tex`
- one tree for `.py`

## Benefits

- avoid repeated checkout and stash cycles
- keep writing work cleaner than experiment work
- reduce accidental cross-contamination between exploratory code and manuscript text

## Rules

- One worktree should have one main purpose.
- Avoid editing the same high-risk file in multiple worktrees at once.
- Treat the writing worktree as the cleanest one.
- Let experimental worktrees absorb dirty outputs and exploratory runs.

## For mixed code-writing projects

If outputs are large or partially tracked:

- do not generate manuscript-facing citations from half-finished experiment worktrees
- freeze the output directory first
- then propagate only the stable result back into the writing worktree

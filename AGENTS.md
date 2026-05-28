# AGENTS.md

## Project Scope

This repository is the public code and documentation surface for the SIGSPATIAL synthetic population work. Keep it focused on the national synthetic population framework, release utilities, lightweight tests, and reusable source modules.

Out-of-scope material should stay outside the repository:

- proposal drafts;
- private reading notes;
- local data mirrors;
- experiment outputs;
- legacy Detroit-only analysis products;
- generated model checkpoints and large figures.

## Repository Rules

- Keep repository text in English.
- Do not commit raw data, licensed data, model weights, generated full-population files, or local run outputs.
- Keep large or private assets in external storage and document only stable access points.
- Prefer concise, maintainable code over over-engineered abstractions.
- Preserve reproducibility through scripts, schemas, checksums, and small smoke tests.

## Editing Rules

- Use focused commits.
- Do not mix manuscript text, data products, and code refactors in the same commit.
- Before removing or archiving files, make sure they are recoverable from Git history or an external archive.
- Keep public-facing documentation aligned with the current method: target/condition construction, hierarchical diffusion, and spatial assignment.

## Validation

Before pushing changes that affect code, run the most relevant lightweight tests. If full tests cannot be run because data are unavailable, state that clearly in the commit or handoff notes.

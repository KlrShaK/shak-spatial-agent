# Repository Guidelines

## Project Structure & Module Organization

This Python research repository studies video spatial reasoning and dataset analysis. `d4rt_agent/` contains agents, DSI-Bench evaluation, and trajectory/orientation tools in `sam_orientany_d4rt_test/`. `dsi_dataset_exploration/` and `dyn_dataset_exploration/` contain taxonomy and analysis; Dyn-Bench decomposition is in `dyn_dataset_exploration/decomposition/`. Prompts and notes live beside relevant code. `scripts/` holds SLURM launchers; `third-party-tools/` holds pinned submodules. Do not commit generated results, logs, weights, or datasets unless needed.

## Build, Test, and Development Commands

There is no separate package build step. Set up a project Python environment using `environment.yml` or `requirements.txt`. Run focused, GPU-free tests with:

```bash
python -m pytest d4rt_agent/agent_v4/test_agent_v4.py
python -m pytest dyn_dataset_exploration/test_decomposition.py -k <name>
```

Run the repository test suite with `python -m pytest d4rt_agent dsi_dataset_exploration dyn_dataset_exploration`. Submit GPU or other heavy workloads through SLURM from the repository root; see `CLAUDE.md` for launcher examples and offline model setup. Do not run heavy model jobs on the login node.

## Coding Style & Naming Conventions

Use four-space indentation and descriptive `snake_case` names. Follow nearby code; prompts belong in the relevant `prompts/` directory. No repository-wide formatter or linter is configured.

## Testing Guidelines

Use pytest and keep tests GPU-free. Place tests beside the affected module as `test_*.py`; start with focused tests, using `-k <name>` as needed, then run the full suite when practical.

## Commit & Pull Request Guidelines

Follow `[type]scope: concise description`, for example `[fix]dsi-bench: handle empty judge response`; see `CONTRIBUTING.md` for types and scopes. Keep commits atomic and separate code, data, results, and launcher changes. Pull requests should summarize changes, report validation, and link issues or include screenshots when relevant. Never add agent attribution to commit messages.

## Research and Compute Practices

Preserve the project principle that the model grounds and chooses, host code measures and computes, and ground truth is used only for scoring. For Dyn-Bench frame or time handling, read `dyn_dataset_exploration/logs/DECOMPOSITION.md` first. Keep secrets, local environment files, caches, and downloaded artifacts out of version control.

## Behavioral Guidelines

Bias toward caution; use judgment on trivial tasks. Before coding, state assumptions, surface tradeoffs, and ask when uncertain or when interpretations differ. Choose the smallest complete solution; avoid speculative features, one-use abstractions, and needless defensive handling. Keep edits surgical: follow local style, leave unrelated code untouched, and clean up only orphans your changes create. Define concrete success checks; for multi-step work, state a brief plan and verify each step. For bug fixes, add a focused regression test when appropriate.

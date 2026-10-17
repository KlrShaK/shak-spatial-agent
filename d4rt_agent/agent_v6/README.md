# AgentV6

Local Qwen plans physical quantities, calls option-blind measurement tools, and chooses the answer from their reports. Ground truth is attached only after answering. The design and implementation record are in [design/PROGRESS.md](design/PROGRESS.md) and [design/QUESTIONS.md](design/QUESTIONS.md).

M0–M8 provide a runner, shared reports, clip workspace, all eleven tools, call repair, phased controller, and offline evaluation. Model inference has not been validated on a GPU; M9 runs require separate approval.

## CPU validation

Run the small synthetic tests in the project Python environment:

```bash
python -m pytest d4rt_agent/agent_v6 -q
```

Catalog reads belong on SLURM. A dry run lists selected questions and writes only answer stubs; it never opens videos or loads models:

```bash
sbatch --partition=normal.4h --time=00:05:00 --cpus-per-task=1 --mem-per-cpu=2G \
  scripts/run_cpu_task.slurm -m d4rt_agent.agent_v6.run \
  --bench dsi --dry-run --limit 5 --results-dir /absolute/path/to/dry-run
```

Other benchmarks are `dynbench` and `vlm4d`. Use a separate results directory for live inference. Point4D is the provisional default; `--backbone d4rt` selects D4RT through the same V5 interface.

## Runtime and resume

The first live invocation freezes repository source under `<results>/source` and reexecutes there. Resume from that directory with the same absolute results path and configuration. Config and reporting-metadata hashes prevent mixing variants or catalogs. `--third-party-dir` identifies the existing pinned model repositories; `SHAK_THIRD_PARTY` supplies its default. Model weights and third-party repositories are not copied into the source snapshot.

V5's frame/geometry/SAM adapters operate in a temporary directory. One archive per source clip and backbone persists their caches. Each question has one answer JSON containing the answer, reports, call trace, process costs, and a separate scoring reference.

`--metadata-file` supplies reporting-only quantity labels and verified levels. Approved selection records also carry these labels. Dyn's existing task-to-paper-level mapping remains explicitly hypothetical until verified metadata is supplied; task and hypothetical-level breakdowns are kept distinct.

## Offline evaluation

```bash
python -m d4rt_agent.agent_v6.evaluation metrics /path/to/run --out metrics.json
python -m d4rt_agent.agent_v6.evaluation compare \
  --candidate /path/to/run --baselines baselines.json \
  --v5 v5 --frames-only frames --v4 v4 --selection-file approved-selection.json --out gate.json
```

`baselines.json` maps artifact names to run directories (or lists of directories for DSI renderings). Comparisons match question IDs and bootstrap source clips. The gate checks positive paired bounds, quantity losses, all four DSI renderings and reversed accuracy against V4.

An `audit` command uses local Qwen in separate text conversations and writes a consolidated flip audit; it must run in an approved model job. `metrics --audit-file` attaches audits without changing predictions and rejects stale audits. The parameterized `select` command implements the proposed quick-check draw; no real selection has been generated.

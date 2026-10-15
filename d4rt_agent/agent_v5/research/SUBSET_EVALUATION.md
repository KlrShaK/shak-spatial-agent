# AgentV5 subset campaign

This campaign uses the latest frozen candidate source, Point4D, local Qwen3-VL-8B,
the compositional stack, rated repair, targeted visuals, 32 steps, four repair
tool calls and a 16,000-token context. Ground truth is used only for scoring.

The DSI pool reuses the four matched 200-question / 190-video manifests with
their existing seed. The same selected IDs run in all augmentations. VLM4D draws
up to 800 questions across source and question-only kind strata, using
`taxonomy.classify(question, [], source)`. Dyn loads `--all`, draws up to 120
clips across eight sources, and retains all seven questions per clip. Both use
seed 20261014. Orders first retain rare kinds, sources and sampling cells, then
fill proportional deficits. Sampling reads no predictions or gold answers.

Pilots are 8 questions per DSI augmentation, 24 VLM4D questions, and 8 whole
Dyn clips. After the pilot, each worker selects the largest nested prefix for
which elapsed GPU time plus 1.25 times projected remaining inference time plus
1,800 seconds fits 34,200 seconds. Rates are estimated by stratum; unobserved
pilot strata use the pilot's pooled unit mean. The budget is per benchmark
worker and excludes queue waiting. Pilot answers and reconstructions are reused.
Final IDs and sizing evidence are written once to `final_selection.json`.

Create and launch from the repository root:

```bash
PY=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python
export PYTHONPATH=$PWD/.cache/subset_eval_python:$PWD
CAMPAIGN=$PWD/d4rt_agent/results/agent_v5_subset_<unique timestamp>
$PY -m d4rt_agent.agent_v5.research.snapshot --out "$CAMPAIGN/source"
$PY -m d4rt_agent.agent_v5.research.subset_eval prepare --out "$CAMPAIGN" --source "$CAMPAIGN/source"
cd "$CAMPAIGN/source"
EVALUATION_PLAN=$CAMPAIGN/plan.json sbatch --parsable scripts/run_agent_v5_subset_control.slurm
```

The CPU controller submits DSI first and VLM4D second to both existing GPU
partitions, with a ten-hour limit, one GPU, four CPUs and 48 GB RAM. The atomic
race claim protects inference. The controller records and validates both job
IDs, reconciles sibling cancellation, then starts Dyn when either benchmark
completes. DSI's worker processes its four augmentations sequentially.

`status.json` updates every 30 seconds; timestamped monitoring records and
reports update every 30 minutes. They record GPU, job IDs, configuration,
question counts, throughput, ETA, failures, recoveries and repair rounds. Fatal
logs or answer failures cancel the affected worker and stop the controller with
an explicit failure. One hour without progress is reported as a stall. There
is no automatic retry with changed source or configuration. Inspect failures
before resubmitting, and preserve the frozen source, final IDs and result paths.
Accounting database outages are recorded and fall back to the live queue;
temporary scheduler outages preserve recorded worker progress for later checks.
Controller fixes may use a separate snapshot while worker submissions continue
to use the original inference snapshot recorded in the campaign plan.

`report.json` includes accuracy by sampling cell, population-weighted accuracy,
source-video bootstrap intervals, answered/status counts, errors and represented population fractions.
DSI is reported separately per augmentation and combined by resampling shared
source videos once. DSI weights refer to its original matched 200-question
sample; VLM4D and Dyn weights refer to their full benchmark populations. Partial
results describe completed questions, not the uncompleted sample.

To inspect a campaign without changing its selection:

```bash
$PY -m d4rt_agent.agent_v5.research.subset_eval status --plan "$CAMPAIGN/plan.json"
```

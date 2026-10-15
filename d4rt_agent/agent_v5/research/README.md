# AgentV5 capability research

This directory records algorithm-design research for **all three target datasets**:
DSI-Bench, Dyn-Bench and VLM4D. All three inform the design. There is no
train-on-two/test-on-third requirement, model training, or parameter fitting in
this study. Benchmark answers are used only to score recorded predictions.

Start with [FINDINGS.md](FINDINGS.md). The executed
[capability_audit.ipynb](capability_audit.ipynb) contains the tables and figures;
[casebook.json](casebook.json) records 32 reviewed question/option cases.

The immediate objective is to establish which distinctions the questions require,
which measurements preserve those distinctions, and which kinds should share a
recipe. A larger menu is not itself evidence of better coverage.

## Reproduce

From the repository root, using the project's CPU environment:

```bash
sbatch --job-name=v5-capability-audit --time=00:30:00 scripts/run_cpu_task.slurm \
  -m d4rt_agent.agent_v5.research.audit
sbatch --job-name=v5-counterexamples --time=00:15:00 scripts/run_cpu_task.slurm \
  -m d4rt_agent.agent_v5.research.counterexamples
sbatch --job-name=v5-cached-tracks --time=00:20:00 \
  --export=ALL,PY=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
  scripts/run_cpu_task.slurm -m d4rt_agent.agent_v5.research.cached_tracks
# After all three analyses finish:
sbatch --job-name=v5-audit-notebook --time=00:15:00 \
  --export=ALL,PY=/cluster/work/igp_psr/spanwar/envs/d4rt_bw_third_party/bin/python \
  scripts/run_cpu_task.slurm -m d4rt_agent.agent_v5.research.build_notebook --execute
```

The plotting/notebook jobs use the existing environment with matplotlib and
nbclient; the default CPU environment lacks those packages. All jobs remain
CPU-only. Completed jobs: census/trace audit `17012746`, synthetic diagnostics
`17012785`, cached tracks `17013368`, executed notebook `17013489`. The initial
cached-track attempt `17013128` stopped at the missing matplotlib import and
produced no comparison results.

Generated records go to the already ignored
`d4rt_agent/results/agent_v5_capability_research_20261013/`.
Source hashes accompany results. The scripts use the main checkout, not the
separate workflow/rated-answer pilot worktree.

Success checks:

1. Census all 1,769 original DSI, 7,000 Dyn and 1,816 VLM4D questions; preserve
   question IDs and identify evidence requirements from question/options only.
2. Replay current measurement functions on exact synthetic inputs to distinguish
   contract/design failures from perception errors. Keep reference calculations
   and measured outputs, including unsuccessful hypotheses.
3. Join existing run traces to the census, report actual denominators, and separate
   missing capability, failed measurement, wrong frame/reduction, and model error.
4. Review concrete benchmark examples before recommending splits or merges.

Lexical screens are **retrieval aids, not validated semantic labels or coverage
estimates**. Existing exploration labels are also hypotheses. A synthetic
counterexample establishes a possible failure, not its frequency on real videos.
Trace accuracy is descriptive and does not establish causality. Algorithm changes
need matched before/after evaluations on each dataset and DSI's four augmentations.

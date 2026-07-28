# D4RT Agent

This package evaluates whether Qwen3-VL can answer video-based spatial
questions more reliably by using live D4RT 4D geometry as a tool.

```text
video + question
      -> exact 32-frame CPU sample
      -> Qwen tool loop
      -> cached live D4RT decoder
      -> restricted numerical calculation
      -> evidence-backed answer and trace
```

Qwen handles semantic reasoning and visual grounding. D4RT supplies 3D
positions across time. The host performs calculations in a restricted numeric
interpreter and validates the evidence cited by the final answer. Ground truth
is used only for evaluation.

## Live agent

The main entry point is:

```bash
python -m d4rt_agent.simple_v2 --point-mode centroid
python -m d4rt_agent.simple_v2 --point-mode ensemble5
```

Every video is sampled with `round(linspace(0, N - 1, 32))`. Qwen receives all
32 sampled frames as separately labelled, full-resolution images. Videos
shorter than 32 frames are rejected, and result artifacts preserve the mapping
between sampled and original frame indices.

Qwen can emit exactly three actions:

- `query_d4rt`: ground an object with a normalized bounding box and request its
  3D position at selected sampled frames.
- `python_math`: calculate from previously recorded evidence using the
  restricted numeric interpreter.
- `final_answer`: provide an answer with evidence IDs and limitations.

Malformed actions are rejected and returned to Qwen for correction. Numeric
answers must cite both D4RT and calculation evidence, and the answer value must
match a cited calculation output. Saved calculations can be replayed on CPU.

The point policy is immutable host configuration and is never chosen by Qwen:

- `centroid` queries the center of Qwen's bounding box.
- `ensemble5` adds four deterministic seed-42 offsets within the box, image,
  and a 12-pixel centroid disk. At least three finite visible predictions are
  required for a target.

The live backend encodes the sampled clip once, caches D4RT video memory, and
supports repeated `(Tsrc, Ttgt, Tcam=0)` decoder calls. It never consumes
predicted trajectories from a pre-built demo bundle. For the basketball
experiment, it reads only the saved alignment scale from demo metadata and
loads matched ground truth directly from the WorldTrack NPZ.

Run the controlled basketball comparison on A100:

```bash
sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_a100.slurm
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_a100.slurm
```

Or on Blackwell:

```bash
sbatch --export=ALL,POINT_MODE=centroid scripts/run_simple_v2_blackwell.slurm
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_simple_v2_blackwell.slurm
```

After both point-policy artifacts exist:

```bash
python -m d4rt_agent.simple_v2 --aggregate
```

The aggregate compares centroid and ensemble-5 against matched WorldTrack
ground truth. The comparison carries an important caveat: point ensembling
approximates an object-center trajectory, while WorldTrack supplies a sparse
surface-point trajectory.

Implementation details and the controlled protocol are recorded in
[V2_PROGRESS.md](V2_PROGRESS.md).

## DSI-Bench evaluation

The DSI-Bench harness evaluates the same live agent on a reproducible
25-question pilot and compares it with a tool-free Qwen control.

```bash
python -m d4rt_agent.dsi_bench_run --dry-run
```

The manifest builder selects five questions per source dataset using a fixed
Latin-rectangle coverage pattern across the six DSI task categories. The
manifest records source metadata, selected rows, video hashes, option parsing,
and ground-truth balance.

Run the agent or control on A100:

```bash
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_a100.slurm
sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_a100.slurm
```

Or on Blackwell:

```bash
sbatch --export=ALL,POINT_MODE=ensemble5 scripts/run_dsi_bench_blackwell.slurm
sbatch --export=ALL,POINT_MODE=ensemble5,MODE=baseline scripts/run_dsi_bench_blackwell.slurm
```

The runner loads Qwen and D4RT once, writes one JSON file per question, and
skips completed results when resubmitted. The baseline uses the same Qwen model
and sampled frames without geometry tools.

Review results with:

```bash
python -m d4rt_agent.dsi_bench_show --list
python -m d4rt_agent.dsi_bench_show <video-or-question-substring>
python -m d4rt_agent.dsi_bench_report
```

The report contains the sampled-frame animation, indexed contact sheet,
grounding boxes, complete tool trace, agent answer, control answer, and ground
truth for each question. Copied source videos are ignored by Git.

The recorded pilot and analysis are in:

- [DSI_BENCH_RESULTS.md](results/dsi_bench/DSI_BENCH_RESULTS.md)
- [FINDINGS.md](results/dsi_bench/FINDINGS.md)

## Code map

| File | Purpose |
|---|---|
| `simple_v2.py` | Offline Qwen loading, evidence-bound tool loop, trace replay, and CLI |
| `simple_v2_backend.py` | Cached live D4RT encoder/decoder and point aggregation |
| `simple_v2_contracts.py` | Video sampling, action validation, grounding policies, and restricted math |
| `simple_v2_eval.py` | Matched WorldTrack ground truth, scoring, and point-policy aggregation |
| `prompts/simple_v2_system.md` | Generic live-agent policy and worked tool examples |
| `dsi_bench_data.py` | DSI metadata parsing, balanced sampling, and manifest generation |
| `dsi_bench_run.py` | Restartable agent and tool-free benchmark passes |
| `prompts/dsi_bench_system.md` | DSI-specific spatial reasoning policy |
| `dsi_bench_show.py` | Individual trace inspection |
| `dsi_bench_report.py` | Visual benchmark report generation |
| `test_simple_v2.py` | CPU tests for live-agent contracts, orchestration, scoring, and aggregation |
| `test_dsi_bench.py` | CPU tests for dataset selection and benchmark behavior |

## Verification

Run the CPU suite with:

```bash
python -m unittest \
  d4rt_agent.test_simple_v2 \
  d4rt_agent.test_dsi_bench -v
```

GPU smoke-test only the live decoder with:

```bash
python -m d4rt_agent.simple_v2 --point-mode centroid --smoke
```

Qwen3-VL dependencies and weights are prepared once with
`scripts/prepare_qwen3vl_login.sh`. The SLURM launchers then run with Hugging
Face and Transformers offline modes enabled.

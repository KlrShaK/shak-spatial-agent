# Compositional AgentV5

The candidate shares algorithms across DSI-Bench, Dyn-Bench and VLM4D. All three datasets
inform capability design. Ground truth enters only the runner's scoring record; it is not
passed to decomposition, tools, repair or final answering. The implementation uses the existing
frozen models and local Qwen3-VL-8B. It introduces no training or threshold fitting.

The research basis is [research/FINDINGS.md](research/FINDINGS.md), the
[counterexamples](research/counterexamples.py), cached-track diagnostics and the answer-blind
question audit. Those identify frame selection, quantity semantics, instance identity,
temporal scope and evidence validity as separate problems.
The subsequent pilot workflow integration is documented in
[WORKFLOW_INTEGRATION.md](WORKFLOW_INTEGRATION.md), including the current answer/context defaults.

## Families and recipes

Families organize the first small decision. A kind is a recipe for acquiring evidence, not a
closed list of answers or an exclusive tool pipeline. `add_requirement` composes recipes;
`add_entity` adds participants while preserving their roles.

| Family | Kinds | Main measurements |
|---|---|---|
| motion | trajectory, body_rotation, projected_rotation, kinematics, stability | entity/camera paths, orientation, persistent-point projected rotation, path speed and variance, camera residuals |
| spatial | range, relative_position, viewpoint, region_relation, group_configuration | scalar distance, signed offsets, facing/elevation, projected boundaries and 3D surface distance, member dispersion/alignment |
| view | visibility, framing, zoom, revisit | mask presence/extent, intrinsics and translation, start/end view comparison |
| entity_sets | count_entities, select_entities | instance census, predicates, identity reconciliation and scoped reductions |
| events | locate_events, order_events, count_events | signal transitions, inspected occurrence intervals, order/count/duration/unique-participant reductions |
| visual | visual_state, interaction | targeted frames/crops and recorded visual predicates with participant/part roles |

The definitions and permitted actions live in [kinds.py](kinds.py); each recipe has a short
playbook under `prompts/playbooks/`. Old menus remain available for comparison. The annotation
concept `tracked_objects_count` is absent from the new menu: a benchmark's hidden tracked set
is not a visually observable count.

## Contracts and execution

`program.py` adds typed `Requirement`, `FrameSpec`, `InstanceSet`, `EventOccurrence` and
`EvidenceRecord` records. A requirement retains entities, frame/reference/anchor, window,
units, reduction, counting unit, predicate, event declarations and dependencies. The controller
uses sequential small decisions rather than asking the 8B model to emit a large graph at once.

`composition.py` shows the question and options at the start, then requests a primary recipe,
participants and relevant frame/count/event fields. Options are hypotheses, not observations.
Any number of complementary requirements can be added within the step budget. More than two
events can be declared, including events mentioned only in options. The default budget is
32 measurement actions and one conditional repair round of at most four calls within that budget.
A missing observation remains unresolved
when that budget expires; the VLM still chooses the final answer and reports limitations.

The final answer uses a fresh conversation with compact current reports and relevant targeted
images. Measurement questions receive no automatic overview or standard frame set; purely visual
requirements retain the adapter's standard frames. Grounding/selection revisions invalidate dependent tracks, orientations and evidence.
Revised predicates/events supersede previous assertions; revised event anchors invalidate their
resolved windows. Old conversation conclusions are not copied into the final evidence bundle.

Tools report `supported`, `partial`, `conflicting` or `unresolved`, plus quantity, entities,
revision, frame, window, units and source where applicable. These describe support, not calibrated
probabilities. Existing numeric floors remain descriptive. Host code computes quantities; the
model grounds semantic roles, interprets observations and chooses the answer.

## Measurement changes

* **Frames:** trajectory/signed-position requirements explicitly choose image, camera, levelled
  scene, world0, initial-body or current-body coordinates. An initial-body query never silently
  substitutes the first later usable orientation. The anchor/reference are preserved; a missing
  orientation is unresolved. Scalar range is separate from signed relative position.
* **Time and motion:** derivatives and irregular smoothing use actual timestamps. Phase windows
  follow elapsed duration. Reports expose generator-frame IDs for follow-up inspection, including
  sparse reconstructions and clips with unknown FPS. Path speed and net displacement rate are separate. Metric speed
  retains the calibrated `speed_mode` (default `net`) and `speed_scale_mult`; path rates use coarse,
  noise-corrected steps without bridging missing track intervals. Camera return trajectories are not labelled static from
  their endpoints alone. Orientation reports accumulated unwrapped turns and aliasing limitations.
* **Instances:** SAM3 video instance masks supply a census on the generator-frame axis without
  requiring depth. Brief tracks and all visibility intervals are retained. Identity aliases are
  explicit; a new prompt session does not establish a new physical object. No panoptic API is
  assumed. Semantic qualification uses `true`, `false`, `unknown`; geometric predicates can be
  evaluated for every candidate and combined with semantic predicates using AND.
* **Counts:** unique instances, instances at a moment and maximum simultaneous instances are
  distinct reductions. Counts describe detected candidates; no detection is not proof of absence.
  Unknown qualifying predicates and unresolved cross-session identities are reported separately.
* **Events:** visual occurrence intervals cite inspection IDs. Ordering retains onset brackets;
  repetitions require a completed occurrence. Duration is the union of intervals in the requested
  window. With unknown FPS, count/order/window/duration can use generator frames; seconds remain
  unavailable. Visibility transitions use masks directly. Arbitrary actions require inspection.
* **Passing trees:** census the category, propose encounters from visibility or travel-relative
  geometry, inspect before/during/after, then record actor/target occurrences. Count unique target
  IDs for “trees passed”; count occurrences for “times passing.” Panning and visibility alone do
  not establish a pass. Dribbles/spoonfuls likewise count occurrences or completed cycles.
* **Correspondence:** visible-surface centroids remain explicitly labelled proxies. `track_quality`
  compares them with a fixed point cohort and mask agreement. Non-image entity trajectories get
  this diagnostic automatically. Late-anchor Point4D queries run forward/backward and align to the
  existing anchor geometry; alignment residuals and anchor disagreement are retained. D4RT uses
  native source-frame queries. Neither confidence nor smoothing proves physical correspondence.
* **Geometry:** projected rotation uses persistent correspondences; ambiguous symmetric silhouettes
  or insufficient points remain unresolved. Region masks provide projected containment/boundary
  distance/tangent, separate from physical contact and 3D surface proximity. Group geometry measures
  dispersion and velocity alignment rather than relying only on a centroid.
* **Metric scale and poses:** D4RT no longer treats native depth as metres. Both backbones use the
  existing DA3 estimator with provenance. Old native-scale cache metadata is suppressed. Failed
  D4RT pose fits carry validity flags instead of silently becoming certain static evidence.

The main implementations are [capabilities.py](capabilities.py), [spatial_tools.py](spatial_tools.py),
[tools.py](tools.py), [measure.py](measure.py) and `backbones/`. The frozen Dyn decomposition
research implementation is unchanged.

## Reproducible evaluation

The default runner remains `--stack legacy` pending evidence of improvement. This means the old
controller with the corrected shared measurements. For the *original* V5 baseline, freeze a
separate baseline snapshot: `research.snapshot --baseline` restores tracked V5 source/prompts
from the repository revision while retaining the common dataset/scoring runner. Its manifest
lists every restored file and source hash. A baseline snapshot must run with `--stack legacy`.

From the repository root, with a configured Python environment:

```bash
python -m d4rt_agent.agent_v5.research.snapshot --out /path/to/candidate-source
python -m d4rt_agent.agent_v5.research.snapshot --baseline --out /path/to/baseline-source
```

Submit from the corresponding snapshot directory. Results, model dependencies and datasets
remain outside the snapshot. `source_manifest.json` is hashed into `run_config.json`.

```bash
# Candidate snapshot: 3 questions per dataset, runtime validation only.
TASK_RUN_ROOT=/path/to/smoke sbatch scripts/run_agent_v5_composition_smoke.slurm

# One benchmark/split/shard per invocation; disjoint result directories.
BENCH=dynbench VARIANT=candidate STAGE=diagnostics \
DIAGNOSTIC_MANIFEST=/path/to/diagnostic_manifest.json RUN_DIR=/path/to/diagnostics/dyn \
sbatch scripts/run_agent_v5_composition_eval.slurm

# Full DSI standard split; repeat for hflip, reverse, reverse_hflip.
BENCH=dsi DSI_SPLIT=std VARIANT=candidate STAGE=full TASK_SHARD=0/4 \
RUN_DIR=/path/to/full/dsi/std/shard0 sbatch scripts/run_agent_v5_composition_eval.slurm
```

Full evaluation is 1,769 DSI questions per augmentation, all 7,000 Dyn questions (`--all`), and
all 1,816 VLM4D questions. Shards use source clips; rerunning a job resumes its unanswered files.
Do not reuse a result directory across variants, snapshots or DSI augmentations. A completed
error record is retained in accuracy; use `--force` deliberately if rerunning failures.

`research.protocol select` generates an answer-blind 100-per-benchmark manifest from the audited
corpus, stratified by source and lexical retrieval tags. Those tags are not gold capability labels.
The 300 review entries initially have `reviewed=false`: review requirements/frame/count semantics
before calling this a human-reviewed diagnostic set. The existing 32-case casebook does not
constitute review of all 300. No leave-one-benchmark-out training split is required.

Compare the original baseline (16 actions), budget-matched original baseline (32), candidate,
candidate with legacy menus, candidate without repair, and candidate with standard answer views.
The launcher variants are `baseline`, `legacy_budget32`, `candidate`, `legacy_menu`, `no_repair`
and `standard_views`. The first two require a baseline snapshot. Keep backbone/model/cache policy
fixed within each comparison. A larger budget's gain must not be attributed solely to taxonomy.

```bash
python -m d4rt_agent.agent_v5.research.protocol compare \
  --baseline /path/to/baseline/shard0 /path/to/baseline/shard1 \
  --candidate /path/to/candidate/shard0 /path/to/candidate/shard1 \
  --out /path/to/paired_comparison.json
```

Comparison requires identical question IDs, including errors/unanswered questions. DSI
augmentation keys stay distinct while the bootstrap clusters by original source clip.
Provide all augmentation/shard directories to obtain the combined DSI estimate. Compare each
benchmark separately; report per-source/task accuracy, coverage/errors, and routing/grounding/
measurement/selection failures alongside paired gains. A positive source-clip bootstrap interval
is evidence of improvement; synthetic tests and runtime smoke alone are not.

## Validation and remaining work

The pre-workflow compositional check passed **57 tests** on 2026-10-14. GPU-free tests in
`test_composition.py` cover frame preservation, irregular time, round trips,
full turns, visibility windows, unknown FPS, identity revisions, qualified counts, occurrence
units, unsupported evidence, multiple event declarations, fresh final context and paired scoring.
Run `python -m pytest d4rt_agent/agent_v5`.

The broad repository suite should exclude generated source snapshots:

```bash
PYTHONPATH=third-party-tools/Open-D4RT:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest d4rt_agent dsi_dataset_exploration dyn_dataset_exploration \
  --ignore=d4rt_agent/results --ignore=dyn_dataset_exploration/outputs \
  --ignore=dsi_dataset_exploration/outputs
```

The initial broad check produced 613 passes, one skip and two failures. The Open-D4RT import
test subsequently passed with the `PYTHONPATH` above. The unchanged frozen Dyn test
`test_elevation_high_eye_low` also fails in isolation (`KeyError: angle`); no research-module fix
is bundled into this change. The focused V5 suite was rerun after the later additions.

Live model integration, review of all diagnostic cases, matched controls and full-benchmark
accuracy evaluation remain necessary. The GPU smoke is scheduled separately; do not interpret
queue submission as a passed test. SAM coverage, re-identification, physical contact, fast rotation,
absolute scale and tracker drift remain perceptual limitations, explicitly represented in evidence.

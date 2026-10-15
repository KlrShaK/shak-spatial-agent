# Workflow integration into compositional V5

The current working tree of `/cluster/scratch/spanwar/tmp/agent-v5-workflow-pilot` supplied
the answer parser, free-form assessment protocol, submission recovery and reporting design.
These were ported onto the compositional controller, preserving six families, 21 recipes and
the current numerical algorithms. No model training or threshold fitting occurs.

Before implementation, both sources were captured under
`d4rt_agent/results/agent_v5_workflow_integration_20261014/`. The main `source_before` snapshot
matches the frozen `source_candidate_v3` V5 hashes; the pilot capture includes its uncommitted
free-form changes. Earlier nested-JSON pilot scores do not evaluate this implementation.

## Answer flow and defaults

`--stack compositional` defaults to `--answer-policy rated_repair`:

```
measure -> current evidence packet -> assess options + compact submission
        -> optional requirement-directed repair -> rebuild packet -> reassess + submit
```

`rated` disables repair. `plain` uses the new packet without requesting scores and is the
reporting/context control. `--stack legacy` retains the legacy controller and defaults to
`plain`; rated policies require the compositional stack.

One repair round may execute at most `--repair-tool-limit 4` calls within `--max-steps 32`.
Binding, composition and failed calls consume the budget. Repair identifies an existing
requirement, option, unresolved claim, missing observation, current references and why resolving
it could change the choice. It revisits bindings/frames or composes requirements while retaining
valid evidence. Automatic gap reviews run only in the plain control. `--repair-cycles 0`
disables repair; a positive value enables one rated round regardless of the old cycle count.

Each option is assessed in prose, followed by one terminal final-answer object. Ordinal scores
describe support, not probabilities. Host code validates labels, envelopes, budgets and
structured references; the model judges support and chooses. Three submission attempts permit
one answer-only recovery. Formatting retries retain the original assessment as unverified
advisory text. Failed recovery remains unanswered.

## Final context

`--answer-views targeted` supplies no automatic overview or standard frames for measurement
or mixed questions. Purely visual questions retain `clip.answer_frames`. All active requirements
determine the policy; a visual primary kind does not make a compound question purely visual.

Packets contain question/options, relevant bindings/requirements, current evidence summaries,
unresolved limitations and selected visual artifacts. Full series and historical traces remain
in saved answers. Chronological tables, a few samples and existing turning points preserve
sequence information; compact context must not collapse a trajectory to net displacement.

* Counts include up to four cached SAM3 mask overlays with stable logical IDs. Instant counts
  use the requested frame, simultaneous counts use peak occupancy, and unique counts greedily
  cover candidates. Remaining unreviewed candidates are recorded; inspection does not prove
  detection completeness. Assembly invokes no new segmentation or reconstruction.
* Targeted packets include at most eight images. They prioritize counting checks and referenced
  inspections and retain distinct scopes supporting current claims. Identical artifacts are
  deduplicated. Caps and context trimming are recorded in the visual manifest.
* `inspect.frame_ids` selects one to eight exact GF integers. `inspect.annotation` is
  `mask`, `box` or `none`, defaulting to masks for a named entity. Crop and interval modes remain
  available. Missing masks yield a labelled plain image and a limitation, never proof of zero.
* Every generation records actual visual IDs after fitting. Trimmed/superseded artifacts are
  absent from the live reference registry. Earlier routing/answer conclusions are excluded.

`overview` explicitly adds the compact overview; `standard` adds adapter answer frames. Both
retain relevant annotations and serve as evaluation controls.

## Evidence and audit contracts

The existing `EvidenceRecord` registry is canonical. `ToolResult.evidence` contains component
records; the old `data.evidence` primary alias remains readable. Rhythm/jitter and facing/bearing
have independent availability, quality and observation coverage.

Records retain requirements, entity revisions, quantity/reduction, frame/axes/units, selected
instances, shared source IDs, observed samples/gaps, heuristic or unknown quality, limitations
and visual dependencies. Sparse observation runs are estimator-sample runs mapped to GF endpoints;
audit `sample_gfs` gives exact samples. Runs do not assert observation of every intermediate GF.
Detection completeness remains separately unknown. Unknown metadata does not trigger perception.

Schema version 3 records assessment/history, repair transitions, recovery, visual manifests and
generation costs. Input tokens are estimates; output counts require an available tokenizer.
Old answer files remain readable. Resume rejects a different/unversioned source or policy
signature; use distinct directories for comparisons, including when using `--force`.

```bash
python -m d4rt_agent.agent_v5.research.assessment_audit RUN_DIR --out audit.json
python -m d4rt_agent.agent_v5.report RUN_DIR --out report.md
```

The offline audit checks mechanically identifiable option coverage, ordinal scores,
assessment–answer consistency and citations missing from the final packet. Unparseable prose
remains unknown. Grounding and repair usefulness require manual review; prose presence and
valid JSON are not validated explanations.

## Verification and prepared evaluation

Run `python -m pytest d4rt_agent/agent_v5` for GPU-free behavioral regressions. The tests cover
numeric preservation, reporting without geometry, separate components, sparse/unknown time,
mask inspection, context policies, revision/trimmed references, repair budgets, dynamic labels,
malformed submissions, preserved assessments/recovery and compatible reporting/resume contracts.
Cached instance coverage follows the named identity. Count verification retains all census
candidates even after an earlier single-instance selection. Legacy frame-valued reductions and
turning points are translated to generator frames in the packet; original sample indices remain
in the saved tool data. The workflow integration itself preserved the top-level numerical
functions in `measure.py` relative to `source_before`.

A subsequent metric-speed correction restores the existing calibrated `speed_mode` (default
`net`) and `speed_scale_mult`. Path estimates again use `speed_step_s` sampling and
`speed_noise_mult` subtraction; steps spanning missing track intervals are excluded. Both
rates are labelled explicitly, with sampled path duration and limitations. No calibration
parameters were fitted or changed. GPU-free regressions are in `test_metric_speed.py`.
The earlier workflow snapshots and evaluation matrix predate this correction and remain
historical records; create matched sources incorporating the speed fix before resuming evaluation.

Prepare separate frozen runs for the pre-integration controller, `workflow_plain`,
`workflow_rated`, `workflow_repair`, and `standard_views`. The launcher supports the new variants.
Review the existing answer-blind 100-question manifest per benchmark before treating it as a
reviewed set. When evaluation resumes, run runtime smoke, matched diagnostics, then full
DSI/Dyn/VLM4D evaluation with four DSI augmentations and source-clip paired intervals. Use the
existing A100/Blackwell first-start claim for resumed queueing.

GPU evaluation remains stopped at the user's request. There is no new benchmark accuracy result.

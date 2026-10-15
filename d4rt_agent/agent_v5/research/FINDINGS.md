# AgentV5 Family / Kind / Measurement research

Research date: 2026-10-13. This is an evidence report and a working design,
not an implemented agent redesign or a claim of improved benchmark accuracy.

**Main finding:** expand the stack by making measurements compose around explicit
entities, reference frames, time windows and counting units. Merely adding kinds
to the existing tree would leave several demonstrably wrong measurement
semantics in place. Preserve small, guided model decisions; share computation
underneath them.

All three target datasets inform the design. There is no requirement to develop
on two and test on the third. This study performs no training or parameter
fitting. Existing constants were left unchanged in the diagnostics. The model
remains responsible for grounding, semantic judgments and final answer choice;
the host computes measurements. Ground truth scores recorded predictions only.

## What was actually done

1. Read the V4/V5 implementations, exploration studies, benchmark papers and
   separate workflow/rated-answer pilot reports.
2. Census **10,585 original questions**: 1,769 DSI, 7,000 Dyn and 1,816 VLM4D.
   The analysis corpus excludes answers and existing inferred taxonomy labels.
3. Join **5,416 historical V5 answer traces** to their original questions and
   options: 800 DSI, 2,800 Dyn and all 1,816 VLM4D.
4. Review **32 question-and-option cases** for the evidence they require. These
   are analyst text reviews, not video-grounded annotations or coverage gold.
5. Replay **12 exact synthetic diagnostics through current production
   measurement functions**. All 12 suspected failures reproduced.
6. Compare visible-surface trajectories against cached persistent Point4D tracks
   on **five selected real clips**, and inspect their contact sheets. No new
   perception inference was run.

Reproduction scripts, an executable notebook and the annotated casebook live
beside this file. Raw generated outputs and plots are in
[`agent_v5_capability_research_20261013`](../../results/agent_v5_capability_research_20261013/).
The main checkout at the start was `f809d95`; current measurement source hashes
are saved with the experiments. Historical runs retain their own provenance.
The separate workflow/rated-answer worktree is not the baseline code audited here.

## V4 and V5: what should transfer

V4 was a DSI agent with four host-driven recipes: subject motion in its own
frame, camera motion, subject-camera range change and camera bearing in the
subject's frame. It combined SAM identity masks, D4RT point trajectories and
OAV orientation. Its useful architectural property was a short, enforced
sequence from task choice through grounding to a relevant report.

V5 expanded to seven families and 23 kinds, including 21 measurement routes and
two visual routes. Its local Qwen model chooses the family, kind, entities and
tools. Point4D/D4RT provide geometry; SAM provides masks; OAV supplies facing.
The broader menu inherited much of the Dyn decomposition's measurement layer,
but V5 lets the model choose the final answer. The baseline hides answer options
until after measuring, even when they contain necessary entities or quantities.

The historical V4 200-question DSI evaluation reported 66.5% on standard videos
and 60.4% averaged across four augmentations, versus 58.5% and 41.5% for its
matched visual control. Those figures are not directly comparable to V5's
different 800-question standard-only run. The existing 100-question overlap
analysis reported V5 50%, V4 67%, control 59%.
See [V4 results](../../results/agent_v4_random200/PER_CATEGORY_VS_PUBLISHED.md)
and [V5 DSI comparison](../../results/agent_v5_dsi800_point4d_20261010T012544Z/RESULTS.md).

**Keep V4's constrained recipes, not its DSI-only taxonomy.** V2/V3 previously
lost to their control despite live tools, and the latest rated-answer pilot
again shows that richer model output schemas can fail operationally.

## Baseline behavior: tools present does not mean questions covered

Recomputed from saved answer files; unanswered questions count as incorrect.

| Historical Point4D run | Correct / evaluated | Accuracy | Unanswered | Any explicitly unavailable measurement | All attempted measurements unavailable |
|---|---:|---:|---:|---:|---:|
| DSI standard subset | 411 / 800 | 51.38% | 4 | 112 | 78 |
| Dyn evaluation subset | 1,646 / 2,800 | 58.79% | 3 | 289 | 205 |
| VLM4D complete | 1,000 / 1,816 | 55.07% | 24 | 354 | 280 |

These are different evaluation populations, not a pooled leaderboard. Some tool
records lack a universal availability field; absence of `available=False` does
not establish successful grounding, correct semantics or sufficient evidence.

Concrete findings from the joins:

- **DSI reference frame:** among 339 evaluated category-0/1 subject questions,
  whose templates specify the subject's starting orientation, 262 used only
  scene-frame track calls; 74 attempted an own-frame track. This is a task/tool
  mismatch before deciding whether the geometry itself is accurate.
- **DSI compound motion:** 414 of the 1,769 original questions have a rotation,
  turning or orbit expression somewhere in the options, with none found in the
  question text by this lexical screen. In the evaluated subset, those 186
  screened questions score 59/186 (31.7%). This is a retrieval-defined slice,
  not an independently validated rotation-task label.
- **VLM4D counting:** the count-word screen retrieves 254 questions, scoring
  96/254 (37.8%). Of these, 203 route to `instance_count`, including examples
  requiring filtered entities, dribbles or spoonfuls. Twenty route to
  `tracked_objects_count`, a kind intended for hidden benchmark annotations.
- **VLM4D rotation:** 254 of 317 rotation-screen questions route to
  `traj_subject_own_frame`; 125 of the 317 have every attempted measurement
  explicitly unavailable. Body-facing, wheel spin and orbital motion require
  different observables despite sharing rotation words.
- **Dyn event decomposition:** saved programs contain 56 `other` and four
  `stops_moving` event predicates. The baseline tool explicitly cannot measure
  either. These are event records, not 60 distinct questions.
- **VLM4D presence:** the existing answer-blind absence-option parser identifies
  795 questions with an absence alternative. Our deliberately narrower generic
  lexical screen finds only 229; do not mistake that screen for coverage.

The full overlapping screen tables are in
[`census_tables.md`](../../results/agent_v5_capability_research_20261013/census_tables.md).
For example, signed-relation expressions occur in 809 Dyn questions/options,
721 in options only. Such counts can include contextual mentions and distractors;
they retrieve review cases rather than establish exact capability demand.

## Controlled experiments: fix the meaning of existing measurements first

These tests bypass perception errors where possible. They demonstrate possible
failures, not the fraction of real questions affected.

| Diagnostic | Current measured behavior | Required distinction |
|---|---|---|
| Reflected pair positions | Identical range reports for offsets −1 and +1 | Scalar distance cannot identify left/right |
| Round trip, 4 units in 6 seconds | Reported average speed 0; path-speed field 0.667 | Speed and net-displacement rate are different reductions |
| Camera travels out and back | Dominant motion `static`, despite excursion 2 and path length 4 | Static requires low excursion/path, not zero net translation |
| Valid 2D track, no depth | Image-frame track unavailable | Image displacement must not require a finite 3D position |
| Entry requested in seconds 3–5 | Reports first entry at 0.5s, instead of 3.5s | Event localization must honor the query window |
| Two exits at 1.5s and 4.5s | First-event tool returns 4.5s | All transitions are needed before selecting first/last/occurrence |
| `stops_moving` predicate | Explicitly unavailable | Schema support is not measurement support |
| Count in seconds 0–1.5 | Counts three entities, although only one appears in that interval | Count needs window and set semantics |
| One complete 360° turn | Final rendered `turned` value is 0° | Wrapped orientation is not cumulative rotation; raw rows retain more information |
| Constant speed, irregular sample times | Reported speed ranges 0.182–1.818 instead of 1 | Differentiate using actual timestamps |
| First half of an irregularly sampled 11s clip | Window ends at 1s rather than target 5.5s | Percentage windows refer to elapsed duration |
| Fixed surface, changing visible portion | Apparent displacement 3.286 with residual noise 0.014 | Visible-surface median is not a persistent physical reference point |

The last experiment matters beyond any one kind. A biased correspondence can
look smooth and therefore receive a small noise estimate. Adding more reducers
on that track propagates the error into direction, range, speed, events and
filtered counts. Track reliability must include identity/correspondence evidence,
not just smoothing residuals.

See [counterexample outputs](../../results/agent_v5_capability_research_20261013/counterexamples.json)
and [experiment implementation](counterexamples.py). Time-sampling failures are
conditional on irregular sample times; their prevalence in existing caches has
not been measured. The event-window case also sets the program window, so the
failure does not depend on the unsupported per-call window argument.

## Real cached geometry: persistent tracks are promising, not an automatic fix

For five deliberately selected VLM4D/DAVIS clips, compare the existing
visible-surface median with median displacement of the fixed frame-0 query
points inside the selected SAM instance. No confidence cutoff was fitted or
applied. Project those tracked points into each frame and compare with the
selected SAM mask. This checks agreement, not ground-truth 3D accuracy.

| Clip / target | Initial seed points | Mean projected seeds inside mask | Endpoint disagreement, scene scales |
|---|---:|---:|---:|
| varanus-tree / lizard | 53 | 47.4% | 0.927 |
| bear / bear | 72 | 88.2% | 0.079 |
| bike-packing / bike | 49 | 74.8% | 0.128 |
| bmx-trees / rider | 13 | 41.0% | 0.808 |
| carousel / toy horse | 68 | 25.3% | 0.993 |

Contact-sheet inspection shows reasonable agreement on much of the bear clip,
but substantial projected-point/mask disagreement on the lizard, rider and
carousel. The lizard's surface trajectory also has a large late jump. This does
**not** justify replacing every median track with unfiltered Point4D tracks.
Both estimates can fail; visible-mask disagreement can also result from occlusion.
Point4D confidence values in these caches are not probabilities and must not be
treated as such.

The useful algorithmic direction is to compare correspondences, retain persistent
identity, support reseeding when the target appears late, expose coverage/gaps,
and check agreement between projected tracks, masks and camera geometry. Resolve
large contradictions before emitting confident direction/speed labels. A
geometric centre, a tracked material point and a semantic body anchor should not
be silently interchangeable.

See [cached-track results](../../results/agent_v5_capability_research_20261013/cached_tracks.json),
[lizard contact sheet](../../results/agent_v5_capability_research_20261013/contact_varanus-tree.png),
[carousel contact sheet](../../results/agent_v5_capability_research_20261013/contact_carousel.png)
and [comparison code](cached_tracks.py). These five clips do not estimate
dataset-wide tracking quality or establish which estimate is more accurate.

## What the taxonomy should represent

The current families mix **actor** (`camera`), **relation** (`two_entities`),
**aggregation** (`count`) and **reasoning operation** (`events`). An object count
conditioned on a spatial relation naturally belongs to multiple current families.
That is evidence for composition, not a new family for each combination.

Separate three layers:

1. **Family:** a small routing menu for the main reasoning operation.
2. **Kind:** a compact default recipe for an observable or query operation.
3. **Measurement:** a typed computation over grounded evidence, with explicit
   actor/reference, frame and frame anchor, window, reduction, units and validity.

The same range computation should serve camera-object and object-object queries.
The same event intervals should serve event order, duration, repetition and an
event-relative spatial query. Entity count and event count must still specify
different units. A kind should make relevant tools accessible; it should not
pretend that selecting the kind establishes evidence coverage.

A requirement such as `count distinct cyclists moving image-left during W`
should retain **set + predicate + frame + window + aggregation**. A question
about left-hand spoonfuls instead needs **event occurrences + actor/hand
condition + counting unit**. Keep this typed state in host code while asking the
8B model one small decision at a time. Do not require it to emit a large nested
program or score every claim in one response.

### Candidate family menus to test, not a frozen 21-kind specification

| Family | Candidate kinds | Shared evidence/computation |
|---|---|---|
| Motion | trajectory; body rotation; projected rotation; kinematics; stability | 2D/3D tracks, camera/object pose, unwrapped angles, actual-time derivatives |
| Spatial relations | range; relative position; viewpoint; region relation; group configuration | Relative vectors, body-frame bearings, region geometry, pairwise/group statistics |
| Image/view state | visibility; image extent/framing; lens zoom; view overlap | Masks/boxes, image occupancy, camera intrinsics, background correspondence |
| Entity sets | entity count; identity by predicate | Candidate identities, per-instance predicate evidence, filtering and aggregation |
| Events | event localization; event order; event count | Preserved predicates, state transitions, onset/end intervals, occurrence identity |
| Visual semantics | visual state/activity; interaction state | Targeted frames/crops, actor/object/part roles, semantic state evidence |

Use an actor parameter for camera versus object wherever the quantity is shared.
Use a reference parameter for subject-relative orbital paths. Explicitly keep
projected spin separate from body yaw. Keep semantic event localization available
when geometry cannot identify an action; host code then orders/counts the located
intervals. Handedness is a role/frame requirement, not a universal image-left rule.

Whether `trajectory` needs separate direction/path submenus, and whether the two
visual kinds should remain one, are still empirical routing questions. The prior
six-family/22-kind proposal should not be treated as settled by this audit.

## Keep, merge, split or replace in the current stack

| Current kinds/tools | Proposed treatment | Reason |
|---|---|---|
| `camera_path`, `camera_translation`, `subject_direction`, `traj_subject_own_frame` | Share trajectory computations; expose actor, frame, anchor and reduction | Four routes partly duplicate position change; defaults currently select wrong frames |
| `cam_subject_range`, `pair_relation`, `metric_distance` | One range primitive; preserve trend/value and unit requirements | Same underlying scalar quantity; separate signed relations from it |
| `metric_speed`, `subject_motion_profile`, `camera_rhythm` | Shared kinematics; distinct path speed, net velocity, acceleration and interval comparison | Existing names mix quantity, units and temporal reduction |
| `camera_shake` / `camera_profile` | Reuse pose series; keep stability semantics distinct from speed | Pose residual includes estimation error; do not equate it automatically with physical shake |
| `camera_viewpoint`, `cam_subject_elevation`, `orientation` | Reuse pose/relative-vector primitives but return distinct quantities | Bearing, elevation and body heading have different frames and meanings |
| `orbit_direction` | Replace heuristic with relative trajectory around a defined reference | Opposing lateral motion/pan does not establish an orbit around the named subject |
| `landmark_relation` / `region_relation` | Retain nearest-surface distance; add boundary/tangent/containment predicates | Along, across, inside and contact are not nearest-distance trends |
| `subject_visibility`, `apparent_size` | Share mask evidence; retain separate presence and area/framing reductions | Existence, visibility, centre/edge and size are different questions |
| `camera_zoom`, `scene_revisit` | Keep distinct recipes; share camera/background evidence | Focal change, camera approach, apparent enlargement and revisiting are not equivalent |
| `instance_count` | Add typed predicates, windows and per-instance evidence | Raw instance count does not answer filtered counts or action counts |
| `tracked_objects_count` | Remove as a general visual skill; retain an explicit limitation for annotation-defined counts | Hidden annotation counts cannot generally be recovered from RGB alone |
| `event_order` / `event_onset` | Build occurrence intervals first; derive order/count/duration | First/last appearance is insufficient; `other` must not erase event text |
| `semantic` | Give it targeted evidence collection and typed state/event outputs | Narrative or hand-object interaction needs visual reasoning, not fabricated geometry |

The measurement interface needs a contract for **what can be concluded**, not
just `available=True`. For example, valid range evidence remains insufficient
for a left/right claim. Record support scope, identity, time coverage, frame,
units, provenance and unresolved dependencies. Structural checks should guide
the next measurement; they should not replace the model's final answer decision.

## Concrete implementation directions supported by this evidence

**First, correct existing contracts.** In [tools.py](../tools.py),
[measure.py](../measure.py), [scene.py](../scene.py) and
[render.py](../render.py): distinguish net/path quantities, preserve angular
unwrap, allow depth-free image tracks, honor windows in events/counts, use
timestamps for derivatives and phase selection, and separate confidence in
correspondence from smoothing residual. Reuse these diagnostics as focused
regression cases when implementation starts.

**Second, make requirements explicit before routing.** In
[program.py](../program.py), [kinds.py](../kinds.py),
[orchestrator.py](../orchestrator.py) and their prompts: show the question and
options before selecting the measurement recipe; retain explicit frame anchors,
option-defined actors/events, counting units and all meaningful predicates.
Use small validated decisions. Treat multiple required quantities as a bounded
recipe, not as repeated arbitrary kind switching.

**Third, add genuinely missing computations.** Signed relative vectors and
frame transforms; predicate-filtered entity sets; event occurrence intervals and
conditional event counting; projected rotation; region/boundary predicates;
group dispersion/alignment; relative speed and option-defined interval
comparisons. Reuse the same typed primitives across all three benchmarks.
Introduce a new frozen perception model only when a measured failure shows that
the existing evidence cannot support the required observable.

**Fourth, validate entity/time evidence before richer answering.** Version
grounding and instance selections, invalidate dependent caches, and supersede
stale reports. Targeted temporal/crop inspection should resolve an identified
ambiguity. Do not infer absence from failed segmentation, static from missing
motion evidence, or anatomical left from image left.

The latest [rated pilot](../../results/agent_v5_rated_pilot_20261013/REPORT.md)
scored 105/168 versus baseline 98/168 on reused development clips, but only
52/168 and 57/168 runs retained complete structured assessments in the two rated
variants. Neither variant executed a repair. Most decisions fell back to
answer-only recovery, and the report marked both variants ineligible for merge.
This supports simplifying individual model decisions, not claiming that the
rating/repair algorithm has been validated.

## What the external research adds

DSI explicitly probes coupled observer/object motion and flip/reversal
robustness. Its reported rotation/translation confusion supports keeping those
observables separate and evaluating all four augmentations, rather than trusting
a standard-video accuracy increase alone.
[DSI-Bench paper](https://arxiv.org/html/2510.18873v1).

Dyn reports Qwen3-VL-8B results of 53.8% with raw video, 53.8% with masked frames
alone, and 57.1% with mask-guided fusion. This motivates a controlled comparison
of raw frames plus our predicted, verified masks/correspondences. It does not
justify using benchmark annotation masks or assuming the same gain for V5.
[Dyn-Bench paper, Table 6](https://arxiv.org/html/2603.12746v1).

VLM4D includes translation, rotation, spatiotemporal counting and false-positive
queries. Its definition of counting explicitly includes objects performing
particular motions; therefore plain instance count is not sufficient by design.
Its Ego4D sources also make actor/hand/frame distinctions relevant.
[VLM4D paper](https://arxiv.org/html/2508.02095v2).

## Remaining experiments before committing to the final stack

- Compare the current hierarchy and candidate quantity-based menus on the same
  representative questions from **all three datasets**, using local Qwen and
  human-reviewed frame/quantity/counting-unit requirements. Measure invalid
  decisions, calls and coverage, not just whether a kind label sounds plausible.
- Evaluate correspondence checks and reseeding on clips with camera following,
  occlusion, articulation, late appearances and changing visible surfaces.
  The five-cache study rejects an unconditional fixed-track replacement; it does
  not yet select a best track-construction algorithm.
- Implement/compare corrected existing measurements separately from new
  primitives and from reporting changes. Include simple positive controls as
  well as the failure cases. Changing a menu and three measurements at once
  obscures which change caused a gain or regression.
- Establish matched visual controls for all three benchmarks, including VLM4D;
  report each benchmark's before/after accuracy, failures, calls, latency and
  evidence correctness. Include identical extra visual observations in the
  control when testing the value of geometry.
- Use clip-level paired comparisons; DSI augmentations from one source clip stay
  together. Dyn and VLM4D share source footage, so a pooled confidence interval
  should also account for that dependency. This is an evaluation accounting
  issue, not a prohibition on using all three datasets for algorithm design.

At this stage there is strong evidence for the **required distinctions and
several contract fixes**, but no new end-to-end result proving that the proposed
family menus or added tools improve accuracy. The next implementation should be
driven by these failure cases and controlled comparisons, not by the number of
kinds added.

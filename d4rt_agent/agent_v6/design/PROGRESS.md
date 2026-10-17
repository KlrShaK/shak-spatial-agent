# AgentV6 implementation progress

Implemented M0–M8 on `feat/agent-v6` in the separate `shak-spatial-agent-v6` worktree. The main checkout remains on its original branch. Existing V4/V5 modules are imported, never edited. No GPU jobs, real quick-check selection, benchmark inference or flip-audit inference have run.

## Plan and success checks

- M0: runner, atomic answers and model-free dry runs; synthetic tests plus real catalog checks on CPU SLURM.
- M1: option-blind reports and merged phases; synthetic direction, noise, time and row-limit checks.
- M2: clip workspace, named instances, revisioned caches and both backbones; synthetic grounding, frame and invalidation checks.
- M3: camera translation/rotation, zoom and revisit; synthetic roll, orbit, out-and-back and dolly checks.
- M4: entity, orientation and group measurements; synthetic relative frames, side changes, full turns, speed and fallback checks.
- M5: inventory, masks, frames and isolated video helpers; synthetic visibility, overlays, sampling and caption-cache checks.
- M6: normalization, schema validation, bounded doctor and answer recovery; malformed calls replayed with mocked Qwen.
- M7: entity/plan/act/answer workflow; six scripted examples, context retention, budget and real tool integration checks.
- M8: parameterized answer-blind selection, metadata, metrics, paired clip comparisons, flip audit and gate; synthetic evaluation checks.

Each milestone received a fresh GPT 6.1 Sol review at medium effort. Findings were fixed and regression checks added. Code, tests and documentation are committed separately. Provisional choices and implementation limits are recorded in `QUESTIONS.md`; numbered design decisions are unchanged.

## M0 — Scaffold and runner

Built the V6 CLI using V5 loaders and selection, one atomic answer JSON per question, model-free dry-run stubs, source snapshot execution and resume signatures, and a temporary working cache persisted as one clip/backbone archive. Ground truth is attached after solve only. `--third-party-dir` points frozen runs to the existing pinned model repositories.

Validation: 9 synthetic tests pass, covering all three dry-run adapters, scoring references, overwrite protection, archive round-trip and frozen-source reexecution. The CPU SLURM check covers the real expected catalog counts (DSI 1,769; Dyn 2,800; VLM4D 1,816) and five-stub dry runs. The stalled pending job 17403791 was replaced by direct normal.4h submission 17404184, pending at this update; these real counts are not yet claimed as verified.

Fresh review found mismatched snapshot hash serialization, execution continuing from mutable sources, and missing catalog identity in resume signatures. Fixed all three: reexecute frozen source before inference imports, hash manifest bytes consistently, and include the enriched catalog hash. External model-library PYTHONPATH entries are preserved after the frozen source. Final live-wiring review also found relative model/import paths breaking after the cwd change; existing local model paths and retained PYTHONPATH entries now become absolute before reexecution, while Hugging Face IDs remain unchanged.

## M1 — Report layer

Built Report/Phase records, majority/minority headlines, chronological tables capped at six display rows, full/short contracts, units/provenance, safe time labels and unavailable reports. Adapted V4 contracts while preserving geometry signs.

Validation: 20 synthetic tests pass, including the design's literal rendering fixture, coherent slow motion, a brief strong reversal, out-and-back, jitter, diagonal motion and phase packing. Fresh review found lost slow motion, lost excursions during uniform resampling, and a component/norm threshold mismatch. Fixed with cumulative coherent displacement, source-sample-aware simplification and a Euclidean noise floor. Suppression of short weak phases is explicit in reading notes.

## M2 — Workspace and geometry access

Built lazy V5 Models/SceneCache access for Point4D and D4RT; named entities and stable instance labels; explicit candidate choice; revisioned track, facing and dependent caches; implicit camera; metric provenance; seconds and shown-frame mapping.

Validation: 14 synthetic tests pass. Fresh review found that report rendering could register unseen frames. Fixed with explicit `mark_shown` registration and report labels mapped to already displayed frames. Unknown fps without a displayed frame produces a clear unavailable result.

## M3 — Camera tools

Built camera translation/pan/tilt/roll phases, tempo/jitter, orbit-sign contracts, focal-ratio versus forward motion, and geometric start/end revisit. Host-only windows preserve the original time origin. Invalid geometry produces unavailable reports.

Validation: 22 synthetic tests pass. Fresh review and follow-up checks cover roll-only motion, out-and-back, clockwise orbit, zoom/dolly separation, finite positive endpoint intrinsics/depth, and invalid geometry. Missing final depth cannot silently become a negative revisit result.

## M4 — Entity, orientation and group tools

Built own and relative motion in world/camera/subject coordinates, distances/bearings/elevations and side phases, camera-corrected unwrapped headings, group directions/spread/membership, and the three-crop eight-label facing fallback. Speed uses path length divided by observed time and reports net displacement per time separately. Missing facing preserves world/camera motion.

Validation: 35 synthetic tests pass. Fresh review found incorrect off-axis fallback conversion, invented turns across missing orientation spans, lost distance observations, and incomplete group temporal summaries. Fixed the camera-ray correction, split valid facing runs, retained distance independently of facing, and added group extrema/phases and direction counts. Own motion uses starting facing; relative observations use instantaneous facing. Full turns retain 360 degrees. Short reading contracts and composed temporal row limits are covered.

## M5 — Inventory, frames and VLM helpers

Built whole-clip mask inventory and visibility/size/edge summaries, four-frame range views, stable colored candidate montages and explicit selection. Captioning is question-blind and cached per clip. `ask_video` runs an isolated conversation. Both use the local Qwen wrapper with source timestamps, 2 fps sampling and a bounded visual-token budget; unknown fps uses the main agent's eight labeled keyframes.

Validation: 18 synthetic tests pass. Fresh review and integration fixes ensure every candidate appears in the montage, pending choice is partial evidence, captions are keyed by prompt/model/sampling settings, counting wrappers preserve video metadata, and odd frame counts reserve temporal-padding tokens. Installed processor source was inspected; a slow CPU import smoke was terminated. Actual model/processor execution remains for M9.

## M6 — Tool-call robustness

Built small phase-aware schemas, balanced outer-envelope parsing, local normalization, clear grounding failures, a five-call shared doctor budget, and answer recovery. Repairs preserve already-valid values; conflicting aliases are returned to the model. Invalid calls do not consume the measurement budget; corrected calls consume one call.

Validation: 24 synthetic tests pass. Fresh review exposed nested-object extraction, valid-value mutation through conflicting aliases, and ambiguous prose answers being accepted. Fixed outer-envelope decoding, immutable valid fields and positive unambiguous declarations. Recovery rejects alternative letters, negation and conflicting final declarations; it cannot create an undeclared choice.

## M7 — Controller

Built caption/keyframes → entities → plain-text plan → act → final-answer phases, tool visibility, category hints, plan revision, 14-call budgeting and warnings on calls 12–14. Full schemas/contracts appear first, then shorter reminders. Final instructions enforce evidence precedence and the V4 answer format. Process traces count model/tool/doctor calls, tokens and model-declared completed plan items.

Validation: 13 controller tests and 4 real-tool integration tests pass, including all six scripted design examples. Fresh review found premature plan completion and missing generation reservation in context checks. Completion now requires explicit model declarations; revisions reset completion. Trimming removes requested images first and preserves reports, plans and opening keyframes. Generation tokens are reserved before every main-agent call. Integration covers camera measurement, candidate choice followed by measurement, missing-facing fallback, and separate caption/video calls.

## M8 — Evaluation harness

Built deterministic parameterized selection, reporting-only metadata enrichment, benchmark/quantity/task/level metrics, DSI consistency, process/cost statistics, paired source-clip bootstrap comparisons, offline flip audit and gate checks. Approved selection coverage can be checked exactly. DSI renderings are clustered by source clip. No real selection or model audit has run.

Validation: 30 metadata/evaluation tests pass. Fresh review found incomplete metadata propagation, unsupported claims of official Dyn levels, unused audit results and missing coverage checks. Fixed manifest/selection enrichment and provenance, separated hypothetical Dyn levels from supplied levels, attached fingerprint-checked audit files to metrics, and enforced ID/rendering coverage. Missing ignored metadata in worktrees/snapshots falls back lazily to the shared repository artifact; explicit metadata paths retain precedence.

## Final validation and handoff

- Final full V6 suite: **189 passed in 12.07 seconds**, including metadata fallback and snapshot launch-path regressions.
- Reused V4/V5/trajectory tests: **62 passed, 10 skipped in 7.92 seconds**. Skips require an absent saved trajectory-run fixture.
- All tests use small CPU fixtures without model weights or dataset reads. Real catalog validation stays on SLURM.
- No changes to V4/V5, model weights, datasets or generated benchmark results are included.
- M9 GPU smoke/quick-check/benchmark runs remain outside this implementation handoff and require approval per run.

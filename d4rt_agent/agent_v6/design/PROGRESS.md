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

## Review corrections before M9 — 2026-10-18

Addressed the 14 reported correctness findings in the original checkout on `feat/agentv6-draft`.

- **Report words and magnitudes:** absorbed phases are relabelled from their combined displacement, including after adjacent phases merge. Stillness at 50% or less is described as the largest time share. Camera headlines and phase tables expose ranked net components. Entity speed uses the same filtered phase distances as the motion report; absorbed weak jitter in a still phase contributes only unresolved endpoint drift.
- **Subject axes:** use the first resolvable facing with a valid position inside the question's interval, and state its reference time. Missing start/middle/end observations can use the visual fallback while retaining native facing elsewhere. Turns are not accumulated across changes between native and coarse facing. Group motion in each entity's own frame follows the same reference rule. Camera references bypass entity facing and cropping.
- **Answer recovery and doctor:** prose option tokens are case-sensitive, so articles do not become choices. Explicit JSON answer keys and literal letter values tolerate small formatting variations. A doctor can recover an unusual explicit decision, with source checks that reject missing, conflicting, ambiguous, negated or merely described options. Allowed `To the agent:` guidance reaches the conversation unchanged.
- **Controller:** answer declarations transition to the answer instructions without using the tool doctor. Doubled braces produce a recoverable parse error. A tool's full reading contract is considered delivered only when it actually appears in a report.
- **Question state:** host inventory probes do not consume the agent's grounding rename attempt. Every question resets shown-frame labels while retaining expensive clip observations.

Validation: **253 passed in 11.81 seconds** with `python -m pytest d4rt_agent/agent_v6 -q`, including 57 added regression cases and the real-tool scripted integrations. `git diff --check` passes. Tests use small CPU fixtures and mocked models.

M9 and the quick-check/full evaluation remain paused following the user's `wait`. No GPU evaluation was launched or resumed for these fixes.

## M9 — Evaluation resumed, 2026-10-18

The user requested resuming GPU evaluation, retaining the earlier authorization for full runs conditional on the quick-check gate. The campaign uses seed `20261018`: 40 DSI IDs across all four renderings, 70 Dyn questions, and 70 VLM4D questions. Full runs remain conditional on every benchmark passing design 15's gate, then run in DSI → Dyn → VLM4D order.

CPU preparation job `17436851` freezes corrected source, benchmark manifests, answer-blind selections and existing baselines under `d4rt_agent/results/agent_v6_eval_20261018`. Compatibility checks cover V4's `answer_text`, the DSI control's `baseline/` records and `parsed_letter`, and the official Dyn Qwen records. No new control or ablation is authorized by this campaign. The SLURM controller records progress every 30 minutes; the Monitor tool is unavailable in this session, so scheduler status and saved logs are used.

Preparation `17436851` stopped before inference because the original Dyn catalog and `eval_rows.json` were deleted. The user confirmed the deletion was to reduce file count and authorized regenerating only needed artifacts. Replacement preparation job `17437983` restores the exact 2,800 IDs from the saved V5 prediction set, validates them against the raw benchmark, and recovers logged frames-only/letter/evidence decisions. Missing D7 predictions are regenerated only for the frozen 70-question Dyn quick-check, using the committed `calibrated.eval.json` without refitting. They are labelled `d7_regenerated` in comparisons. Frames and intermediate files stay temporary; one archive per clip and one consolidated prediction map are retained.

Persistent controller `17438155` depends on successful preparation, then performs smoke → required D7 regeneration → all quick-check benchmarks → gate → conditional full runs. The separate pending runtime smoke `17437353` was cancelled to avoid duplicate work. Current results live under `d4rt_agent/results/agent_v6_eval_20261018`; no GPU result or gate outcome is claimed yet. Campaign regressions cover historical record formats, dependency order, selected-only regeneration, packed clip caches, resume and promotion checks.

The exact Dyn catalog was reconstructed and all original reference answers/options validated. Existing log recovery found 1,925 frames-only, 1,928 letter and 1,548 evidence decisions; missing frames-only answers are regenerated only when needed by the selected quick-check. The controller also waits for preparation completion job `17439030`, which exits immediately if the frozen plan is already complete and otherwise finishes preparation with a 30-minute reservation. Test-only scheduler estimates placed Blackwell ahead of A100; actual starts remain scheduler-dependent.

Campaign GPU smoke `17439251` is queued with an `afterok:17439030` dependency. The controller state already names this job, preventing a duplicate submission. Historical Dyn references with lowercase gold letters are normalized to the canonical option keys, preserving the official stored scores. D7 regeneration can reuse existing V5 Point4D scenes only when the generator-frame clock, image sizes and geometry stride match; reused files receive recorded hashes, and missing/invalid metric scales are recomputed when needed.

The user subsequently reported an empty queue. Accounting confirms preparation `17439030` failed, and dependent controller `17438155` and smoke `17439251` were cancelled before any GPU inference. The blocker was a stored paper result whose `validation_7` ID describes translation while the current benchmark uses rotation. Paper comparisons now require matching question, option mapping and ground-truth text; incompatible records are excluded with a frozen per-model audit, without changing benchmark identities, selections or required V5/control comparisons. Campaign and evaluation regressions: **32 passed**, including reused-ID reference checks. Replacement CPU preparation `17441680` is queued; preparation must finish successfully before submitting the replacement controller.

Replacement preparation `17441680` validated all baselines but stopped while writing outputs: the shared JSON writer requires `Path` rather than string arguments. Both output sites are corrected. Numeric paper choices are normalized using the same text conversion as the benchmark loader, retaining equivalent counting questions. Replacement preparation `17442478` performs the final freeze; no GPU inference has run yet.

Preparation `17442478` completed successfully (exit 0, 7m15s). The frozen plan has 300 quick-check records, 14 verified input hashes and 568 source files. Only 62 genuinely incompatible VLM4D paper records are excluded (Grok 28 and GPT-4o 3, for each of two prompts); numeric counting answers retain coverage. Blackwell smoke `17443409` was submitted without a preparation dependency using a four-hour reservation. Its active state is recorded before submitting the CPU controller, preventing duplicate GPU submissions. The focused campaign/evaluation fixtures pass **33 tests**.

Persistent controller `17443428` is submitted against the completed frozen plan and adopts GPU smoke `17443409`. Sequence: smoke → selected missing Dyn comparisons → DSI/Dyn/VLM4D quick-check → all-benchmark gate → conditional DSI/Dyn/VLM4D full evaluations. The GPU smoke was verified pending on priority; no inference result or gate outcome is claimed.

## GPU race scheduling correction — 2026-10-18

The user reiterated #73: submit A100 and Blackwell together and keep the first starter. Queue estimates must not replace the race. The single-platform CPU controller `17443428` and still-pending unguarded Blackwell smoke `17443409` were cancelled. Replacement smoke pair: Blackwell `17449877`, A100 `17449917`, with shared attempt `c384a9b90b754115845bf3590e304439`. The existing atomic `scripts/race_claim.sh` selects one worker and cancels its sibling before model setup. The controller now submits one race pair for each stage and monitors the claim winner, including interrupted-run resumes.

Frozen inference source, original source hash, all 14 input hashes, question selections and gate policy are unchanged. Revised scheduling code and launcher are separately frozen under `orchestration/race_20261018/manifest.json`. Twenty campaign tests pass, including simultaneous-start single-worker execution, sibling cancellation, loser completion handling, paired submissions and cleanup on partial submission failure.

Replacement persistent CPU controller `17449940` adopts the race pair and applies the same race scheduling to subsequent stages. Both GPU contenders were verified pending in Slurm.

The user requested moving the supervisor onto the login node. CPU supervisor job `17449940` was cancelled before execution. A detached, low-priority, singleton-locked process was launched on `eu-login-11`, PID `1884529`, using the frozen scheduling revision; process metadata is in `jobs/control_login.json` and output in `jobs/control_login.out`. Its first poll adopted both pending contenders, and the status file records `controller_job: null`. GPU inference stays in the A100/Blackwell race workers.

## Slurm request audit — 2026-10-18

At the user request, compared both live race requests with completed V5 jobs 16167131, 16174484, 16174481, 16345919 and 17033191 using accounting plus saved launchers/logs. GPU count, CPU cores (4), RAM (48GB), account, QoS and effective CPU constraints match proven requests. The 24GB temporary-disk request exceeds the old 16GB request but is comfortably within both planned nodes' advertised disks. Slurm has evaluated both jobs recently and assigned eligible planned nodes; both wait on Priority without holds or dependencies. No request changes or requeue were needed. Detailed audit: `jobs/slurm_parameter_audit_20261018.md` in the campaign results.

## First GPU smoke and processor correction — 2026-10-18

A100 `17449917` won the race on eu-a65-06, cancelled Blackwell `17449877`, loaded Qwen on A100 80GB, and failed at the first caption request (exit 1, 4m16s). The first question recorded `TypeError: string indices must be integers`; no valid quick-check prediction or gate result was produced. The supervisor correctly stopped the campaign on the infrastructure failure.

The installed Transformers ProcessorMixin tokenization traverses every message as a list of typed content blocks. VideoQwen previously forwarded bare system/text strings. The V6 adapter now converts strings to text blocks for both video and image/text calls while preserving media objects, metadata, prompts and caller history. Existing V4/V5 code is unchanged. The runner now prints complete exception tracebacks to GPU logs. Fifty focused visual/runner/campaign tests pass, including the actual typed-content traversal contract. Restart in `agent_v6_eval_20261018_r1` with a new source snapshot and the identical frozen inputs/selection; retain the failed attempt for audit.

Prelaunch review found the independent flip-audit entry also passed bare text directly to OfflineQwen. Its prompts now use typed text blocks, with a strict processor-contract regression. The prepared `_r1` snapshot was not submitted; retain it as an unused preparation and launch final corrections in `_r2`, preserving snapshot immutability. All benchmark inputs and selections remain identical.

Final processor corrections: **69 focused tests passed** (visual adapter, runner, campaign/race, evaluation/audit). Frozen `_r2` has 569 source files and verifies all original 14 input hashes. Detached login supervisor PID `2061703` on eu-login-11 submitted new smoke race: Blackwell `17457002`, A100 `17457006`, attempt `3919dc93540f44f28bdd3ef58b329d0e`. Corrected results live in `d4rt_agent/results/agent_v6_eval_20261018_r2`. No completed quick-check or gate is claimed.

## Independent recurring monitoring — 2026-10-18

The user requested checks and key metrics every 30 minutes, including these infrastructure failures. Added a separately frozen login-node watcher under `d4rt_agent/results/agent_v6_monitoring`, independent of the experiment supervisor so its crash is detected. The watcher polls health every 30 seconds, deduplicates immediate failure alerts, and queues 30-minute metric checks into this existing Codex session through the supported `codex queue` command. Bootstrap delivery was verified. Six focused watcher tests pass. Configuration points to the current `_r2` plan; repairs must update that pointer, preserve selections/gates and keep a single watcher. Process metadata and heartbeat are recorded in process.json and watcher_state.json. Stop on user request or after final complete/gate-failed notification.

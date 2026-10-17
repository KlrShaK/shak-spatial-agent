# AgentV6 design notes

Design for AgentV6, written while we brainstormed (2026-10-17, branch `feat/agentv6-draft`).
All six themes are settled; nothing is implemented yet. Context files record what we know before
deciding. Theme files record what we decided and why. Each file should be readable on its own.

**Start here:** read the glance below, then [20_implementation_plan.md](20_implementation_plan.md),
which gives the reading order and hands the build to an agent one milestone at a time.

## AgentV6 at a glance

A 4D spatial reasoning agent: local Qwen3-VL-8B plans in physical quantities and calls a small set
of measurement tools whose reports read like V4's. The model chooses; the host measures.

```
captioner (separate Qwen, clip as 2-fps video) → ~50-word caption, no motion words
main agent: system prompt + caption + question + options + 8 keyframes
  1. entities (names in words; camera implicit)
  2. plan: "this question is about …" + ≤4 items tagged
     [camera motion | entity motion | entity and camera | entities and landmarks | counting | visual]
  3. tool calls (≤14), host reminder + category hint after each report; continue · revise · answer
  4. answer in the same conversation: measurement decides 3D questions, frames/ask_video decide
     visual facts; "C: <option> — <what decided it>"
```

| Theme | In one line | File |
|---|---|---|
| Reports | Headline with committed direction words, size and noise floor; merged phase table; reading notes; option-blind | [10](10_tool_reports.md) |
| Tools | `track_camera`, `track_entity(main, others, frame_of_reference)`, `track_orientation`, `track_multiple_entities`, `track_camera_zoom`, `check_scene_revisit`, `detect_all_entities`, `get_segmentation_mask`, `get_frames`, `ask_video`, `choose_segmentation_candidate` | [11](11_toolset.md) |
| Planning | Caption, entities, a tagged plain-text plan, a 14-call budget | [12](12_planning.md) |
| Tool calls | Causes designed away, then normaliser, doctor, grounding retries, facing fallback | [13](13_tool_calls.md) |
| Answering | Same conversation, no added frames, precedence rule, V4 answer format, 32k context | [14](14_answering.md) |
| Evaluation | Quick-check set from previously answered questions, gate, ablations with approval, full runs DSI → Dyn → VLM4D | [15](15_evaluation.md) |
| Assumptions | DA3 metric scale (A1); unmarked directions mean the world frame (A2) | [Assumptions](Assumptions.md) |

## Files

| File | What it holds | Status |
|---|---|---|
| [01_baseline.md](01_baseline.md) | Where V4 and V5 stand on DSI-Bench, Dyn-Bench and VLM4D, with matched comparisons | context |
| [02_v4_v5_anatomy.md](02_v4_v5_anatomy.md) | How V4 and V5 run a question, the code map, and what changed between them | context |
| [03_evidence_so_far.md](03_evidence_so_far.md) | What runs, reviews and pilots have established, including where tool failures come from | context |
| [04_shak_findings.md](04_shak_findings.md) | Your manual V5-versus-V4 review, verbatim | reference |
| [05_discussion_tracker.md](05_discussion_tracker.md) | Every point from 04 with an ID, my first take, evidence and status, grouped into five themes | live |
| [06_question_census.md](06_question_census.md) | All 10,585 questions by physical quantity, with the V6 tool and the evidence on whether measuring helped | context |
| [10_tool_reports.md](10_tool_reports.md) | Theme 1: what a tool returns | settled 2026-10-17 |
| [11_toolset.md](11_toolset.md) | Theme 2: eight measurement tools, `get_frames`, the ask-video and candidate-choice tools, and how question types map onto them | settled 2026-10-17; two names open |
| [12_planning.md](12_planning.md) | Theme 3: inputs, prompt, and the planning flow | settled 2026-10-17 |
| [13_tool_calls.md](13_tool_calls.md) | Theme 4: calling tools reliably (design, normaliser, doctor, perception fallbacks) | settled 2026-10-17 |
| [14_answering.md](14_answering.md) | Theme 5: from measurement to answer (the final-step flip) | settled 2026-10-17 |
| [15_evaluation.md](15_evaluation.md) | Theme 6: comparisons, question sets, gate, ablations, metrics, run hygiene | settled 2026-10-17; quick-check selection to confirm |
| [20_implementation_plan.md](20_implementation_plan.md) | Reading order, build milestones M0–M9, what to reuse, tests, and a prompt to hand each milestone to an agent | live |
| [Assumptions.md](Assumptions.md) | What V6 relies on but cannot check while answering (A1 metric scale, A2 default frame) | live |
| [analysis/](analysis/) | Small scripts behind numbers quoted in these notes | as needed |

## Principles carried over unless we decide otherwise

These have held since v1 or were set by you explicitly. Changing one is a decision for the log.

- The model grounds and chooses; the host measures and computes; ground truth is used only for scoring.
- Local Qwen3-VL-8B (`OfflineQwen`) for every language/vision step, no API model.
- One small, validated decision per model call. Prompts and schemas stay small enough for an 8B model.
- Prefer rules derived from geometry and question wording over thresholds picked by accuracy. All three
  benchmarks may be used freely to find algorithms; no holdout protocol is required.

## Decision log

| # | Date | Decision | Why, in one line | Doc |
|---|---|---|---|---|
| 1 | 2026-10-17 | Every report opens with a natural-language headline that commits to direction words, with motion size and noise floor beside them; then a short table; then reading notes for that report only | V4's format; orients the 8B model while keeping numbers checkable | [10](10_tool_reports.md) |
| 2 | 2026-10-17 | Tools report over the whole clip by default; a window only when the question names a time | The model should not pick timestamps to sample a changing quantity | [10](10_tool_reports.md) |
| 3 | 2026-10-17 | The model sees the options as candidate choices and answers by best match or elimination | Measured "still, then forward" should pick "forward" when it is the closest option | [10](10_tool_reports.md) |
| 4 | 2026-10-17 | A caption of about 50 words, for planning and grounding only *(it stays in context through the answer: #36)*; the scene inventory runs SAM3 on nouns from question, options and caption *(inventory on demand only: #30)* | Finds entities the question does not name; keeps caption priors out of the answer | [05](05_discussion_tracker.md) T11, P8 |
| 5 | 2026-10-17 | Scene revisit stays (it is geometric) under a self-describing name | 71.7% vs frames-only 58.3% on Dyn | [05](05_discussion_tracker.md) T13 |
| 6 | 2026-10-17 | Reports never mention the options; the model does the matching | Host thresholds describe motion but never pick the answer | [10](10_tool_reports.md) |
| 7 | 2026-10-17 | The model sees the options from the planning step onward | Options often name quantities the stem omits (rotation, orbit) | [10](10_tool_reports.md) |
| 8 | 2026-10-17 | Report tables list merged phases; the headline gives the majority motion first, then minority motions with when they happen | Dominance and chronology both stated, without the V5 prompt conflict | [10](10_tool_reports.md) |
| 9 | 2026-10-17 | Full reading contract with a tool's first report in a question; 1–2 short lines afterwards | Keeps V4's key rules (orbit sign) without repeating them | [10](10_tool_reports.md) |
| 10 | 2026-10-17 | The model picks the reference frame as a tool argument (as in V5), with wording cues in each frame's description and the frame printed in every report title | Any alternative also needs the question interpreted; one decision is simplest | [11](11_toolset.md) |
| 11 | 2026-10-17 | *(superseded by #20)* One `relative_track(main, reference)` replaces relative position, pair distance, viewpoint/orbit, region relation and elevation | One tool, named for its purpose, for every two-thing relation | [11](11_toolset.md) |
| 12 | 2026-10-17 | Metric distance and speed are reported without warnings; the DA3 scale is an explicit assumption | If the scale is wrong the agent cannot fix it | [Assumptions](Assumptions.md) A1 |
| 13 | 2026-10-17 | The VLM decides counts, helped by segmented frames and the inventory's count and timeline as hints; `group_track` gives per-instance directions | Host counts scored 43.8% vs the VLM's 65.6–81.2% | [11](11_toolset.md) |
| 14 | 2026-10-17 | *(superseded by #21)* One `track_motion(who, frame)` for the camera and for entities, with a contract per mover | Smaller menu; contracts keep the two report types readable | [11](11_toolset.md) |
| 15 | 2026-10-17 | Two frames: world (default, start view with gravity) and the entity's starting facing, each with a one-line cue and example; camera-view questions go to the camera-relative tools *(the camera frame now lives inside `track_entity`, #20)* | The unmarked default matches Dyn's convention; "from the camera" is a relative-motion question | [11](11_toolset.md), [Assumptions](Assumptions.md) A2 |
| 16 | 2026-10-17 | *(superseded by #20, #23)* Relative motion and orientation are split by reference type: `track_relative_motion_entity_entity`, `track_relative_motion_entity_cam`, `track_orientation_entity_entity`, `track_orientation_entity_cam` (refines #11) | Names must say whether the reference is the camera or another entity | [11](11_toolset.md) |
| 17 | 2026-10-17 | Keep a look tool, named `get_segmentation_mask`: frames at times in seconds, with masks | Needed for visual events, count checks and choosing instances | [11](11_toolset.md) |
| 18 | 2026-10-17 | Names: `detect_all_entities`, `track_camera_zoom`, `check_scene_revisit`, `track_multiple_entities` | Literal names that say what is measured | [11](11_toolset.md) |
| 19 | 2026-10-17 | Camera-only motion and orientation questions go to the camera track, never to an entity–camera tool | Keeps the entity tools about entities | [11](11_toolset.md) |
| 20 | 2026-10-17 | `track_entity(main, others=[], frame_of_reference = world / camera / subject)` is the one entity tool; with no `others`, the camera is the implicit reference | One familiar tool covers own motion, pair relations, observer distance and the camera's side of a subject | [11](11_toolset.md) |
| 21 | 2026-10-17 | `track_camera` is its own tool | `main`/`others` make no sense for the camera; it has its own contract | [11](11_toolset.md) |
| 22 | 2026-10-17 | `track_entity`'s world and subject reports include the entity's own turning, with camera rotation removed | Otherwise a panning camera looks like the subject turning | [11](11_toolset.md) |
| 23 | 2026-10-17 | One `track_orientation(main, frame_of_reference = camera / own / an entity)` | Facing relative to the camera, its own start, or another entity | [11](11_toolset.md) |
| 24 | 2026-10-17 | The frame parameter is always `frame_of_reference`, used in `track_entity`, `track_orientation` and `track_multiple_entities` (where it appears: to confirm, G1) | One name, one meaning, across tools | [11](11_toolset.md) |
| 25 | 2026-10-17 | Frames are described in plain words (`entity_in_cam_frame`), never `A_from_B` transform notation | No convention for us or the 8B model to decode | [11](11_toolset.md) |
| 26 | 2026-10-17 | In the `subject` frame, `main`'s own motion uses its starting facing; where others (or the camera) are relative to `main` uses its facing at each moment | Matches DSI's "starting orientation" wording and V4's c5 bucket | [11](11_toolset.md) |
| 27 | 2026-10-17 | When `others` is given, the camera is left out of `track_entity`'s relative part | Keeps the report focused; ask for the camera with `frame_of_reference=camera` | [11](11_toolset.md) |
| 28 | 2026-10-17 | The caption comes from a separate captioning subagent with no question or planning context; the main agent gets caption + question + options | An unsteered description of the video | [12](12_planning.md) |
| 29 | 2026-10-17 | *(order revised by #37)* Planning flow: what the question is about → which physical quantities → which entities → small plan → tool calls, adapting after each report; one decision per turn | V5's staged decisions, asked in V4's terms of physical quantities | [12](12_planning.md) |
| 30 | 2026-10-17 | `detect_all_entities` runs only when the agent calls it (one Qwen call lists nouns, then SAM3) | The agent decides when it needs the full inventory | [12](12_planning.md) |
| 31 | 2026-10-17 | *(revised by #32)* No frames unless the main agent asks for them (answer turn: theme 5) | Frames on request, as with V5's show/inspect | [12](12_planning.md) |
| 32 | 2026-10-17 | The main agent's first turn shows 8 keyframes automatically (as in V5); more frames and segmentation masks on request | Enough visual context to plan; detail when needed | [12](12_planning.md) |
| 33 | 2026-10-17 | *(revised by #41)* `get_segmentation_mask(entities, times)` is the only frame tool; its description explains its parameters (no entities → plain frames) | One tool, explained by its schema | [11](11_toolset.md) |
| 34 | 2026-10-17 | The caption covers entities, distinguishing attributes, setting and whether the clip is first-person, with no motion directions | Motion words in a caption are the prior that reversed video punishes | [12](12_planning.md) |
| 35 | 2026-10-17 | The captioner gets the clip as a video and Qwen's own sampler decides (2 fps, default budget of ~12k visual tokens); one call per clip, cached | All frames would be near-duplicates squeezed to ~380 px; 2 fps keeps near-full detail and catches briefly visible entities | [12](12_planning.md) |
| 36 | 2026-10-17 | The caption stays in the main agent's context through the answer (revises #4) | With no motion words, it is context rather than a motion prior | [12](12_planning.md) |
| 37 | 2026-10-17 | System prompt opens "You are a 4D spatial reasoning agent with access to various measurement tools …" and mixes V4 (honest reading of measurements) with V5 (generic across benchmarks); step order about → entities → physical quantities and plan (revises #29) | V4's prompt was fitted to DSI; V5's was generic | [12](12_planning.md) |
| 38 | 2026-10-17 | First-person clips: the camera wearer is the observer; "the person" may be the camera → `track_camera` | 113 VLM4D Ego4D questions | [12](12_planning.md) |
| 39 | 2026-10-17 | `get_segmentation_mask` runs on the whole video and reports each object's visible spans and relative size (% of image pixels), besides frames at requested times; it answers on-screen size and visibility questions | 146 apparent-size and 106 visibility questions on Dyn had no tool | [11](11_toolset.md) |
| 40 | 2026-10-17 | The mask table gives size in phases (`6% → 14% → 9%`), an image-edge column, one row per labelled instance; `detect_all_entities` reuses the same table for every noun | Matches what apparent-size and visibility questions ask; one format, one code path | [11](11_toolset.md) |
| 41 | 2026-10-17 | Plain frames come from a separate `get_frames(times ≤ 4, or range)` returning 4 frames; `get_segmentation_mask` always needs entities | Each tool does one thing its name says | [11](11_toolset.md) |
| 42 | 2026-10-17 | Keep `own` in `track_multiple_entities` (one Orient-Anything pass per instance) | Needed for "how many move backwards from their own perspective" | [11](11_toolset.md) |
| 43 | 2026-10-17 | *(superseded by #51)* Turn 1 ("what is the question about") is a single choice from a menu, as in V5 | Mixed questions are handled by the ask-video tool instead of two choices | [12](12_planning.md) |
| 44 | 2026-10-17 | New tool: open questions about the video to a separate VLM agent ("At which time does the skier jump?"); its times let measurement tools answer mixed questions | Events and actions are visual; frames-only Qwen scores 79% on them vs 73% with our measurement | [11](11_toolset.md) |
| 45 | 2026-10-17 | The agent orchestrates its own tool calls (writes them); how the plan is written is open | Your PB answer | [12](12_planning.md) |
| 46 | 2026-10-17 | Budget: at most 4 plan items and 14 tool calls, a warning on each of the last 3 calls, early answer always allowed | Room for multi-step questions without open-ended loops | [12](12_planning.md) |
| 47 | 2026-10-17 | When revising, the agent may call any tool directly; the doctor catches malformed calls | Flexibility to adapt to reports (P12) | [12](12_planning.md) |
| 48 | 2026-10-17 | Entities are named in words and resolved by the host; with several SAM3 candidates the agent sees one image with each candidate in its own colour and picks one, through a choice tool shown only then | No IDs to get wrong; one image instead of several crops | [11](11_toolset.md) |
| 49 | 2026-10-17 | The plan is plain text (numbered items), not JSON; the host only reads it to show reminders | Chain of thought grounds the next calls; JSON plans failed repeatedly | [12](12_planning.md) |
| 50 | 2026-10-17 | Plan items name quantities as thinking, not tool calls; actual measurements are chosen while calling tools, and the host reminds the agent to follow or rewrite its plan | Plans should adapt to what the reports say | [12](12_planning.md) |
| 51 | 2026-10-17 | Each plan item carries one of six category tags (camera motion · entity motion · entity and camera · entities and landmarks · counting · visual), which selects the hint shown with it; replaces the "about" menu turn (supersedes #43) | A menu answered once served no purpose; per-item tags drive per-step hints | [12](12_planning.md) |
| 52 | 2026-10-17 | For visual items the agent may pass the benchmark question and options to `ask_video` | Frames-only Qwen scores 79% on visual Dyn questions, V5 71% | [12](12_planning.md) |
| 53 | 2026-10-17 | Grounding: after a first SAM3 miss the host asks for an alternative singular, short, descriptive name; after a second miss it suggests `get_frames` or `ask_video` | Failed detection is not proof of absence | [11](11_toolset.md) |
| 54 | 2026-10-17 | When OAV2 finds no front, a VLM is shown a crop and returns one of 8 facing labels with greedy decoding, queried 3 times over the clip or at given times | Facing is the top unavailability reason in V4 and V5 | [11](11_toolset.md) |
| 55 | 2026-10-17 | Names `ask_video` and `choose_segmentation_candidate`; the latter is available only right after a call that asked to choose an entity | Literal names; no tool that cannot be used | [11](11_toolset.md) |
| 56 | 2026-10-17 | Motion never depends on orientation: without facing, `track_entity` still reports world- and camera-frame motion and marks only facing-dependent lines unavailable | Facing is inherently harder; motion must survive its failure | [11](11_toolset.md) |
| 57 | 2026-10-17 | The plan opens with one free-text line, "this question is about …", before the tagged items | A framing sentence at no cost | [12](12_planning.md) |
| 58 | 2026-10-17 | The facing fallback's eight labels are camera-view wording (facing the camera, facing away, facing image-left/right, and the diagonals) | Unambiguous for the VLM; the host converts them to a heading | [11](11_toolset.md) |
| 59 | 2026-10-17 | A deterministic normaliser fixes calls first; the doctor handles only what it cannot | Most V5 refusals were mechanical | [13](13_tool_calls.md) |
| 60 | 2026-10-17 | The doctor is the local Qwen3-VL-8B in a separate short text-only conversation | Local-Qwen rule; small context | [13](13_tool_calls.md) |
| 61 | 2026-10-17 | The doctor may fill missing or invalid values, never change valid ones; if a filled value must change, it sends the question back to the agent; one attempt per call; the agent is told what was corrected | The doctor must not quietly change what the agent meant | [13](13_tool_calls.md) |
| 62 | 2026-10-17 | A corrected call counts once towards the 14; doctor runs are free, capped at 5 per question | Repairs should not eat the measurement budget | [13](13_tool_calls.md) |
| 63 | 2026-10-17 | Malformed final answers: deterministic letter extraction first, then the doctor; never invent a letter | V5 lost 24 VLM4D questions to answer format | [13](13_tool_calls.md) |
| 64 | 2026-10-17 | The answer happens in the same conversation: no fresh packet, no re-sent dossier | Compositional V5's fresh text packets lost the fallback; re-sent frames invited flips | [14](14_answering.md) |
| 65 | 2026-10-17 | No frames are added at the answer; only frames the agent requested (the 8 keyframes stay in context) | Performance drops as visual input grows; measurements often decide | [14](14_answering.md) |
| 66 | 2026-10-17 | Precedence: the measurement decides 3D motion, direction, distance and facing unless unavailable or shown to be on the wrong object; frames and `ask_video` decide visual facts; replaces V5's "rely on the frames" rule | Frame impressions of 3D motion are priors (control 58.5% → 25.0% on reversed DSI) | [14](14_answering.md) |
| 67 | 2026-10-17 | Answer format: letter, option text, what decided it; `{"letter", "because"}`; placeholder, never an example letter | V4's form; removes the A-bias suspect; makes flips measurable | [14](14_answering.md) |
| 68 | 2026-10-17 | Context budget 32k tokens; drop the oldest requested images first, always keep reports and the plan | V5 exceeded 16k at 18–20k | [14](14_answering.md) |
| 69 | 2026-10-17 | No new control runs: compare with published Qwen numbers and our existing prediction sets on identical IDs (VLM4D's paper has no Qwen3-VL-8B row) | Controls exist or are published | [15](15_evaluation.md) |
| 70 | 2026-10-17 | A 200–300-question quick-check set from all three benchmarks, drawn from questions previous systems already answered | Answer-by-answer comparison with V4/V5, including their failures | [15](15_evaluation.md) |
| 71 | 2026-10-17 | Full runs in order: DSI 200 × 4 (V4's draw), Dyn 2,800 split, VLM4D random 1,000 | Every run pairs with existing predictions | [15](15_evaluation.md) |
| 72 | 2026-10-17 | Gate to full runs: beat V5 and frames-only Qwen on the quick-check set with a positive paired lower bound; no quantity group >5 points below the best existing system; DSI reversed renderings not below V4 | Format success alone promoted V5's failed workflow | [15](15_evaluation.md) |
| 73 | 2026-10-17 | V6 evaluation runs may race A100 and Blackwell: keep the first to start, cancel the other at once (project exception to the no-race rule) | Faster starts | [15](15_evaluation.md) |
| 74 | 2026-10-17 | Ablations one at a time on the quick-check set, cheapest first; each needs your approval before submission | Isolate what helps; you control GPU spend | [15](15_evaluation.md) |
| 75 | 2026-10-17 | Every run reports accuracy (per benchmark, quantity group, official breakdown), process metrics (unavailable, doctor, plan adherence, flips, letters) and cost | Accuracy alone hid V5's failures | [15](15_evaluation.md) |
| 76 | 2026-10-17 | Frozen source snapshot and config hash per run; one answer JSON per question; caches packed per clip | Reproducibility; V5 hit the file quota | [15](15_evaluation.md) |
| 77 | 2026-10-17 | `get_frames` is the final name of the plain-frame tool (#41) | Says what it does | [11](11_toolset.md) |
| 78 | 2026-10-17 | The 2-job limit is for GPU jobs only (CPU-only jobs exempt); no heavy work on login nodes; Monitor every running test, eval or job and update you every 30 minutes | Your working rules for runs; also in your global CLAUDE.md | [15](15_evaluation.md) |
| 79 | 2026-10-17 | Tool visibility is limited by phase only: no calls in the entity and plan turns (the plan turn sees the tool list), all tools in act turns, `choose_segmentation_candidate` only after an ambiguity image, only the final answer at the end; no gating by plan category | V5's kind-scoped menus locked the model out of the right tool after a wrong routing choice | [12](12_planning.md) |

## Open items

- **Quick-check selection** (#70): pools and counts proposed in [15](15_evaluation.md), awaiting your OK.
- **Assumptions A1 and A2** are checked by the first full runs ([Assumptions](Assumptions.md)).
- **Backbone** (Point4D, D4RT, or both): never decided; recommendation in [20](20_implementation_plan.md).
- **Commit the design docs** before the implementation worktree is created.

# 20 — Implementation plan

How to turn the design into code without handing an agent everything at once. The design is about
80 decisions across six theme files; no implementing agent should read all of it for one task.
Each milestone below names the few sections and decision numbers it needs, the V4/V5 code to reuse,
what to build, GPU-free tests, and when it is done.

## Reading order

**To understand the agent** (about 30 minutes):

1. [README](README.md), "AgentV6 at a glance": the flow and the tools in one screen.
2. [12_planning](12_planning.md): what happens turn by turn.
3. [11_toolset](11_toolset.md): the tools and how question types map onto them.
4. [10_tool_reports](10_tool_reports.md): what every tool returns.
5. [14_answering](14_answering.md), then [13_tool_calls](13_tool_calls.md), then
   [15_evaluation](15_evaluation.md).
6. [Assumptions](Assumptions.md).

Context files 01–03 and 06 explain *why* a decision was made. Open them only when a decision's
reason matters. 04 and 05 are the historical record of our discussion.

**To build:** follow the milestones below, in order.

## Before M0: three things to settle

| Item | Status | Needed by |
|---|---|---|
| **Commit the design docs** on `feat/agentv6-draft`, so the implementation worktree contains them | waiting for your OK | M0 |
| **Backbone**: never decided. My recommendation: Point4D by default through V5's backbone interface, since A1's metric scale assumes Point4D × DA3 and Dyn showed a tie (58.8 vs 58.1). Keep D4RT selectable, and compare both on the DSI part of the quick-check set as ablation 8, because V4's DSI win used D4RT and V5's camera gap on DSI (45% vs 78%) is confounded with the backbone. | to decide | M2 |
| **Quick-check selection** (#70) | waiting for your OK | M8 |

## Where the code goes

A new package beside this folder: `d4rt_agent/agent_v6/`, built in a separate worktree on a new
branch (`git worktree add ../shak-spatial-agent-v6 -b feat/agent-v6 feat/agentv6-draft`). The main
checkout is never switched. V4 and V5 code is imported, never edited.

```
d4rt_agent/agent_v6/
  reports.py        report record, phases, headline, renderer                      (M1)
  workspace.py      clip, scene cache, entities by name, instance labels, seconds  (M2)
  tools/            camera.py (M3) · entity.py, orientation.py, group.py (M4)
                    inventory.py, frames.py, ask_video.py, captioner.py (M5)
  calls.py          schemas, normaliser, doctor, answer recovery                   (M6)
  agent.py          phases, plan, reminders, budget, answer                        (M7)
  prompts/          system, entities, plan, hints, answer, contracts, captioner,
                    ask_video, doctor, facing fallback                             (M1–M7)
  run.py            CLI, benchmark loaders, resume, snapshot                       (M0, M8)
  evaluation/       quick-check selection, metrics, paired comparison, flip audit  (M8)
  test_*.py         GPU-free tests, colocated
```

## Milestones

| | Milestone | Depends on | GPU |
|---|---|---|---|
| M0 | Scaffold and runner | — | no |
| M1 | Report layer | M0 | no |
| M2 | Workspace and geometry access | M0, backbone decision | no (tests); GPU later |
| M3 | Camera tools | M1, M2 | no |
| M4 | Entity, orientation and group tools | M1, M2 | no |
| M5 | Inventory, frames and VLM helpers | M1, M2 | no (VLM mocked) |
| M6 | Tool-call robustness | M3–M5 | no (doctor mocked) |
| M7 | Controller | M6 | no (scripted Qwen) |
| M8 | Evaluation harness | M7, quick-check OK | no |
| M9 | GPU runs | M8, your approval per run | yes |

M3, M4 and M5 are independent and can be done in any order.

### M0 — Scaffold and runner

- **Read:** README glance; [15](15_evaluation.md) "Run hygiene" (#76, #78).
- **Reuse:** `agent_v5/run.py` (`dynbench_entries`, `dsi_entries`, `vlm4d_entries`, `select`,
  `open_clip`, `validate_resume`), `agent_v5/clips.py`, `agent_v5/research/snapshot.py`.
- **Build:** the package skeleton; `run.py` with a `--dry-run` that lists the questions without
  loading models; the answer-file schema (one JSON per question).
- **Tests:** loader counts (DSI std 1,769; Dyn split 2,800; VLM4D 1,816); dry-run writes nothing but
  answer stubs.
- **Done when:** `python -m d4rt_agent.agent_v6.run --bench dsi --dry-run --limit 5` works for all
  three benchmarks.

### M1 — Report layer

- **Read:** [10](10_tool_reports.md) in full; decisions #1, #2, #6, #8, #9.
- **Reuse:** `sam_orientany_d4rt_test/traj3d_buckets.py` (`rank_components`, `segment_report`,
  `_segment_prose`), `agent_v5/render.py` (`table`, `named_axes`), `agent_v4/prompts/contract_*.md`
  as the first full contracts.
- **Build:** the report record (title, headline, phase rows, reading notes full and short, status,
  raw data), phase segmentation, majority-then-minority headlines, units and time labels, and the
  unavailable form.
- **Tests:** the synthetic bank from 10's "To check": still-then-forward, early-left-then-right,
  out-and-back, jittery-but-static → one "still" phase; row cap; seconds labels; reports never
  contain option text.
- **Done when:** every test passes and the example in 10 renders exactly from synthetic data.

### M2 — Workspace and geometry access

- **Read:** [11](11_toolset.md) implementation notes; [13](13_tool_calls.md) "Layer 1"; #20, #41,
  #48, #56; [Assumptions](Assumptions.md) A1.
- **Reuse:** `agent_v5/backbones/base.py` (`get_backbone`, `SceneCache`),
  `backbones/point4d.py`, `backbones/d4rt.py`, `agent_v5/models.py` (`Models`, `ground_phrase`),
  `agent_v5/scene.py` (`metric_scale`, `smooth_track`, `merge_fragments`, `lift_instances`),
  `agent_v5/orientation.py`.
- **Build:** one workspace per clip. Entities are resolved by name, with stable instance labels
  ("car 2"). Caches are revisioned, so re-grounding invalidates dependants. Seconds map to frames,
  with labels of shown frames when the fps is unknown. The camera is always available. The metric
  scale is recorded with its source.
- **Tests:** name and label resolution; re-grounding clears dependent caches; seconds mapping
  including unknown fps; no generator-frame integer ever appears in rendered text.
- **Done when:** the tools of M3–M5 can be written against the workspace alone.

### M3 — Camera tools

- **Read:** 11, rows for `track_camera`, `track_camera_zoom`, `check_scene_revisit`; #19, #21;
  [03](03_evidence_so_far.md) §3 (verified defects).
- **Reuse:** `agent_v5/measure.py` (`camera_state`, revisit and zoom measures);
  `traj3d_buckets.camera_pose_angles`, `level_basis`; V4's `contract_traj_camera_own_frame.md`.
- **Tests:** a pure roll registers motion; out-and-back is not "static"; left translation with a
  right pan reads clockwise per the contract; zoom versus dolly are told apart.

### M4 — Entity, orientation and group tools

- **Read:** 11, decisions and implementation notes; #20, #22, #23, #26, #27, #42, #54, #56, #58;
  [Assumptions](Assumptions.md) A2.
- **Reuse:** `agent_v5/capabilities.py` (`position`, `frame_vectors`, `relative_position`,
  `group_geometry`), `agent_v5/orientation.py` (`subject_basis`, `camera_bearing`, `sector`),
  `agent_v5/spatial_tools.track_quality`.
- **Tests:** two entities swapping sides at constant distance read as opposite sides; main's own
  motion uses its starting facing and others use its facing at each moment (#26); without facing,
  world-frame motion is still reported (#56); a full turn reads 360°; group counts by direction; the
  facing fallback (mocked VLM) maps 8 labels to 45° bins.

### M5 — Inventory, frames and VLM helpers

- **Read:** 11, rows for `detect_all_entities`, `get_segmentation_mask`, `get_frames`, `ask_video`,
  `choose_segmentation_candidate`; [12](12_planning.md) "Captioner input"; #28, #30, #34, #35,
  #39–41, #44, #48, #52, #55.
- **Reuse:** `agent_v5/final_context.render_frame` (mask overlays), `agent_v5/capabilities.py`
  (`census`, `image_series`, `view_state`), `agent_v5/models.ground_phrase`.
- **Tests:** mask table from synthetic masks (spans, size phases, edge column, labels); `get_frames`
  returns 4 frames for a range; the caption is cached per clip and contains no motion words (prompt
  check); the candidate image gives each candidate a distinct colour.

### M6 — Tool-call robustness

- **Read:** [13](13_tool_calls.md) in full; #53, #59–63.
- **Reuse:** `simple_v2.extract_first_action`.
- **Tests:** replay V5's refused calls listed in 03 §6 (`c1#0`, a name instead of an ID,
  generator-frame windows, `cam`, inspection handles). Each must be normalised or sent back with a
  clear message. The doctor (mocked) never changes a valid value (#61). A corrected call counts once
  (#62). Answer recovery never invents a letter (#63).

### M7 — Controller

- **Read:** [12](12_planning.md) and [14](14_answering.md) in full; #37, #38, #46, #47, #49–51, #57,
  #64–68, #79.
- **Reuse:** `agent_v5/context.Conversation` (budget and trimming), `simple_v2.OfflineQwen`.
- **Build:** phases with tool visibility (#79); entity turn; plan turn (opening line, tagged items,
  numbered lines read for reminders); act loop with reminders, category hints, 14 calls and warnings;
  answer turn with the precedence rule and V4 format; system prompt.
- **Tests:** scripted-Qwen replays of the six worked examples in 12, each producing the expected
  call sequence and a valid answer; trimming never drops reports, the plan or the keyframes;
  warnings on calls 12–14.

### M8 — Evaluation harness

- **Read:** [15](15_evaluation.md) in full; #69–76.
- **Reuse:** `agent_v5/research/protocol.py` (`prediction_rows`, `clip_key`, `paired_compare`),
  `agent_v5/report.py`.
- **Build:** quick-check selection (after your OK), metrics per #75, paired comparisons against
  existing predictions with DSI renderings kept together, the flip audit, the gate check (#72).
- **Tests:** selection is deterministic and answer-blind; pairing clusters by source clip.

### M9 — GPU runs (each needs your approval)

Smoke run (about 10 questions per benchmark) → quick-check run → gate → ablations, one at a time →
full runs: DSI 200 × 4, then Dyn 2,800, then VLM4D 1,000. GPU queue at most 2 running and 1 pending;
the A100/Blackwell race is allowed (#73); every running job is watched with Monitor, with an update
to you every 30 minutes (#78).

## How to hand a milestone to an agent

One milestone per session. Start the agent in plan mode, so it proposes files, functions and tests
before writing code, and approve the plan before it builds. Paste this, filled in:

```text
Implement AgentV6 milestone M<k> (<name>) in the worktree ../shak-spatial-agent-v6,
branch feat/agent-v6.

Read first, and only these: d4rt_agent/agent_v6/design/README.md ("AgentV6 at a glance"),
then <files and sections from the milestone's "Read" line>. The decision numbers in them
(#N) are binding.

Reuse, by importing (never edit agent_v4/ or agent_v5/): <the milestone's "Reuse" line>.
Build: <the milestone's "Build" line>.
Tests: GPU-free pytest in d4rt_agent/agent_v6/test_*.py covering <the milestone's "Tests"
line>. Run them as a CPU job: sbatch scripts/run_cpu_task.slurm -m pytest d4rt_agent/agent_v6 -q
— not on the login node.

Rules: follow CLAUDE.md and CONTRIBUTING.md (commits "[type]agent-v6: …", about 150–200
lines each, code separate from docs, no agent attribution). Local Qwen3-VL-8B only. No GPU
jobs. If the design is ambiguous or contradicts the code, stop and ask; never change a
numbered decision silently. Record implementation choices under "Implementation notes" in
the relevant design file.

Start in plan mode and wait for my approval. Done when: <the milestone's "Done when" line,
or all its tests pass>.
```

Between milestones, review the diff (the `/code-review` skill helps), then start the next one in a
fresh session with the next prompt.

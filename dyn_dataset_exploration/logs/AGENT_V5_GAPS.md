# What AgentV5 needs that AgentV4 does not have

Hand analysis over `TAXONOMY_REPORT.md` and `BUCKET_REQUIREMENTS_V5.md`. Not generated —
a `taxonomy_report.py` rerun will not overwrite it.

`BUCKET_REQUIREMENTS_V5.md` says what each new measurement needs. This file says what has
to change in `d4rt_agent/` for any of it to be reachable, and in what order.

---

## The one-paragraph version

AgentV4 scores 60.4% mean on DSI-Bench with four measurement buckets. Those four buckets
cover **100 of 600** sampled Dyn-Bench questions — 17%. The other 83% ask for something
else: 442 name a quantity AgentV4 does not produce, 52 need no measurement at all, and 6
fit no bucket we could name. The single sharpest number is that
`traj_subject_own_frame` — the bucket both DSI-Bench `cate` 0 and `cate` 1 mapped onto, the
one the whole Orient-Anything-V2 stage exists to serve — collects **10 of 600** questions
here. Dyn-Bench does not ask where an object went in its own axes. It asks where it went
relative to the camera, relative to the scene, or relative to another object.

So AgentV5 is not AgentV4 plus a few builders. Four things above the builder layer have to
change first, and they are what the rest of this document is about.

---

## 1. The state machine holds one subject. Dyn-Bench names two.

**Evidence:** `subject_subject_relation` 56/600 (est. 864 rows, 12%) plus
`traj_subject_scene_frame` 53/600 (est. 773, 11%) — 109 of 600 sampled questions, and the
two largest new buckets, both need a second grounded entity. *"Between the start and the
end, how does the person's position relative to the lamp change?"*

**Where it is blocked.** Not in the geometry — two subjects queried from the same D4RT
encoding of the same clip already share a frame at every timestamp, so there is no
registration problem. It is blocked three layers up:

- `agent_v4/contracts.py` — `ACTION_SCHEMAS` has one `name_subject` action taking one
  `subject` string.
- `agent_v4/orchestrator.py` — `_loop` holds a single `expected` action name and rejects
  anything else; the sequence `choose_bucket → name_subject → [choose_candidate] →
  final_answer` has exactly one subject slot, and `_advance` returns one
  `(next_expected, done)` per step.
- `agent_v4/pipeline.py` — `QuestionPipeline` caches per `(video, subject)`. This layer is
  already correct: the cache is keyed exactly right for holding two entries per clip. It is
  the state machine above that never asks for a second.

**The change.** Let `Bucket` declare how many subjects it needs (`needs_subject: bool`
becomes a count, or a `subjects: tuple[str, ...]` of role names like
`("subject", "reference")`), and let the orchestrator loop the naming step once per role.
`choose_candidate` already handles disambiguation per grounding and needs only to be
scoped to the role being named. `subject_key()` and the per-`(video, subject)` cache carry
over unchanged.

**Why it is first.** It is the largest single block of uncovered questions, it unblocks two
buckets rather than one, and every other item on this list is smaller if the orchestrator
is already general over roles.

---

## 2. There is no time axis. 492 rows name a second.

**Evidence:** the `timestamped` family is 492 rows as a primary label, 610 counting
questions that match it alongside another family. *"Between 0.0s and 2.17s, how does the
person's position relative to the snowy slope change?"* · *"At the 5-second mark, what is
the approximate distance between the person and the small animal?"* These cut across six
buckets, so this is one change that partially unlocks all of them.

**Where it is blocked.**

- Nothing in `traj3d_*` knows what a second is. Reports are indexed in frames.
- `traj3d_buckets.uniform_time_segments` interpolates between `frames[0]` and `frames[-1]`
  — the whole visible span, with no sub-window argument. `checkpoint_frames` likewise
  returns first/middle/last of the whole visibility mask.
- AgentV4 samples 32 frames (`NUM_FRAMES = 32` in `traj3d_query`, `NUM_SAMPLED_FRAMES` in
  `simple_v2_contracts`, and D4RT's `OpenD4RT_32CLIP` checkpoint is a 32-clip model).
  Dyn-Bench's own protocol samples 16 (`frame_sampling.json`), and its clips run from 9 to
  3199 frames (median 50). A question about second 5.33 of a 3199-frame PointOdyssey clip
  is asking about a frame a uniform 32-frame sample almost certainly skipped.

**The change.** Two parts, and the second is the hard one.

1. *Cheap:* carry frames-to-seconds through. `logs/index.json` already holds
   `total_frames`, `frame_indices` and the resolved frame filenames per clip — everything
   needed. Give the bucket builders a `window: tuple[float, float] | None` in seconds and
   have `uniform_time_segments`/`checkpoint_frames` honour it.
2. *Hard:* decide what to do when the named second is not in the 32-frame sample. Either
   resample the clip around the question's window (cheap, but the agent then sees different
   frames than it reasoned about), or query D4RT at native frame indices inside a window
   (correct, and what `Point4D`'s chunked long-range tracking is for — see §5).

---

## 3. Everything is scale-relative. 428 rows want metres.

**Evidence:** `metric_distance` (est. 225) + `metric_speed` (est. 203) = est. 428 rows, 6%.
Options like `A. 5 meters / B. 10 meters / C. 15 meters / D. 20 meters` and
`A. 0.5 m/s / B. 1.2 m/s / C. 2.1 m/s / D. 3.0 m/s`.

**Where it is blocked.** D4RT normalises each clip by its own mean depth
(`traj3d_geometry.cloud_scale`), and every threshold in `traj3d_buckets.py` is deliberately
a multiple of `scene_scale` because of it. There is no route from the current pipeline to
an absolute unit — not a missing feature, a property of the representation.

**The change is a decision, not code.** Three options, in
`BUCKET_REQUIREMENTS_V5.md` §12: decline outright; anchor on a category size prior
(adult ≈1.7 m, sedan ≈4.5 m); or add a metric monocular-depth model. Dyn-Bench's metric
options are usually spaced roughly 2× apart, which a size prior can often resolve and a
declination never can.

**The recommendation** is the size-prior anchor, with the scale source and its uncertainty
reported alongside the number so the agent can decline when the prior cannot separate two
adjacent options. What must not happen is reporting a scale-relative number as if it were
metres: `traj3d_buckets`' module docstring already states "distances are D4RT units —
scale-relative, comparable only within one clip", and any metric bucket that does not carry
its own scale provenance breaks that invariant silently.

---

## 4. Several large buckets are not trajectory problems at all.

AgentV4's whole design is "ground a subject, track it in 3D, describe the path". Four of
the new buckets are not that shape, and three of them are nearly free because the numbers
already exist and are being discarded.

| bucket | est. rows | what it actually needs | already on disk? |
| --- | --- | --- | --- |
| `camera_motion_profile` | 650 (9%) | the **residual** that `running_mean` removes — shake *is* the high-frequency component | yes: `displacement_noise` and `angle_noise_deg` compute it in order to suppress it |
| `subject_visibility` | 321 (5%) | D4RT visibility logits + mask border contact | yes: `d4rt/subject.npz`, already sigmoided against `VISIBILITY_THRESHOLD = 0.5` |
| `apparent_size_change` | 190 (3%) | SAM3 mask area per frame, normalised | yes: `masks/masks.npz`; no D4RT query needed at all |
| `camera_zoom_vs_dolly` | 238 (3%) | per-frame focal length vs. camera forward translation | nearly: `estimate_intrinsics` is called once on `camera["xyz_3d"][0]`; indexing `[t]` in a loop gives the series |
| `instance_count` | 216 (3%) | detections *associated*, not ranked | partly: `pipeline._anchor_candidates` finds them, then keeps the top `MAX_CANDIDATES_SHOWN = 4` for disambiguation and discards the rest |
| `scene_revisit` | 164 (2%) | frame-pair appearance similarity | no — and it is image retrieval, not geometry |
| `event_order` | 234 (3%) | two onsets from two *different* signals, compared | no, and see below |

`apparent_size_change` deserves a note out of proportion to its 3%: it needs no D4RT query,
so it is the natural thing to fall back to when the pipeline declines, and it is the
cheapest bucket in the whole set to build.

`event_order` is the one that does not fit the architecture rather than merely being
unbuilt. *"Which happens first: the camera begins a rightward pan, or the person in the
dark jacket enters the scene?"* needs a camera-side measurement **and** an object-side
measurement, compared. AgentV4's contract is one bucket per question. This is the strongest
argument for AgentV5 letting `choose_bucket` return more than one.

---

## 5. Two model-level changes this analysis argues for

**`semantic_only` — a route, not a measurement.** Est. 495 rows (7%) are decided by what is
happening, not by geometry: *"What best describes the main activity in the video?"*
AgentV4's `choose_bucket` offers four geometric buckets and no way to decline, so today it
picks one and the host runs SAM3 + D4RT + OAV2 for a question the pipeline cannot inform.
Adding `semantic_only` to the bucket list costs one entry and one contract file, saves the
GPU work on 7% of the benchmark, and removes a source of confidently-wrong measurements.
The wording already exists — it is what `orchestrator._deliver` says on its three
`available: False` paths.

**Point4D.** `third-party-tools/Point4D` was added as a submodule and has **zero code
references**; the only mention in the repo is the last line of `v5-agent_draft.md`. A venv
exists at `/cluster/work/igp_psr/spanwar/envs/point4d`. Two Dyn-Bench facts make it more
than a nice-to-have: clips run to 3199 frames against D4RT's 32-frame window, and
`subject_visibility` (est. 321 rows) is explicitly about objects leaving and re-entering
the frame — the exact case a single-chunk tracker cannot follow. If §2's hard half is
attempted, this is the mechanism for it.

---

## 6. Below the agent: the loader

`d4rt_agent/dsi_bench_data.py` cannot read Dyn-Bench, and the reasons are not incidental —
five datasets vs eight, six `cate` values vs three coarse task labels, `"A: a; B: b"` vs
`["A. a", "B. b"]`, one `metadatas/*.csv` vs two parquets. `dyn_dataset_exploration/
dyn_bench_data.py` already does the whole job and was written to be promoted to
`d4rt_agent/dyn_bench_data.py` unchanged: same public shape (`load_rows`, `build_manifest`,
`write_manifest`/`read_manifest`, `format_question_prompt`, `file_sha256`, `video_slug`),
so `run.py` and the judge port across.

Two things it already handles that are easy to get wrong:

- **Media naming.** The eight sources agree on nothing. Frames are `frame_0000.jpg`
  (DAVIS, DynPose, SA-V, YTVIS), `00001.jpg` (Spring, Total-Recon), `rgb_00000.jpg`
  (PointOdyssey) and `046ac8-3_obj_source_left-0000.png` (DynamicReplica); SA-V's mp4s drop
  the `_manual` suffix the parquet keeps and gain a `sav_` prefix; DynamicReplica ships
  masks every fifth frame against 300 RGB frames. `dyn_bench_data.build_index` resolves all
  1000 clips, with `mask_frame_stride` stated per clip rather than assumed.
- **Answer hygiene.** Four of the 7000 `answer` values ship lowercase. `load_rows`
  upper-cases them; `stats.json` carries the structural-issue list so nothing is fixed
  silently.

---

## 7. Two findings that are about the benchmark, not the agent

Both affect how any Dyn-Bench number should be read, including ours.

**The shipped `task` column is not the paper's level taxonomy.** Blind classification lands
in the hypothesised level for 89% of `cameraqa`, 74% of `qa`, and **48% of `sceneqa`** —
most of `sceneqa`'s remainder reads as inter-object perception. Grouping on the shipped
`task` column and calling the result a per-level score does not reproduce the paper's
levels.

**The option text leaks the answer.** Across all 7000 rows the correct answer is the
longest option 38.7% of the time against a 25% baseline — and in `sceneqa` it is **64.7%**.
`A` is correct 38.9% of the time overall. Both are properties of the shipped text, measured
by a single pass over the parquet in `dyn_bench_data.build_stats`, with no model involved.
They bound how much of this benchmark is a vision task, and `sceneqa` is the subset where
that bound bites hardest.

---

## 8. Suggested order

1. **Two-subject orchestrator** (§1) — largest block of uncovered questions, unblocks
   `subject_subject_relation` and `traj_subject_scene_frame` together, and makes everything
   after it smaller.
2. **`traj_subject_camera_frame`** (`BUCKET_REQUIREMENTS_V5.md` §3) — the data is already on
   disk in the right frame; a builder and a contract file, and close to nothing else.
3. **The three discard-and-recover buckets** — `camera_motion_profile`,
   `subject_visibility`, `apparent_size_change` (est. 1161 rows between them, 17%). All
   three read numbers the pipeline already computes and throws away.
4. **`semantic_only`** (§5) — one bucket entry and one contract; stops 7% of GPU work.
5. **Time axis, cheap half** (§2.1) — unblocks part of six buckets.
6. **Metric scale decision** (§3) — decide before building, and record the decision.
7. Everything else, by estimated rows.

---

## 9. What this pass did not look at

The **3000 dynamic-object-grounding rows** (`cameramask` / `objmask` / `scenemask`, scored
by J&F over per-frame masks) were out of scope by agreement and are the largest untouched
capability gap. AgentV4 cannot address them at all: `final_answer`'s `kind` must be
`"text"`, so there is no action through which the agent can emit geometry. The ground truth
is on disk and dense (`masks/<dataset>/<clip>/frame_*_obj*.png`, 1–9 objects per clip, 413
clips with one object and 587 with more), so the evaluation is ready whenever the output
modality exists.

No agent was run and no accuracy was measured on Dyn-Bench in this pass. Every "AgentV4
cannot do X" above is read off the code, not off a failed run.

# 11 — The toolset

Theme 2 of the discussion. It covers tracker points T2–T16, P7 and P11. Status: **settled** on
2026-10-17 (decision log #5, #10, #12, #13, #15, #17–27, #39–44, #48, #52–56, #58).

## Principle

Accuracy comes from a few tools used well, not from many tools. V4 had 4 buckets and won on DSI.
V5 had 23 kinds, never beat frames-only Qwen on Dyn, and lost on DSI. The D7 study shows where
measuring helps at all. It clearly helps for camera motion, zoom and view revisits. It gives no
gain for elevation, event order and landmark relations, and it hurts for pair relations, metric
speed and counting (tracker, theme 2 table; [06](06_question_census.md) by quantity). V6 keeps
eight measurement tools, a frame tool (`get_frames`) and a question tool for visual facts (#44),
and hands the rest to the VLM.

## The toolset

| Tool | Arguments | Reports (whole clip unless the question names a time) |
|---|---|---|
| `detect_all_entities` | called by the model when it wants the full inventory (#30) | One Qwen call lists the nouns in question, options and caption; then the `get_segmentation_mask` table for every noun, plus instance counts per category and segmented frames with instance labels (#40) |
| `track_camera` | — | The camera's own motion in its start view (gravity-levelled): translation and pan/tilt/roll per phase, tempo, jitter |
| `track_entity` | `main`, `others=[]`, `frame_of_reference` = `world` / `camera` / `subject` | **world:** paths of `main` (and `others`) in the start view, plus each one's own turning. **camera:** as the moving camera sees them: towards/away, left/right in the view, distance to the camera. **subject:** `main`'s own motion in its *starting* facing, plus where each of `others` is relative to `main` in main's facing *at each moment* (#26). If `others` is empty, that relative part is about the camera; if `others` is given, the camera is left out (#27). |
| `track_orientation` | `main`, `frame_of_reference` = `camera` / `own` / an entity | **camera:** which side `main` shows the camera over time. **own:** `main`'s turning relative to its starting facing, with camera motion removed. **entity:** `main`'s facing relative to that entity. |
| `track_multiple_entities` | `category`, `frame_of_reference` = `world` / `camera` / `own` (#42) | Every instance of the category: direction of each, spread over time, counts by direction |
| `track_camera_zoom` | — | Focal ratio and forward motion |
| `check_scene_revisit` | — | How much of the start view is seen again at the end |
| `get_segmentation_mask` | `entities` (required), `times` (seconds, optional) | Runs on the whole video (#39). A table with one row per labelled instance: visible spans, size as a share of the image in phases (`6% → 14% → 9%`), and when it touches the image edge (#40). With `times`, also up to 8 frames at those times with the masks; without, the table only. Entity names come from the planning step, or from `detect_all_entities` first. |
| `get_frames` | `times` (up to 4, seconds) **or** `range` (start, end in seconds) | 4 plain frames: at the given times, or sampled uniformly over the range, each labelled with its time (#41) |
| `ask_video` | an open question in words, e.g. "At which time does the skier jump?"; for visual questions it may carry the benchmark question and options (#52) | A separate VLM agent answers from the clip, with times in seconds (#44, #55) |
| `choose_segmentation_candidate` | the colour of the intended instance | Available only right after a call that asked the agent to choose an entity: one image with each SAM3 candidate in its own colour; the agent picks one (#48, #55) |
| VLM only | — | The counting decision; anything no tool reports |

### How the benchmarks' question types map

| Question type | Call |
|---|---|
| DSI c0/c1: subject motion "relative to its starting orientation" | `track_entity(runner, frame_of_reference=subject)` |
| DSI c2/c3: observer motion | `track_camera` |
| DSI c4: observer–subject distance | `track_entity(character, frame_of_reference=camera)` |
| DSI c5: observer's position relative to the subject's orientation | `track_entity(man, frame_of_reference=subject)` (camera implicit) |
| Dyn "which direction is the bird flying towards" | `track_entity(bird, frame_of_reference=world)` |
| Dyn pair relation: "does the dog move closer to the person" | `track_entity(dog, others=[person], frame_of_reference=subject)` |
| VLM4D "from the camera perspective, which way do the bears move" | `track_entity(bear, frame_of_reference=camera)`, or `track_multiple_entities(bear, camera)` for several |
| VLM4D "is the bear facing away from the camera" | `track_orientation(bear, frame_of_reference=camera)` |
| VLM4D "how many bears move backwards from their own perspective" (6 such questions) | `track_multiple_entities(bear, frame_of_reference=own)`, then the VLM counts with segmented frames (#42) |
| Dyn "does the camera pan left before the skier jumps?" | `ask_video`: "At which time does the skier jump?", then `track_camera` for the pan's onset (#44) |
| Dyn view revisit, zoom | `check_scene_revisit`, `track_camera_zoom` |

## Decisions

**`track_entity` is the one entity tool (#20).** It has a main entity, optional others, and a
`frame_of_reference`. It replaces the four reference-named relative-motion and orientation tools
of #16 and the separate relative tools before them (#11). The camera is always an implicit
reference: with no `others`, the relative part of the report is about the camera.

**`track_camera` is its own tool (#21).** It replaces #14's single `track_motion(who, …)`. `main`
and `others` make no sense for the camera, and the camera's report has its own contract (orbit sign
rule, gravity versus camera axes). Camera-only questions always go here (#19).

**Own turning is part of the entity's report (#22).** The `world` and `subject` reports of
`track_entity` include a heading line with camera rotation removed, as V4 did ("heading +3°, turning
to its right"). Without this, a panning camera looks like the subject turning.

**One `track_orientation(main, frame_of_reference)` (#23)** for facing relative to the camera, to
its own start, or to another entity.

**The parameter is always called `frame_of_reference` (#24).** It appears in `track_entity`,
`track_orientation` and `track_multiple_entities`. It does not appear elsewhere:
`detect_all_entities` and `get_segmentation_mask` work in the image, and `track_camera`, the zoom
and the revisit check are defined in the camera's own start view. Each value's description carries
a one-line cue and example (#15):

- *world:* the question names no frame ("Which direction is the bird flying towards?").
- *subject / own:* the question names the subject's starting position or own perspective ("in
  which direction is the runner moving relative to his starting position?").
- *camera:* the question names the camera or viewer ("from the camera perspective", "towards the
  camera").

**Frames are named in plain words (#25).** Reports and schemas say what is expressed in which frame
with plain phrases such as "the camera, in the man's own frame" or `entity_in_cam_frame`, not with
`A_from_B` transform notation. Nothing has to be decoded by convention.

Earlier decisions still in force: the model chooses the frame (#10); metric quantities reported
without warnings (#12, [Assumptions.md](Assumptions.md) A1); the unspecified-direction default is
the world frame (#15, A2); the VLM decides counts (#13); `get_segmentation_mask` (#17, #39, #40);
plain frames through `get_frames` (#41); the names of #18.

## Implementation notes

- **Relative geometry in `track_entity`.** For each other (or the camera):
  r(t) = B_main(t)ᵀ (p_other(t) − p_main(t)), where B_main is main's basis (OAV2 facing plus
  gravity). Report distance |r|, bearing (0 = in front of main, +90 = its right), vertical angle, and
  the phases of side changes, in the theme 1 format. `camera` frame: B is the camera rotation at
  each time. `world` frame: B is the start view's levelled basis.
- **Heading line.** OAV2 yaw lifted into the world frame with camera rotation removed (V4:
  `OAV2_TO_OPENCV`, then crop-position and camera-rotation correction in `traj3d_analyze.py`).
  Unwrapped and accumulated, so a full turn reads as 360°, not 0°.
- **`track_multiple_entities`.** Instances come from `detect_all_entities`. Per instance: net
  direction in the chosen frame with a noise floor. `own` runs one OAV2 pass per instance at its
  first visible frames (V4: median 2.4 s per subject for a full pass); instances with no resolvable
  front are reported as such and left out of the own-frame counts (#42). Group spread is the median pairwise distance over time, with
  membership changes reported separately. Example headline: "4 of 6 fish move RIGHT in the camera's
  view; 1 left; 1 still".
- **`get_segmentation_mask` table (#39, #40).** Agreed format:

  ```
  visible      object   size (% of image)     at image edge
  0.0–2.2 s    car 1    6% → 14% → 9%         right edge from 1.9 s
  1.5–3.0 s    person    8% → 8%              —
  ```

  Size phases follow the theme 1 rules (merge runs, noise floor), so "larger", "smaller",
  "unchanged" and "fluctuating" can be read directly; Dyn's 130 apparent-size questions ask exactly
  that. The edge column separates "fully visible" from "partially visible: cut off" in Dyn's
  visibility questions. Occlusion stays the VLM's call from the frames. Instance labels are shared
  with the mask overlays and accepted by other tools (`track_entity("car 2", …)`). SAM3 video
  propagation already covers the whole clip, so the table costs no extra model call.
- **`get_frames` (#41).** Times are mapped to the nearest frame; a range gives 4 frames evenly spaced
  from start to end, inclusive. No masks, no measurement.
- **`detect_all_entities`.** Called on demand (#30). Same code path and table as
  `get_segmentation_mask` (#40). SAM3 on each noun once per clip, cached. Other
  tools ground lazily: SAM3 on an entity's name the first time a tool needs it, shared through the
  same cache. Segmented frames for counting: up to 8 frames at peak instance occupancy, stable
  instance labels.
- **Removed tools.** Event, predicate and elevation tools are gone from the menu. Geometric
  transitions (starts or stops moving, enters or leaves view) appear as lines in track and
  inventory reports.

- **`ask_video` (#44, #52, #55).** A separate Qwen3-VL-8B conversation that sees the clip as video
  at 2 fps, as the captioner does (#35). It receives only what the agent writes. For visual
  questions the agent may pass the benchmark question and options (#52); otherwise it asks a
  narrower fact. It answers in at most two sentences, with times in seconds when asked. Its
  description says it is for visual facts (events, actions, appearance, when something happens);
  3D motion, distance and facing come from the measurement tools.
- **`choose_segmentation_candidate` (#48, #55).** When SAM3 returns several candidates for a name,
  the host renders one frame with each candidate's mask in a distinct colour and a legend. The tool
  is in the menu only on the turn right after that image.
- **Grounding retries (#53).** First SAM3 miss → the host replies: "No object found with the name
  <name>. Suggest an alternative name: singular, short, descriptive words." Second miss → "Still not
  found. This is not proof that it is absent: check with `get_frames` or `ask_video`."
- **Facing fallback (#54).** When OAV2 resolves no front on its first try, a VLM fallback runs: it
  is shown a crop of the subject and returns one of eight camera-view labels (#58): facing the
  camera, facing away, facing image-left, facing image-right, and the four diagonals (e.g. facing
  the camera and image-left), with deterministic (greedy) decoding. It is queried 3 times, at the start, middle and end of the clip or at given times. Each
  label maps to a 45° heading bin, which feeds the same pipeline as OAV2 facing (lifted to the world
  frame, camera rotation removed). Reports name the source: "facing from the VLM fallback, coarse
  to ±22.5°".
- **Motion never depends on orientation (#56).** Motion and orientation stay separate because facing
  is inherently harder to estimate. If facing is unavailable even after the fallback,
  `track_entity` still reports motion in the world and camera frames, and marks only its heading
  line and subject-frame part unavailable. `track_multiple_entities(own)` lists instances without a
  resolvable front separately.

